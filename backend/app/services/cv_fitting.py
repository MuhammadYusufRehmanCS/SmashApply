"""Fit generated wording to the fixed PDF design before it can be cached."""
import asyncio
import json
import logging
import math

from app.services.cv_tailor import (
    TailorCVResult, TailoringError, LLMExecutionError, _TailoredPayload, _build_prompt, _request_tailored_payload,
    _result_from_payload, _validate_tailored_payload, has_reframed_experience,
)
from app.services.pdf_generator import CVOverflowError, build_ats_pdf
from app.config import get_settings
from app.services.cv_wording import request_wording_replacements


def _wording_limits(measurements: dict, previous: dict | None = None) -> dict:
    """Allocate actual printed lines, not a uniform word count, to editable fields."""
    fields = measurements['fields']
    previous = previous or {}
    lines = {f['path']: min(f['lines'], previous.get(f['path'], {}).get('lines', f['lines']))
             for f in fields}
    available = measurements['available_height'] - measurements['fixed_height'] - 3
    # A multi-page PDF can also result from print fragmentation. Always reclaim
    # at least one rendered line even if continuous-flow height looked adequate.
    if fields:
        available = min(available, sum(f['lines'] * f['line_height'] for f in fields)
                        - max(f['line_height'] for f in fields))
    def height():
        return sum(lines[f['path']] * f['line_height'] for f in fields)

    # Shorten lists before prose; preserve as many two-line achievements as fit.
    ordered = sorted(fields, key=lambda f: (
        0 if f['path'].startswith('technical_expertise') else 1 if f['path'] == 'summary' else 2,
        f['characters'],
    ))
    for field in ordered:
        while height() > available and lines[field['path']] > 1:
            lines[field['path']] -= 1
    if height() > available:
        raise TailoringError('The fixed CV sections leave insufficient room for one line per required field.')

    limits = {}
    for field in fields:
        path = field['path']
        # Reserve width for wrapping at word boundaries and bold glyphs. Actual
        # Chromium/PDF rendering remains the final authority, not this estimate.
        capacity = (field['width'] * lines[path] - field['prefix_width'])
        max_chars = math.floor(0.82 * capacity / max(field['average_char_width'], 1))
        old = previous.get(path)
        if old:
            max_chars = min(max_chars, old['max_characters'])
            if field['lines'] > lines[path]:
                max_chars = min(max_chars, math.floor(old['max_characters'] * 0.8))
        limits[path] = {'lines': lines[path], 'max_characters': max(1, max_chars)}
    return limits


def _field_value(payload: dict, path: str):
    value = payload
    for part in path.split('.'):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def _set_field(payload: dict, path: str, value: str):
    parts = path.split('.')
    parent = payload
    for part in parts[:-1]:
        parent = parent[int(part)] if isinstance(parent, list) else parent[part]
    parent[int(parts[-1]) if isinstance(parent, list) else parts[-1]] = value


def _candidate_payload(result) -> dict:
    context = result.template_data
    return dict(role_title=context['role_title'], keywords=result.keywords,
                summary=context['summary'], technical_expertise=context['technical_expertise'],
                experience_bullets=context['experience_bullets'])


async def fit_tailored_cv(result, master, title, company, description):
    sections = json.loads(master.sections_json)
    prompt, experience, skills = _build_prompt(sections, title, company, description)
    settings = get_settings()
    summary_required = any("summary" in section["name"].lower() for section in sections)
    last_error = None
    candidate_changed = True
    limits = {}
    measurements = None
    for attempt in range(4):
        try:
            if candidate_changed:
                if result.used_fallback or not has_reframed_experience(result.text, sections):
                    raise TailoringError("CV must contain complete model-generated experience sections.")
                pdf = await asyncio.to_thread(build_ats_pdf, result.text)
                return result, pdf
        except (CVOverflowError, TailoringError) as exc:
            last_error = exc
            if isinstance(exc, CVOverflowError) and exc.measurements:
                measurements = exc.measurements
                limits = _wording_limits(measurements, limits)
                logging.info('Measured CV fit: %.1fpx content / %.1fpx available; target %s lines',
                             measurements['content_height'], measurements['available_height'],
                             sum(limit['lines'] for limit in limits.values()))
            logging.warning("CV fit check %s failed: %s", attempt + 1, exc)
        if attempt == 3:
            break
        candidate_changed = False
        # Only wording may change. Never scale the body, squeeze spacing, delete
        # a bullet, or truncate PDF content to make a successful-looking download.
        current = _candidate_payload(result)
        rewrite_paths = [f['path'] for f in measurements['fields']
                         if f['lines'] > limits[f['path']]['lines']] if measurements else []
        if measurements and not rewrite_paths:
            # Account for print fragmentation/rounding even if continuous-flow
            # measurements predict a fit: request shorter editable prose.
            rewrite_paths = list(limits)
        if not limits:
            # Structural errors and older/custom renderers may have no metrics.
            word_budget = (22, 18, 14)[attempt]
            request = (
                prompt
                + "\n\nThe candidate below did not pass the fixed one-page layout/rewrite check. "
                + str(last_error)
                + "\nReturn the complete JSON with more concise, job-specific wording. "
                + "Preserve ALL employer groups and the EXACT bullet and skills-category counts. "
                + "Retain the JD-specific scope required by SYSTEM_PROMPT. Reframe every experience sentence, "
                  "not just its opening verb. Preserve action/domain scope, architectural implementation, "
                  "and operational impact in EVERY bullet while shortening wording. Condense repeated "
                  "skills and summary text before sacrificing the engineering narrative. "
                  "Do not alter headings, dates, contact details, or education. "
                + f"Use at most {word_budget} words per experience bullet and 35 words for the summary. "
                  "Condense skills lists to the most relevant tools; avoid repeated keywords. "
                  "Do not output formatting or layout changes.\nCurrent candidate (content to edit):\n"
                + json.dumps(result.text)
            )
        try:
            if limits:
                replacements = await request_wording_replacements(settings, {
                    path: {'text': _field_value(current, path),
                           'max_characters': limits[path]['max_characters'],
                           **({'category_label': skills[int(path.split('.')[1])]['label']}
                              if path.startswith('technical_expertise.') and skills else {})}
                    for path in rewrite_paths
                }, title)
                for path in rewrite_paths:
                    _set_field(current, path, replacements[path])
                payload = _TailoredPayload.model_validate(current)
            else:
                payload = await _request_tailored_payload(settings, request)
            _validate_tailored_payload(payload, summary_required, experience, skills,
                                       reject_unchanged_categories=False)
            result = _result_from_payload(sections, payload, cacheable=True, target_job_title=title)
            candidate_changed = True
        except LLMExecutionError:
            # An unavailable provider cannot be fixed by a shorter writing prompt.
            raise
        except TailoringError as exc:
            last_error = exc
            logging.warning("CV wording revision %s rejected: %s", attempt + 1, exc)
    raise TailoringError(
        "Could not produce a fully reframed one-page CV after three wording revisions. "
        "The fixed design and saved CV were preserved. Please retry."
    ) from last_error
