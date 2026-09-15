"""Fit generated wording to the fixed PDF design before it can be cached."""
import asyncio
import json
import logging

from app.services.cv_tailor import (
    TailorCVResult, TailoringError, _build_prompt, _request_tailored_payload,
    _result_from_payload, _validate_tailored_payload, has_reframed_experience,
)
from app.services.pdf_generator import CVOverflowError, build_ats_pdf
from app.config import get_settings


async def fit_tailored_cv(result, master, title, company, description):
    sections = json.loads(master.sections_json)
    prompt, experience, skills = _build_prompt(sections, title, company, description)
    settings = get_settings()
    summary_required = any("summary" in section["name"].lower() for section in sections)
    last_error = None
    for attempt in range(4):
        try:
            if result.used_fallback or not has_reframed_experience(result.text, sections):
                raise TailoringError("CV must contain complete model-generated experience sections.")
            pdf = await asyncio.to_thread(build_ats_pdf, result.text)
            return result, pdf
        except (CVOverflowError, TailoringError) as exc:
            last_error = exc
            logging.warning("CV fit check %s failed: %s", attempt + 1, exc)
        if attempt == 3:
            break
        # Only wording may change. Never scale the body, squeeze spacing, delete
        # a bullet, or truncate PDF content to make a successful-looking download.
        word_budget = (22, 18, 14)[attempt]
        request = (
            prompt
            + "\n\nThe candidate below did not pass the fixed one-page layout/rewrite check. "
            + str(last_error)
            + "\nReturn the complete JSON with more concise, job-specific wording. "
            + "Preserve ALL employer groups and the EXACT bullet and skills-category counts. "
            + "Keep the factual outcomes and metrics. Reframe every experience sentence, "
              "not just its opening verb. Do not alter headings, dates, contact details, or education. "
            + f"Use at most {word_budget} words per experience bullet and 35 words for the summary. "
              "Condense skills lists to the most relevant tools; avoid repeated keywords. "
              "Do not output formatting or layout changes.\nCurrent candidate (content to edit):\n"
            + json.dumps(result.text)
        )
        try:
            payload = await _request_tailored_payload(settings, request, 0.3)
            _validate_tailored_payload(payload, summary_required, experience, skills,
                                       reject_unchanged_categories=False)
            result = _result_from_payload(sections, payload, cacheable=True, target_job_title=title)
        except TailoringError as exc:
            last_error = exc
            logging.warning("CV wording revision %s rejected: %s", attempt + 1, exc)
    raise TailoringError(
        "Could not produce a fully reframed one-page CV after three wording revisions. "
        "The fixed design and saved CV were preserved. Please retry."
    ) from last_error
