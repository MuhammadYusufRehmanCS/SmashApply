"""Rewrites a Master CV's Executive Summary, Technical Expertise, and
Professional Experience bullets to mirror a target job description, via a
local Ollama model. Every other section of the CV -- the Header/contact
line, Education, Certifications, Additional, and anything else -- is
reproduced byte-for-byte from the master CV and is never sent to, or
returned by, the model.

Anti-hallucination design: rather than relying on prompt instructions alone
to stop the model from inventing/altering employers, dates, titles, degrees,
or the surrounding document structure, those facts are structurally kept out
of the model's hands entirely --

  - The Professional Experience section is split (in Python, before any
    Ollama call) into per-employer entries. Only each entry's BULLET TEXT is
    sent to the model; the entry's title/company/location/dates line is
    never sent and never comes back from the model -- it's spliced back in
    verbatim when reconstructing the tailored CV.
  - Technical Expertise is split the same way, into per-category entries.
    Only each category's tool-list TEXT is sent; the category LABEL (e.g.
    "Cloud & Infrastructure:") is immutable and spliced back in verbatim --
    the model sees it for context (so it tailors the right kind of tools
    into the right category) but never has to echo, and can never alter, it.
  - Any section that couldn't be confidently split this way (an atypically
    formatted resume), or that isn't Summary/Skills/Experience at all
    (Education, Certifications, Additional, the Header, ...), is passthrough,
    always, unconditionally -- reproduced byte-for-byte from the master CV.
  - Ollama is called with `format: "json"`, and the response is parsed and
    shape-validated (Pydantic) before it's allowed anywhere near the PDF
    renderer. A count mismatch for a given employer's bullets, or for a given
    category's items, falls back to that employer's/category's ORIGINAL
    content rather than risking a corrupted or fabricated line, and anything
    that looks like a leaked schema key (e.g. a bare "experience_bullets"-
    shaped string standing in for real content) is dropped outright. A
    response that fails to parse/validate at all raises, which the caller
    turns into a clean error instead of ever building a PDF from bad data.

Works for ANY scraped role -- DevOps, Platform Engineering, Cloud Engineering,
SRE, Data Engineering, etc. -- by having the model dynamically infer the role
category and required tools from that specific job's title/company/description
rather than assuming a fixed role type.
"""
import json
import re
from typing import Annotated

import httpx
from pydantic import BaseModel, BeforeValidator, Field, ValidationError

from app.config import get_settings
from app.models import MasterCV
from app.services.text_sections import BULLET_PREFIXES, is_bullet_line, looks_like_entry_header, strip_bullet

_SUMMARY_HINTS = ("summary", "objective", "profile")
_SKILLS_HINTS = ("skill", "expertise", "competenc", "tools", "technolog")
_EXPERIENCE_HINTS = ("experience", "employment", "work history")


def _is_summary_section(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _SUMMARY_HINTS)


def _is_skills_section(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _SKILLS_HINTS)


def _is_experience_section(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _EXPERIENCE_HINTS)


def _split_experience_entries(content: str) -> list[dict] | None:
    """Splits a Professional Experience section's raw text into per-employer
    entries: {"header_line", "tagline", "bullets"}. header_line/tagline are
    the immutable title/company/location/dates line (and the optional
    one-line role blurb some resumes put before the bullets); bullets are
    the ONLY part that ever reaches the model.

    Returns None if no entry-header-shaped line was found at all -- an
    atypically-formatted section this heuristic can't confidently split.
    Callers must treat None as "don't touch this section" rather than
    guessing, since a wrong split could scramble which bullets belong to
    which employer.
    """
    lines = [ln for ln in content.splitlines() if ln.strip()]
    entries: list[dict] = []
    current: dict | None = None

    for line in lines:
        if looks_like_entry_header(line):
            current = {"header_line": line.strip(), "tagline": None, "bullets": []}
            entries.append(current)
            continue
        if current is None:
            # Content before any recognized entry header -- the format isn't
            # what this heuristic expects, so don't guess at a split.
            return None
        if is_bullet_line(line):
            current["bullets"].append(strip_bullet(line))
        elif current["tagline"] is None and not current["bullets"]:
            current["tagline"] = line.strip()
        else:
            current["bullets"].append(line.strip())

    return entries or None


# "Cloud & Infrastructure: AWS, Azure, ..." -- label is everything before the
# first colon. Deliberately requires the label to be short (a category name,
# not a whole sentence that happens to contain a colon somewhere).
_CATEGORY_LINE_RE = re.compile(r"^(.{2,60}?):\s*(.+)$")


def _split_technical_expertise(content: str) -> list[dict] | None:
    """Splits a Technical Expertise section into per-category entries:
    {"prefix", "label", "items"}. prefix+label are the immutable bullet
    marker and category name; items is the ONLY part that ever reaches the
    model.

    A category's item list is often word-wrapped across several physical
    lines by the PDF's own text extraction (long tool lists commonly spill
    onto a second or third line) -- a non-bulleted line is treated as a
    continuation of the PREVIOUS category's items, not a new category.

    Returns None if a line can't be attributed to any category at all (no
    bulleted "Label: items" line has been seen yet, or a bulleted line
    doesn't contain a colon) -- an atypically-formatted section this
    heuristic can't confidently split, in which case the whole section
    stays untouched passthrough.
    """
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    entries: list[dict] = []
    for line in lines:
        if is_bullet_line(line):
            prefix = next((c for c in line if c in BULLET_PREFIXES), "-")
            match = _CATEGORY_LINE_RE.match(strip_bullet(line))
            if not match:
                return None
            label, items = match.groups()
            entries.append({"prefix": prefix, "label": label.strip(), "items": items.strip()})
        elif entries:
            entries[-1]["items"] = f"{entries[-1]['items']} {line}".strip()
        else:
            # Content before any recognized category line -- don't guess.
            return None
    return entries or None


# The model's own schema key names -- if any of these show up as a bare
# bullet string (e.g. the model emitted a stray 5th list item that's just
# "experience_bullets" instead of real bullet text, which is exactly the bug
# this guards against), it's unambiguously not real resume content.
_SCHEMA_KEY_NAMES = {
    "keywords",
    "summary",
    "executive_summary",
    "technical_expertise",
    "technical_expertise_items",
    "experience_bullets",
    "bullets",
    "items",
}
_BARE_IDENTIFIER_RE = re.compile(r"^[a-z][a-z_]{2,}$")


def _looks_like_leaked_key(text: str) -> bool:
    """True for anything that isn't real resume prose: a bare schema-key
    token, or (a real observed failure mode) a stringified JSON/Python list
    or dict leaking through as a field's literal text value -- e.g. the
    model returning "['bullet 1', 'bullet 2']" as a plain string for a field
    that should just be prose, rather than (or in addition to) the coercion
    in `_coerce_list_to_str` catching it as an actual JSON array. Used for
    every field the model returns: the summary, each Technical Expertise
    category's items, and every bullet."""
    raw_stripped = text.strip()
    if raw_stripped.startswith(("[", "{")) and raw_stripped.endswith(("]", "}")):
        return True
    stripped = raw_stripped.strip('"').strip().lower()
    if stripped in _SCHEMA_KEY_NAMES:
        return True
    # A real bullet is always prose -- spaces, capitals, punctuation. A bare
    # lowercase_and_underscores-only token with no spaces is unambiguously
    # not a sentence, regardless of whether it happens to match a key name.
    return bool(_BARE_IDENTIFIER_RE.fullmatch(stripped))


def _coerce_list_to_str(separator: str):
    """A small local model asked for a string field quite reasonably
    sometimes returns a JSON list instead (one item per sentence/tool/
    phrase) -- both are semantically fine, so accept either shape rather
    than failing validation over it. The right join separator depends on
    what the field represents: prose paragraphs read naturally joined by
    newlines, a tool list by ", ", and a single bullet's fragments by a
    plain space (it should read as one continuous sentence)."""

    def _coerce(value: object) -> object:
        if isinstance(value, list):
            return separator.join(str(item) for item in value)
        return value

    return _coerce


class _TailoredPayload(BaseModel):
    keywords: list[str] = Field(default_factory=list)
    summary: Annotated[str, BeforeValidator(_coerce_list_to_str("\n"))] = ""
    # One reworded items-string per Technical Expertise category, same order
    # as the categories this job was built from.
    technical_expertise: list[Annotated[str, BeforeValidator(_coerce_list_to_str(", "))]] = Field(
        default_factory=list
    )
    # One inner list of reworded bullets per employer, same order as the
    # employer entries this job was built from.
    experience_bullets: list[list[Annotated[str, BeforeValidator(_coerce_list_to_str(" "))]]] = Field(
        default_factory=list
    )


class TailoringError(RuntimeError):
    """Raised when Ollama can't be reached, or its response can't be parsed
    into a valid, safe tailored CV. Callers must never build a PDF (or
    persist anything) from a request that raised this -- there is no
    "fall back to the job description" or "render whatever text we got"
    path; the caller is expected to surface a clean failure instead."""


SYSTEM_PROMPT = """You are an expert ATS resume strategist for infrastructure, cloud, and data \
careers -- DevOps, Platform Engineering, Site Reliability Engineering, Cloud Engineering, Data/\
Database Engineering, and adjacent roles. You will be given a specific target job (its title, \
hiring company, and full job description), the candidate's existing summary, existing Technical \
Expertise categories, and (for each employer) that employer's EXISTING bullet points. Your job is \
to aggressively rewrite, reframe, and optimize this content so the candidate reads as a top-tier, \
purpose-built match for THAT exact job -- while never inventing a fact, and never adding or \
removing content beyond rewording what already exists.

You are NOT given, and must NEVER write, an employer name, a job title, a location, or a date \
range -- those are handled entirely outside of you, elsewhere in the document. If you don't have a \
piece of information, you have no way to get it wrong -- so never reference or guess at one.

Internal reasoning (for your own thinking only -- never write it into your output):
- Determine the actual role category this specific posting is for (it may be DevOps, Platform \
Engineering, Cloud Engineering, SRE, Data Engineering, or something else entirely -- infer it \
fresh from this job's title and description, never assume).
- Extract the specific tools, languages, cloud platforms, and methodologies THIS job description \
actually names or clearly implies -- only ones actually implied by THIS job description, not a \
generic list reused across postings. These become "keywords".

What you may change:
- "summary": reframe it to mirror this role's exact tech stack and priorities, explicitly \
positioning the candidate as an ideal fit for THIS role at THIS company, naturally working in the \
3-6 most important keywords you extracted above. This is the ONLY field where you may write \
sentences that don't already exist in the candidate's resume -- but every skill/tool you name here \
must still be something the candidate's resume already lists elsewhere (in the summary, the \
Technical Expertise categories, or a bullet); never invent a skill the candidate doesn't have.
- "technical_expertise": for EACH category you're given (identified only by its label, shown to you \
for context), reorder and rephrase that category's EXISTING tool list so the tools most relevant to \
this job are foregrounded first. You may only surface tools the candidate already lists in that \
category or elsewhere -- never introduce a tool the candidate doesn't have.
- "experience_bullets": for EACH employer you're given, reword that employer's EXISTING bullets' \
phrasing and emphasis around this job's tools/keywords wherever it genuinely, truthfully applies. \
You are rewording sentences that already exist -- never author a new bullet from the job \
description's own text, and never copy/paraphrase the job posting's responsibilities \
("you will...", "collaborate with...") into a bullet, since those describe what the EMPLOYER \
wants, not what the candidate already did. Never introduce a tool/skill the bullet doesn't already \
mention.

Hard rules -- read carefully, these are non-negotiable:
- Every employer's "experience_bullets" entry MUST contain EXACTLY the same number of bullets as \
that employer's original bullet list you were given -- same order, one reworded bullet per \
original bullet. Never merge two bullets into one, never split one into two, never add a closing/\
summary bullet, never drop one.
- "technical_expertise" MUST contain exactly one string per category you were given, in the same \
order -- never add a category, never remove one, never merge two categories together.
- Emphasis formatting: wrap 2-4 of the most scannable keywords/tools/metrics PER bullet (and, in \
"technical_expertise", per category) in **double asterisks** (e.g. "Automated infrastructure using \
**Terraform**, reducing manual deployment steps by **80%**"), the way a well-formatted, ATS-safe \
resume bolds the phrases a recruiter's eye should catch first. Bold short phrases only (a tool \
name, a platform, a percentage/metric, a 2-4 word impact phrase) -- never bold an entire sentence. \
Bold words that are ALREADY part of the existing text -- do NOT grow a bullet by tacking a new \
trailing clause onto the end just to have something to bold (never append ", focusing on **X** and \
**Y**" or similar after the original sentence). Every bullet's sentence structure and length should \
stay close to the original -- only word choice, ordering, and what gets bolded should change.
- Do NOT add meta-commentary, notes, disclaimers, or placeholders anywhere, whether as their own \
bullet/field OR appended to the end of one in parentheses -- for example never write "Note:", \
"[unchanged]", "(reworded to emphasize X)", "(mirrors original bullet)", or similar. A bullet or \
category value is ONLY the resume text itself.
- Respond with ONLY a single JSON object, no other text before or after it, no markdown code \
fences, no extra keys beyond the four below, matching EXACTLY this shape:
{"keywords": ["...", "..."], "summary": "...", \
"technical_expertise": ["category 1 tool list", "category 2 tool list"], \
"experience_bullets": [["bullet 1 for employer 1", "bullet 2 for employer 1"], \
["bullet 1 for employer 2"]]}
- Every string inside "technical_expertise" and "experience_bullets" must be actual resume prose --
never a bare word, a field name, or anything that isn't real resume text.
"""


# The model sometimes obeys "reword the bullet" but then appends a parenthetical
# explaining what it did, e.g. "...MTTR. (rewords existing bullet to highlight
# Terraform)". Despite explicit prompt instructions not to, this happens often
# enough on smaller local models that a regex safety net is needed -- strips a
# trailing parenthetical only when it reads as commentary about the edit itself
# (not a legitimate resume parenthetical like "(AWS, Azure, GCP)" or "(2023-2025)").
_INLINE_META_PAREN_RE = re.compile(
    r"\s*\((?:[^()]*\b(?:reword\w*|emphasiz\w*|highlight\w*|mirror\w*|tailor\w*|"
    r"existing\s+bullet|original\s+bullet|keyword\s+emphasis|align\w*\s+with|"
    r"per\s+the\s+job|for\s+this\s+(?:role|job|position))\b[^()]*)\)\s*$",
    re.IGNORECASE,
)

_META_PHRASES = (
    "as an ai language model",
    "as an ai, i",
    "i cannot provide",
    "i can't provide",
    "here is the tailored",
    "here's the tailored",
    "note:",
    "[unchanged]",
    "remains unchanged",
    "same as original",
)

# The system prompt requires every bullet to use a leading "-" (and
# explicitly says to convert any other bullet glyph like "•" to it), but
# smaller local models don't always comply -- and separately, Ollama
# occasionally emits U+FFFD (the Unicode replacement character) in place of
# the bullet glyph entirely, a tokenizer artifact since nothing in this
# pipeline itself performs a lossy decode. Both cases are unambiguously "this
# line is a bullet using the wrong marker" -- normalize them to a clean "-"
# before the text reaches ReportLab or the API, rather than leaving a stray
# glyph (or a literal replacement-character box) in the resume/PDF/preview.
# "*" is deliberately excluded: a line can legitimately start with "**" for
# bold markdown (e.g. a bolded Technical Expertise category label), and
# treating that as a bullet marker would eat the opening ** and break the
# bold span instead of just fixing a bullet.
_NON_DASH_BULLET_CHARS = "•◦▪●‣·�"
_LEADING_BULLET_RE = re.compile(rf"^(\s*)[{_NON_DASH_BULLET_CHARS}]+\s*")


def _normalize_bullet_marker(line: str) -> str:
    return _LEADING_BULLET_RE.sub(r"\1- ", line)


def _clean_field_text(text: str) -> str:
    """Strips stray markdown code fences and LLM conversational notes from a
    single returned string field (a bullet, the summary, ...)."""
    lines = []
    for line in (text or "").splitlines():
        line = _normalize_bullet_marker(line)
        stripped = line.strip()
        if not stripped:
            lines.append(line)
            continue
        if stripped.startswith("```"):
            continue
        lowered = stripped.lower()
        if any(phrase in lowered for phrase in _META_PHRASES):
            continue
        line = _INLINE_META_PAREN_RE.sub("", line).rstrip()
        while True:
            stripped_line = _INLINE_META_PAREN_RE.sub("", line)
            if stripped_line == line:
                break
            line = stripped_line
        lines.append(line)
    return "\n".join(lines).replace("```", "").strip()


def _render_experience_entries(entries: list[dict], tailored_bullets: list[list[str]]) -> str:
    blocks = []
    for i, entry in enumerate(entries):
        lines = [entry["header_line"]]
        if entry["tagline"]:
            lines.append(entry["tagline"])

        original_bullets = entry["bullets"]
        reworded_raw = tailored_bullets[i] if i < len(tailored_bullets) else []
        # Drop anything that's unambiguously not a real bullet (a leaked
        # schema key like "experience_bullets") before even comparing
        # counts -- a dropped item naturally causes a count mismatch below,
        # which is exactly what should trigger falling back to the originals.
        reworded = [b for b in reworded_raw if not _looks_like_leaked_key(b)]
        # Only trust the model's reworded bullets for this employer if the
        # count matches exactly -- otherwise fall back to the untouched
        # originals rather than risk a merged/dropped/fabricated bullet.
        bullets = reworded if len(reworded) == len(original_bullets) else original_bullets

        for bullet in bullets:
            cleaned = _clean_field_text(bullet).strip()
            if cleaned:
                lines.append(f"- {cleaned}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _strip_stray_category_prefix(candidate: str) -> str:
    """A real observed failure mode: the model prefixes its rewording with a
    category-label-shaped "Label: " fragment -- sometimes its own category's
    label, sometimes (apparent cross-contamination between categories in the
    same generation) a DIFFERENT category's. The items text should never
    start with a label at all, since the real label is always supplied
    separately on reconstruction -- so unwrap one if present rather than
    either keeping a visibly wrong "Label: Label: real content" duplicate or
    discarding an otherwise-good rewording over it."""
    match = _CATEGORY_LINE_RE.match(candidate)
    return match.group(2).strip() if match else candidate


def _is_valid_category_rewording(candidate: str, entry: dict) -> bool:
    """Guards against a real observed failure mode: the model echoing the
    category LABEL back as if it were the items ("Cloud & Infrastructure:
    Cloud & Infrastructure"), silently destroying that category's actual
    tool list. Neither `_looks_like_leaked_key` (candidate isn't a bare
    schema-key token) nor the count check used for bullets (there's no count
    here, it's one string) catches this, so it needs its own check."""
    if _looks_like_leaked_key(candidate):
        return False
    if candidate.strip().lower() == entry["label"].strip().lower():
        return False
    # A genuine rewording of a tool list shouldn't collapse to a fraction of
    # the original's length -- that's a sign of truncation or the model
    # substituting something degenerate rather than actually rephrasing.
    if len(candidate) < 0.4 * len(entry["items"]):
        return False
    return True


def _render_technical_expertise(entries: list[dict], tailored_items: list[str]) -> str:
    lines = []
    for i, entry in enumerate(entries):
        items = entry["items"]
        # Per-category fallback: only trust the model's rewording for THIS
        # category if it was actually returned and passes validation --
        # otherwise keep that one category's original items rather than
        # discarding the whole section over one bad entry.
        if i < len(tailored_items):
            candidate = _strip_stray_category_prefix(_clean_field_text(tailored_items[i]).strip())
            if candidate and _is_valid_category_rewording(candidate, entry):
                items = candidate
        lines.append(f"{entry['prefix']} {entry['label']}: {items}")
    return "\n".join(lines)


def _reconstruct_tailored_text(sections: list[dict], payload: _TailoredPayload) -> str:
    experience_idx = next((i for i, s in enumerate(sections) if _is_experience_section(s["name"])), None)
    summary_idx = next((i for i, s in enumerate(sections) if _is_summary_section(s["name"])), None)
    skills_idx = next((i for i, s in enumerate(sections) if _is_skills_section(s["name"])), None)

    experience_entries = None
    if experience_idx is not None:
        experience_entries = _split_experience_entries(sections[experience_idx]["content"])

    skills_entries = None
    if skills_idx is not None:
        skills_entries = _split_technical_expertise(sections[skills_idx]["content"])

    output_blocks = []
    for i, section in enumerate(sections):
        if i == experience_idx and experience_entries is not None:
            content = _render_experience_entries(experience_entries, payload.experience_bullets)
        elif i == skills_idx and skills_entries is not None:
            content = _render_technical_expertise(skills_entries, payload.technical_expertise)
        elif i == summary_idx and payload.summary.strip() and not _looks_like_leaked_key(payload.summary):
            content = _clean_field_text(payload.summary)
        else:
            # Passthrough for literally everything else -- Header, Education,
            # Certifications, Additional, and a Summary/Skills/Experience
            # section this heuristic couldn't confidently split or that the
            # model left empty -- reproduced byte-for-byte from the master
            # CV, exactly matching its original template 1:1.
            content = section["content"]

        if section["name"].lower() == "header" and i == 0:
            output_blocks.append(content)
        else:
            output_blocks.append(f"{section['name'].upper()}\n{content}")

    return "\n\n".join(output_blocks).strip()


def _build_prompt(
    sections: list[dict], job_title: str, company_name: str, job_description_text: str
) -> tuple[str, list[dict] | None, int | None]:
    experience_idx = next((i for i, s in enumerate(sections) if _is_experience_section(s["name"])), None)
    experience_entries: list[dict] | None = None
    employers_block = "(no Professional Experience section could be confidently parsed -- do not \
return any experience_bullets entries)"
    if experience_idx is not None:
        experience_entries = _split_experience_entries(sections[experience_idx]["content"])
        if experience_entries:
            parts = []
            for i, entry in enumerate(experience_entries):
                bullet_lines = "\n".join(f"  - {b}" for b in entry["bullets"])
                parts.append(f"Employer {i + 1}:\n{bullet_lines}")
            employers_block = "\n\n".join(parts)

    summary_idx = next((i for i, s in enumerate(sections) if _is_summary_section(s["name"])), None)
    summary_block = sections[summary_idx]["content"] if summary_idx is not None else "(none)"

    skills_idx = next((i for i, s in enumerate(sections) if _is_skills_section(s["name"])), None)
    skills_block = "(no Technical Expertise section could be confidently parsed -- do not return \
any technical_expertise entries)"
    if skills_idx is not None:
        skills_entries = _split_technical_expertise(sections[skills_idx]["content"])
        if skills_entries:
            parts = []
            for i, entry in enumerate(skills_entries):
                parts.append(f'Category {i + 1} ("{entry["label"]}"): {entry["items"]}')
            skills_block = "\n".join(parts)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- TARGET JOB ---\n"
        f"Job Title: {job_title}\n"
        f"Company: {company_name}\n"
        f"Job Description:\n{job_description_text}\n\n"
        f"--- EXISTING SUMMARY (reframe this) ---\n{summary_block}\n\n"
        f"--- EXISTING TECHNICAL EXPERTISE CATEGORIES (reorder/rephrase each category's tool list \
only -- the category name itself, shown here only for context, is fixed and you do not return it) \
---\n{skills_block}\n\n"
        f"--- EXISTING BULLETS PER EMPLOYER (reword each employer's bullets; you are NOT told \
which company/title/dates these belong to, and must not guess or reference one) ---\n"
        f"{employers_block}\n"
    )
    return prompt, experience_entries, experience_idx


async def tailor_cv(
    master_cv: MasterCV, job_title: str, company_name: str, job_description_text: str
) -> tuple[list[str], str]:
    """Returns (extracted_keywords, tailored_cv_text). Raises TailoringError
    if Ollama can't be reached or its response can't be safely used -- never
    returns a partially-invalid result for the caller to render into a PDF."""
    settings = get_settings()
    job_title = (job_title or "").strip() or "the target role"
    company_name = (company_name or "").strip() or "the hiring company"
    job_description_text = (job_description_text or "").strip() or job_title

    sections = json.loads(master_cv.sections_json)
    if not sections:
        raise TailoringError("Master CV has no parsed sections to tailor.")

    prompt, experience_entries, experience_idx = _build_prompt(
        sections, job_title, company_name, job_description_text
    )

    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
            "num_ctx": 4096,
            "num_predict": 1024,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(f"{settings.ollama_base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            raw_output = (data.get("response") or "").strip()
    except httpx.HTTPError as exc:
        raise TailoringError(
            f"Could not reach local Ollama instance at {settings.ollama_base_url}: {exc}. "
            f"Is `ollama serve` running with the '{settings.ollama_model}' model pulled?"
        ) from exc

    if not raw_output:
        raise TailoringError("Ollama returned an empty response.")

    try:
        parsed_json = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise TailoringError(f"Ollama did not return valid JSON: {exc}") from exc

    try:
        tailored_payload = _TailoredPayload.model_validate(parsed_json)
    except ValidationError as exc:
        raise TailoringError(f"Ollama's JSON response didn't match the expected shape: {exc}") from exc

    # Structural sanity check: if we sent N employers, a response claiming to
    # cover a wildly different number of employers is more likely a
    # malformed/truncated generation than something safe to positionally
    # zip against real employer entries -- refuse it outright rather than
    # guessing which entries line up.
    if experience_entries is not None and tailored_payload.experience_bullets:
        if len(tailored_payload.experience_bullets) != len(experience_entries):
            raise TailoringError(
                f"Ollama returned {len(tailored_payload.experience_bullets)} employer bullet "
                f"lists but {len(experience_entries)} employers were sent -- refusing to guess "
                f"which entries correspond to which employer."
            )

    tailored_text = _reconstruct_tailored_text(sections, tailored_payload)
    if not tailored_text.strip():
        raise TailoringError("Tailoring produced empty output.")

    keywords = [k.strip() for k in tailored_payload.keywords if k and k.strip()]
    return keywords, tailored_text


def _keyword_present(keyword: str, tailored_text_lower: str) -> bool:
    """Whole-word match for plain-text keywords (so "Go" doesn't match inside
    "Google"), falling back to a plain substring check for keywords containing
    punctuation a word-boundary regex can't handle (e.g. "CI/CD", "Node.js")."""
    keyword_lower = keyword.strip().lower()
    if not keyword_lower:
        return False
    if re.fullmatch(r"[\w\s\-\+#]+", keyword_lower):
        return re.search(rf"\b{re.escape(keyword_lower)}\b", tailored_text_lower) is not None
    return keyword_lower in tailored_text_lower


def compute_match_score(keywords: list[str], tailored_text: str) -> int | None:
    """Returns 0-100: the percentage of the JD's extracted technical keywords
    that show up in the tailored CV -- a simple, deterministic proxy for how
    well the tailoring pass actually worked the target job's terms into the
    resume. Returns None when there are no keywords to score against."""
    if not keywords:
        return None
    tailored_text_lower = tailored_text.lower()
    matched = sum(1 for kw in keywords if _keyword_present(kw, tailored_text_lower))
    return round(100 * matched / len(keywords))
