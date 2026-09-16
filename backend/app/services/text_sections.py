"""Heading-detection heuristics shared by the Master CV layout parser and the
ATS PDF generator, so a tailored CV can be re-segmented into the same section
order that was detected on the original upload.
"""
import re

SECTION_KEYWORDS = {
    "summary",
    "executive summary",
    "technical expertise",
    "additional",
    "education & professional development",
    "professional summary",
    "objective",
    "profile",
    "experience",
    "work experience",
    "professional experience",
    "employment history",
    "skills",
    "technical skills",
    "core competencies",
    "certifications",
    "certificates",
    "licenses & certifications",
    "education",
    "projects",
    "tools",
    "tools & technologies",
    "achievements",
    "awards",
    "summary of qualifications",
}

BULLET_PREFIXES = ("-", "*", "•", "◦", "▪", "●", "‣", "·", "\ufffd")


def looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 45:
        return False
    lowered = stripped.lower().strip(":").strip()
    if lowered in SECTION_KEYWORDS:
        return True
    letters = [c for c in stripped if c.isalpha()]
    if len(letters) >= 3 and stripped == stripped.upper() and not stripped.startswith(BULLET_PREFIXES):
        return True
    return False


# A real date RANGE (a 4-digit year followed by a dash/"to" and either
# another year or "present"/"current"), not just any digit -- a bullet like
# "reduced deployment time by 60%" or "managed 40+ AWS accounts" contains
# digits too, and must NOT be mistaken for a job-entry boundary.
_DATE_RANGE_RE = re.compile(
    r"(?:\b(?:19|20)\d{2}|\b\d{1,2}/(?:19|20)\d{2})\s*"
    r"(?:[-\u2010-\u2015\u2212\ufffd]|to)\s*"
    r"(?:(?:[A-Za-z]{3,9}\.?\s+)?(?:19|20)\d{2}|\d{1,2}/(?:19|20)\d{2}|present|current|now|ongoing)",
    re.IGNORECASE,
)
_ROLE_TITLE_RE = re.compile(
    r"^(?:(?:senior|junior|lead|staff|principal|associate)\s+)?"
    r"[A-Za-z &/().-]{0,85}\b(?:engineer|administrator|developer|analyst|architect|consultant|manager)"
    r"(?:\s+(?:I{1,3}|IV|V|[1-5]))?$", re.I,
)


def looks_like_entry_header(line: str) -> bool:
    """True for a "Title | Company | Location | Dates"-shaped line -- the
    boundary between one job entry and the next in a Professional Experience
    section. Used both by the PDF generator (to render it as a subheading)
    and by the tailoring pipeline (to split the section into per-employer
    entries whose company/dates/title get passed through untouched rather
    than ever being sent to the LLM) -- so this has to be precise, not just
    "looks header-ish": a false positive in the tailoring pipeline means a
    bullet gets silently treated as an immutable header instead of being
    sent for tailoring, or a bullet's own text gets used as if it were a
    company/dates line."""
    stripped = line.strip()
    if not stripped or len(stripped) > 400:
        return False
    if stripped.startswith(BULLET_PREFIXES):
        return False
    # Pipe-delimited "Title | Company | Location | Dates" is the dominant
    # convention this targets -- a strong, specific signal on its own.
    if " | " in stripped or stripped.count("|") >= 2 or (
        "|" in stripped and _ROLE_TITLE_RE.fullmatch(stripped.split("|", 1)[0].strip())
    ):
        return True
    if _DATE_RANGE_RE.search(stripped) or _ROLE_TITLE_RE.fullmatch(stripped):
        return True
    return bool(re.search(r"\b(?:engineer|administrator|developer|analyst|architect|consultant|manager)\b.*?(?:\s+at\s+|\s+[-\u2013\u2014]\s+)", stripped, re.I))


def is_bullet_line(line: str) -> bool:
    return line.strip().startswith(BULLET_PREFIXES)


def strip_bullet(line: str) -> str:
    stripped = line.strip()
    for prefix in BULLET_PREFIXES:
        if stripped.startswith(prefix):
            return stripped[len(prefix):].strip()
    return stripped


def segment_sections(raw_text: str) -> list[dict]:
    """Split raw CV text into an ordered list of {"name", "content"} blocks.

    The first block (before any detected heading) is named "Header" and
    normally holds the candidate's name/contact line.
    """
    lines = raw_text.splitlines()
    sections: list[dict] = []
    current_name = "Header"
    current_lines: list[str] = []

    for line in lines:
        is_heading = looks_like_heading(line)
        in_experience = any(word in current_name.lower() for word in ("experience", "employment", "work history"))
        previous = next((value for value in reversed(current_lines) if value.strip()), "")
        # Uppercase titles and the company line below them are employer
        # metadata, not new top-level resume sections.
        if (in_experience and line.strip().lower().rstrip(":") not in SECTION_KEYWORDS
                and (looks_like_entry_header(line) or looks_like_entry_header(previous))):
            is_heading = False
        if is_heading:
            if current_lines:
                sections.append({"name": current_name, "content": "\n".join(current_lines).strip()})
            current_name = line.strip().title()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({"name": current_name, "content": "\n".join(current_lines).strip()})

    return [s for s in sections if s["content"]]
