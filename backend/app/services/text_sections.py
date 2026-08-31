"""Heading-detection heuristics shared by the Master CV layout parser and the
ATS PDF generator, so a tailored CV can be re-segmented into the same section
order that was detected on the original upload.
"""
import re

SECTION_KEYWORDS = {
    "summary",
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

BULLET_PREFIXES = ("-", "*", "•", "◦", "▪", "●", "‣", "·")


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
_DATE_RANGE_RE = re.compile(r"(19|20)\d{2}\s*(?:[-–—]|to)\s*((19|20)\d{2}|present|current)", re.IGNORECASE)


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
    if not stripped or len(stripped) > 100:
        return False
    if stripped.startswith(BULLET_PREFIXES):
        return False
    # Pipe-delimited "Title | Company | Location | Dates" is the dominant
    # convention this targets -- a strong, specific signal on its own.
    if " | " in stripped or stripped.count("|") >= 2:
        return True
    return bool(_DATE_RANGE_RE.search(stripped))


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
        if looks_like_heading(line):
            if current_lines:
                sections.append({"name": current_name, "content": "\n".join(current_lines).strip()})
            current_name = line.strip().title()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({"name": current_name, "content": "\n".join(current_lines).strip()})

    return [s for s in sections if s["content"]]
