"""Heading-detection heuristics shared by the Master CV layout parser and the
ATS PDF generator, so a tailored CV can be re-segmented into the same section
order that was detected on the original upload.
"""

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
