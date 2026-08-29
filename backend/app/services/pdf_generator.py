"""Compiles a tailored CV into an ATS-friendly PDF using reportlab, mirroring
the visual profile extracted from the uploaded Master CV: section order,
font proportions, line height, heading style, bullet styling, margins, and
single/two-column structure.
"""
import io
from xml.sax.saxutils import escape as _xml_escape

from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    PageTemplate,
    Paragraph,
    Spacer,
)

from app.services.text_sections import is_bullet_line, segment_sections, strip_bullet

# reportlab's base-14 fonts don't reliably render these typographic characters
# (they come out as undefined/garbled glyphs) -- normalize to ASCII equivalents,
# which is also friendlier to ATS text parsers.
_UNICODE_TO_ASCII = {
    "–": "-",
    "—": "-",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "…": "...",
    "•": "-",
    " ": " ",
}

_FONT_ALIASES = {
    "helvetica": "Helvetica",
    "arial": "Helvetica",
    "calibri": "Helvetica",
    "verdana": "Helvetica",
    "segoe": "Helvetica",
    "tahoma": "Helvetica",
    "times": "Times-Roman",
    "georgia": "Times-Roman",
    "cambria": "Times-Roman",
    "garamond": "Times-Roman",
    "minion": "Times-Roman",
    "courier": "Courier",
    "consolas": "Courier",
    "menlo": "Courier",
}

_BOLD_VARIANT = {
    "Helvetica": "Helvetica-Bold",
    "Times-Roman": "Times-Bold",
    "Courier": "Courier-Bold",
}

SIDEBAR_SECTION_NAMES = {
    "skills",
    "technical skills",
    "core competencies",
    "certifications",
    "certificates",
    "licenses & certifications",
    "tools",
    "tools & technologies",
    "education",
}


def _base_font(font_family: str) -> str:
    key = (font_family or "").lower()
    for needle, mapped in _FONT_ALIASES.items():
        if needle in key:
            return mapped
    return "Helvetica"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _esc(text: str) -> str:
    """Normalizes typographic punctuation to ASCII, then escapes for
    reportlab's Paragraph mini-XML markup (&, <, >)."""
    for src, dst in _UNICODE_TO_ASCII.items():
        text = text.replace(src, dst)
    return _xml_escape(text)


def _build_styles(layout: dict) -> dict:
    base_font = _base_font(layout.get("font_family", "Helvetica"))
    bold_font = _BOLD_VARIANT.get(base_font, "Helvetica-Bold")

    body_size = _clamp(float(layout.get("body_font_size", 10.5)), 8.0, 13.0)
    heading_size = _clamp(float(layout.get("heading_font_size", body_size * 1.25)), body_size + 1, 20.0)
    spacing_ratio = _clamp(float(layout.get("line_spacing_ratio", 1.25)), 1.0, 1.8)
    leading = round(body_size * spacing_ratio, 1)

    name_size = _clamp(heading_size * 1.35, heading_size, 26.0)

    return {
        "name": ParagraphStyle(
            "CVName", fontName=bold_font, fontSize=name_size, leading=name_size * 1.15,
            spaceAfter=4,
        ),
        "contact": ParagraphStyle(
            "CVContact", fontName=base_font, fontSize=max(body_size - 0.5, 8.0), leading=leading,
            spaceAfter=10, textColor="#333333",
        ),
        "heading": ParagraphStyle(
            "CVHeading", fontName=bold_font, fontSize=heading_size, leading=heading_size * 1.15,
            spaceBefore=10, spaceAfter=4, textColor="#1a1a1a",
        ),
        "body": ParagraphStyle(
            "CVBody", fontName=base_font, fontSize=body_size, leading=leading, spaceAfter=3,
        ),
        "subheading": ParagraphStyle(
            "CVSubheading", fontName=bold_font, fontSize=body_size, leading=leading, spaceBefore=4,
            spaceAfter=1,
        ),
        "bullet": ParagraphStyle(
            "CVBullet", fontName=base_font, fontSize=body_size, leading=leading, spaceAfter=2,
            leftIndent=12, firstLineIndent=-12,
        ),
    }


def _looks_like_subheading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return False
    return bool(any(ch.isdigit() for ch in stripped) or " - " in stripped or " | " in stripped or "|" in stripped)


def _section_flowables(section: dict, styles: dict, include_heading: bool = True) -> list:
    flowables: list = []
    if include_heading:
        flowables.append(Paragraph(_esc(section["name"].upper()), styles["heading"]))

    lines = [ln for ln in section["content"].splitlines() if ln.strip()]

    for line in lines:
        if is_bullet_line(line):
            # Plain-text "- " prefix (not a reportlab ListFlowable bullet glyph) so ATS
            # text extractors read a real hyphen instead of an undefined glyph code.
            flowables.append(Paragraph(f"-  {_esc(strip_bullet(line))}", styles["bullet"]))
            continue
        style = styles["subheading"] if _looks_like_subheading(line) else styles["body"]
        flowables.append(Paragraph(_esc(line.strip()), style))

    flowables.append(Spacer(1, 6))
    return flowables


def _header_flowables(header_section: dict | None, styles: dict) -> list:
    if not header_section or not header_section["content"].strip():
        return []
    lines = [ln.strip() for ln in header_section["content"].splitlines() if ln.strip()]
    if not lines:
        return []
    flowables: list = [Paragraph(_esc(lines[0]), styles["name"])]
    if len(lines) > 1:
        flowables.append(Paragraph(_esc("   |   ".join(lines[1:])), styles["contact"]))
    else:
        flowables.append(Spacer(1, 8))
    return flowables


def build_ats_pdf(tailored_text: str, layout: dict) -> bytes:
    sections = segment_sections(tailored_text)
    if not sections:
        sections = [{"name": "Header", "content": tailored_text}]

    header = sections[0] if sections and sections[0]["name"].lower() == "header" else None
    body_sections = sections[1:] if header else sections

    styles = _build_styles(layout)

    page_width = _clamp(float(layout.get("page_width", 612)), 500, 700)
    page_height = _clamp(float(layout.get("page_height", 792)), 600, 1000)
    margins = layout.get("margins") or {}
    m_left = _clamp(float(margins.get("left", 54)), 28, 100)
    m_right = _clamp(float(margins.get("right", 54)), 28, 100)
    m_top = _clamp(float(margins.get("top", 54)), 28, 100)
    m_bottom = _clamp(float(margins.get("bottom", 54)), 28, 100)

    buffer = io.BytesIO()
    column_count = 2 if layout.get("column_count") == 2 else 1
    header_story = _header_flowables(header, styles)

    if column_count == 2:
        sidebar_sections = [s for s in body_sections if s["name"].lower() in SIDEBAR_SECTION_NAMES]
        main_sections = [s for s in body_sections if s not in sidebar_sections]

        doc = BaseDocTemplate(
            buffer, pagesize=(page_width, page_height),
            leftMargin=m_left, rightMargin=m_right, topMargin=m_top, bottomMargin=m_bottom,
        )
        sidebar_width = (page_width - m_left - m_right) * 0.32
        main_width = (page_width - m_left - m_right) * 0.62
        gutter = (page_width - m_left - m_right) * 0.06

        header_height = 70
        sidebar_frame = Frame(
            m_left, m_bottom, sidebar_width, page_height - m_top - m_bottom - header_height,
            id="sidebar", showBoundary=0,
        )
        main_frame = Frame(
            m_left + sidebar_width + gutter, m_bottom, main_width,
            page_height - m_top - m_bottom - header_height,
            id="main", showBoundary=0,
        )
        header_frame = Frame(
            m_left, page_height - m_top - header_height, page_width - m_left - m_right, header_height,
            id="header", showBoundary=0,
        )

        doc.addPageTemplates([
            PageTemplate(id="TwoCol", frames=[header_frame, sidebar_frame, main_frame]),
        ])

        sidebar_story: list = []
        for section in sidebar_sections:
            sidebar_story.extend(_section_flowables(section, styles))
        main_story: list = []
        for section in main_sections:
            main_story.extend(_section_flowables(section, styles))

        full_story = (header_story or [Spacer(1, 1)]) + [FrameBreak()] + sidebar_story + [FrameBreak()] + main_story
        doc.build(full_story)
    else:
        doc = BaseDocTemplate(
            buffer, pagesize=(page_width, page_height),
            leftMargin=m_left, rightMargin=m_right, topMargin=m_top, bottomMargin=m_bottom,
        )
        full_frame = Frame(m_left, m_bottom, page_width - m_left - m_right, page_height - m_top - m_bottom, id="full")
        doc.addPageTemplates([PageTemplate(id="OneCol", frames=[full_frame])])

        story = list(header_story)
        for section in body_sections:
            story.extend(_section_flowables(section, styles))
        doc.build(story)

    return buffer.getvalue()
