"""Compiles a tailored CV into an ATS-friendly PDF using reportlab. The
"scale 1.0" baseline (fonts, margins, heading weight/color, rule color) is
read directly from the Master CV's own detected layout profile -- not a
fixed universal spec -- so the output mirrors that specific resume's real
design. A shrink-to-fit loop steps that baseline DOWN (never up) if a
particular tailored CV would otherwise overflow one page.
"""
import io
import re
from xml.sax.saxutils import escape as _xml_escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    HRFlowable,
    PageTemplate,
    Paragraph,
    Spacer,
)
from pypdf import PdfReader

from app.services.text_sections import is_bullet_line, looks_like_entry_header, segment_sections, strip_bullet

# Lower bound for the shrink-to-fit scale loop in build_ats_pdf -- below this,
# text becomes illegibly small, so we stop shrinking and accept the overflow.
_MIN_FIT_SCALE = 0.55
_FIT_SCALE_STEP = 0.05

# Absolute floors/ceilings applied to whatever the master CV's own layout
# reports, purely as a sanity guard against a bad parse -- not a preferred
# spec. The real starting point at scale 1.0 is layout["body_font_size"] etc.
MIN_MARGIN_PT = 18.0
MAX_MARGIN_PT = 90.0
MIN_BODY_FONT_SIZE_PT = 7.0
MAX_BODY_FONT_SIZE_PT = 14.0
BULLET_SPACE_AFTER_PT = 2.0

# Near-black fallback for headings/name when the master CV's own color couldn't
# be detected (or the upload predates color extraction) -- most ATS resumes are
# plain black/dark-gray, so this is a safer default than guessing a brand color.
DEFAULT_TEXT_COLOR_HEX = "#1a1a1a"
DEFAULT_RULE_COLOR_HEX = "#a0a0a0"
_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

# "1/1", "1 / 2", "Page 1 of 2", etc. -- a page-number footer line, not part
# of the candidate's contact info. See _header_flowables.
_PAGE_NUMBER_PATTERN = r"(?:page\s+)?\d+\s*(?:/|of)\s*\d+"
_PAGE_NUMBER_RE = re.compile(rf"^{_PAGE_NUMBER_PATTERN}$", re.IGNORECASE)
_TRAILING_PAGE_NUMBER_RE = re.compile(rf"(?:\s*\|\s*|\s+){_PAGE_NUMBER_PATTERN}$", re.IGNORECASE)
_PAGE_NUMBER_NOISE_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\u200b-\u200f\ufeff]+")


def _layout_color(layout: dict, key: str, fallback: str) -> str:
    """Reads a "#rrggbb" color out of the master CV's detected layout, falling
    back to `fallback` if it's missing or malformed."""
    value = layout.get(key)
    if isinstance(value, str) and _HEX_COLOR_RE.match(value):
        return value
    return fallback


# The tailoring prompt marks a handful of key tool/metric phrases per bullet
# with **double asterisks** (mirroring how a well-formatted resume bolds
# scannable keywords) -- reportlab's Paragraph mini-XML already supports
# inline <b> spans, so this just needs converting after XML-escaping.
_BOLD_MARKDOWN_RE = re.compile(r"\*\*(.+?)\*\*")

# reportlab's base-14 fonts don't reliably render these typographic characters
# (they come out as undefined/garbled glyphs) -- normalize to ASCII equivalents,
# which is also friendlier to ATS text parsers. Bullet glyphs are handled
# separately in _section_flowables, which renders its own "•" marker.
_UNICODE_TO_ASCII = {
    "–": "-",
    "—": "-",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "…": "...",
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

_ITALIC_VARIANT = {
    "Helvetica": "Helvetica-Oblique",
    "Times-Roman": "Times-Italic",
    "Courier": "Courier-Oblique",
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


def _esc_with_markup(text: str) -> str:
    """Like _esc, but also converts **bold** markdown spans (from the
    tailoring prompt) into reportlab's inline <b> markup. The local model
    doesn't always pair ** markers correctly across a long generation --
    any leftover asterisk after well-formed pairs are converted is, by
    definition, an orphaned marker, so it's dropped rather than left visible
    (a plain phrase beats a phrase wrapped in stray asterisks)."""
    converted = _BOLD_MARKDOWN_RE.sub(r"<b>\1</b>", _esc(text))
    return converted.replace("*", "")


def _build_styles(layout: dict, scale: float = 1.0) -> dict:
    base_font = _base_font(layout.get("font_family", "Helvetica"))
    bold_font = _BOLD_VARIANT.get(base_font, "Helvetica-Bold")
    italic_font = _ITALIC_VARIANT.get(base_font, "Helvetica-Oblique")

    # Read the master CV's own detected sizes as the scale-1.0 baseline
    # (clamped only as a sanity guard against a bad parse), then apply the
    # shrink-to-fit scale on top.
    raw_body_size = _clamp(float(layout.get("body_font_size", 10.5)), MIN_BODY_FONT_SIZE_PT, MAX_BODY_FONT_SIZE_PT)
    body_size = _clamp(raw_body_size * scale, 6.5, raw_body_size)

    ratio = _clamp(float(layout.get("line_spacing_ratio", 1.25)), 1.0, 2.2)
    leading = round(body_size * ratio, 2)

    # Headings may legitimately be the SAME size as body text (colored/styled
    # instead of enlarged) -- don't force them larger than what was detected.
    raw_heading_size = _clamp(float(layout.get("heading_font_size", raw_body_size)), raw_body_size, 20.0)
    heading_size = _clamp(raw_heading_size * scale, body_size, raw_heading_size)
    heading_bold = bool(layout.get("heading_bold", True))
    heading_font = bold_font if heading_bold else base_font

    raw_name_size = _clamp(float(layout.get("name_font_size", raw_heading_size * 1.35)), raw_heading_size, 30.0)
    name_size = _clamp(raw_name_size * scale, heading_size, raw_name_size)

    # Reuse the master CV's own detected colors (extracted from the uploaded
    # PDF) rather than imposing an unrelated brand color -- most ATS resumes
    # are plain black/dark-gray, so that's the fallback when nothing could be
    # detected (e.g. an upload from before color extraction existed).
    heading_color = _layout_color(layout, "heading_color", DEFAULT_TEXT_COLOR_HEX)
    name_color = _layout_color(layout, "name_color", heading_color)
    rule_color = _layout_color(layout, "rule_color", DEFAULT_RULE_COLOR_HEX)

    return {
        "name": ParagraphStyle(
            "CVName", fontName=bold_font, fontSize=name_size, leading=name_size * 1.15,
            spaceAfter=max(2.0, 4 * scale), textColor=colors.HexColor(name_color),
        ),
        "contact": ParagraphStyle(
            "CVContact", fontName=base_font, fontSize=max(body_size - 0.5, 6.0), leading=leading,
            spaceAfter=max(4.0, 10 * scale), textColor="#333333",
        ),
        "heading": ParagraphStyle(
            "CVHeading", fontName=heading_font, fontSize=heading_size, leading=heading_size * 1.15,
            spaceBefore=max(6.0, 12 * scale), spaceAfter=max(1.0, 2 * scale), textColor=colors.HexColor(heading_color),
        ),
        "rule_color": colors.HexColor(rule_color),
        "body": ParagraphStyle(
            "CVBody", fontName=base_font, fontSize=body_size, leading=leading, spaceAfter=max(1.0, 3 * scale),
        ),
        "subheading": ParagraphStyle(
            "CVSubheading", fontName=bold_font, fontSize=body_size, leading=leading, spaceBefore=max(4.0, 8 * scale),
            spaceAfter=max(0.5, 1 * scale),
        ),
        "tagline": ParagraphStyle(
            "CVTagline", fontName=italic_font, fontSize=body_size, leading=leading,
            spaceAfter=max(1.0, 3 * scale), textColor="#333333",
        ),
        "bullet": ParagraphStyle(
            "CVBullet", fontName=base_font, fontSize=body_size, leading=leading,
            spaceAfter=max(1.0, BULLET_SPACE_AFTER_PT * scale),
            leftIndent=12 * max(scale, 0.7), firstLineIndent=-12 * max(scale, 0.7),
        ),
    }


def _section_flowables(section: dict, styles: dict, scale: float = 1.0, include_heading: bool = True) -> list:
    flowables: list = []
    if include_heading:
        flowables.append(Paragraph(_esc(section["name"].upper()), styles["heading"]))
        # Thin rule directly beneath the section title -- a neutral divider
        # color independent of the heading text color, matching how resumes
        # typically draw this line (a gray rect, not colored to match text).
        flowables.append(
            HRFlowable(
                width="100%", thickness=max(0.4, 0.75 * scale), color=styles["rule_color"],
                spaceBefore=0, spaceAfter=max(2.0, 4 * scale),
            )
        )

    lines = [ln for ln in section["content"].splitlines() if ln.strip()]

    # Tracks whether the previous line was a job/entry subheading (e.g.
    # "Title | Company | Dates") so the very next non-bullet line -- a short
    # one-line role summary many resumes place before the bullets -- renders
    # in italics instead of as a plain body paragraph.
    prev_was_subheading = False
    for line in lines:
        if is_bullet_line(line):
            flowables.append(Paragraph(f"•  {_esc_with_markup(strip_bullet(line))}", styles["bullet"]))
            prev_was_subheading = False
            continue
        if looks_like_entry_header(line):
            flowables.append(Paragraph(_esc_with_markup(line.strip()), styles["subheading"]))
            prev_was_subheading = True
            continue
        style = styles["tagline"] if prev_was_subheading else styles["body"]
        flowables.append(Paragraph(_esc_with_markup(line.strip()), style))
        prev_was_subheading = False

    flowables.append(Spacer(1, max(2.0, 6 * scale)))
    return flowables


def _fit_font_size(text: str, font_name: str, start_size: float, max_width: float, min_size: float = 7.0) -> float:
    """Steps the font size down until `text` fits `max_width` on one line, so
    the name/tagline header line never wraps."""
    size = start_size
    while size > min_size and stringWidth(text, font_name, size) > max_width:
        size -= 0.5
    return max(size, min_size)


def _strip_header_page_number(line: str) -> str:
    normalized = _PAGE_NUMBER_NOISE_RE.sub("", line).strip()
    if _PAGE_NUMBER_RE.match(normalized):
        return ""
    return _TRAILING_PAGE_NUMBER_RE.sub("", normalized).rstrip(" |")


def _header_flowables(
    header_section: dict | None, styles: dict, scale: float = 1.0, available_width: float | None = None
) -> list:
    if not header_section or not header_section["content"].strip():
        return []
    lines = [_strip_header_page_number(ln) for ln in header_section["content"].splitlines() if ln.strip()]
    # pypdf's text extraction pulls the page-number footer ("1/1", "Page 1 of
    # 2", ...) into the same text stream as everything else on the page, and
    # since it appears before the first detected section heading it lands in
    # this "Header" block alongside the name/contact line. It's not part of
    # the contact info -- drop it rather than pipe-joining it in.
    lines = [ln for ln in lines if ln]
    if not lines:
        return []

    name_line = lines[0].upper()
    name_style = styles["name"]
    if available_width:
        fitted_size = _fit_font_size(name_line, name_style.fontName, name_style.fontSize, available_width)
        if fitted_size < name_style.fontSize:
            name_style.fontSize = fitted_size
            name_style.leading = fitted_size * 1.15

    flowables: list = [Paragraph(_esc(name_line), name_style)]
    if len(lines) > 1:
        flowables.append(Paragraph(_esc("   |   ".join(lines[1:])), styles["contact"]))
    else:
        flowables.append(Spacer(1, max(3.0, 8 * scale)))
    return flowables


def _page_count(pdf_bytes: bytes) -> int:
    try:
        return len(PdfReader(io.BytesIO(pdf_bytes)).pages)
    except Exception:  # noqa: BLE001 - fall back to "fits" if the page count can't be read
        return 1


def build_ats_pdf(tailored_text: str, layout: dict) -> bytes:
    sections = segment_sections(tailored_text)
    if not sections:
        sections = [{"name": "Header", "content": tailored_text}]

    header = sections[0] if sections and sections[0]["name"].lower() == "header" else None
    body_sections = sections[1:] if header else sections

    page_width = _clamp(float(layout.get("page_width", 612)), 500, 700)
    page_height = _clamp(float(layout.get("page_height", 792)), 600, 1000)

    column_count = 2 if layout.get("column_count") == 2 else 1
    raw_margins = layout.get("margins") or {}

    def _margin(key: str, default: float) -> float:
        return _clamp(float(raw_margins.get(key, default)), MIN_MARGIN_PT, MAX_MARGIN_PT)

    def _render(scale: float) -> bytes:
        """Renders the CV at the given font/spacing/margin scale (1.0 = the
        master CV's own detected layout). Called repeatedly by the
        shrink-to-fit loop below so the output always lands on exactly one page."""
        styles = _build_styles(layout, scale)
        m_left = max(MIN_MARGIN_PT, _margin("left", 40.0) * scale)
        m_right = max(MIN_MARGIN_PT, _margin("right", 40.0) * scale)
        m_top = max(MIN_MARGIN_PT, _margin("top", 40.0) * scale)
        m_bottom = max(MIN_MARGIN_PT, _margin("bottom", 40.0) * scale)

        buffer = io.BytesIO()
        header_story = _header_flowables(header, styles, scale, page_width - m_left - m_right)

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

            header_height = 70 * scale
            # Frame() defaults to 6pt of internal padding per side, which would
            # silently shrink the usable width below what _fit_font_size (and
            # the margins themselves) were computed against -- zero it out so
            # our margins are the only inset in effect.
            frame_kwargs = dict(leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
            sidebar_frame = Frame(
                m_left, m_bottom, sidebar_width, page_height - m_top - m_bottom - header_height,
                id="sidebar", showBoundary=0, **frame_kwargs,
            )
            main_frame = Frame(
                m_left + sidebar_width + gutter, m_bottom, main_width,
                page_height - m_top - m_bottom - header_height,
                id="main", showBoundary=0, **frame_kwargs,
            )
            header_frame = Frame(
                m_left, page_height - m_top - header_height, page_width - m_left - m_right, header_height,
                id="header", showBoundary=0, **frame_kwargs,
            )

            doc.addPageTemplates([
                PageTemplate(id="TwoCol", frames=[header_frame, sidebar_frame, main_frame]),
            ])

            sidebar_story: list = []
            for section in sidebar_sections:
                sidebar_story.extend(_section_flowables(section, styles, scale))
            main_story: list = []
            for section in main_sections:
                main_story.extend(_section_flowables(section, styles, scale))

            full_story = (
                (header_story or [Spacer(1, 1)]) + [FrameBreak()] + sidebar_story + [FrameBreak()] + main_story
            )
            doc.build(full_story)
        else:
            doc = BaseDocTemplate(
                buffer, pagesize=(page_width, page_height),
                leftMargin=m_left, rightMargin=m_right, topMargin=m_top, bottomMargin=m_bottom,
            )
            full_frame = Frame(
                m_left, m_bottom, page_width - m_left - m_right, page_height - m_top - m_bottom, id="full",
                leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
            )
            doc.addPageTemplates([PageTemplate(id="OneCol", frames=[full_frame])])

            story = list(header_story)
            for section in body_sections:
                story.extend(_section_flowables(section, styles, scale))
            doc.build(story)

        return buffer.getvalue()

    # Shrink-to-fit: render at full scale first (the master CV's own detected
    # fonts/margins), then step font size/spacing/margins down together until
    # the output fits on exactly one page.
    scale = 1.0
    pdf_bytes = _render(scale)
    while _page_count(pdf_bytes) > 1 and scale > _MIN_FIT_SCALE:
        scale = round(scale - _FIT_SCALE_STEP, 2)
        pdf_bytes = _render(scale)

    return pdf_bytes
