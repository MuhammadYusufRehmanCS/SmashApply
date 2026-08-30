"""Parses an uploaded Master CV PDF into raw text plus a layout/formatting
metadata profile: font family/sizes, heading vs. body sizing, line spacing,
margins, single vs. two-column structure, and section order.

`pypdf` does the plain text extraction; `pdfplumber` exposes per-character
positioning/font data used to derive the layout profile. Both are heuristic
by nature -- there's no reliable way to recover a PDF's original design
intent, only to approximate it closely enough for the reportlab generator to
mirror proportions and ordering.
"""
import io
from collections import Counter

import pdfplumber
from pypdf import PdfReader

from app.services.text_sections import looks_like_heading, segment_sections

DEFAULT_LAYOUT = {
    "font_family": "Helvetica",
    "body_font_size": 10.5,
    "heading_font_size": 13.0,
    "line_spacing_ratio": 1.25,
    "margins": {"left": 54.0, "right": 54.0, "top": 54.0, "bottom": 54.0},
    "page_width": 612.0,
    "page_height": 792.0,
    "column_count": 1,
    "heading_color": "#1a1a1a",
    "heading_bold": True,
    "name_color": "#1a1a1a",
    "rule_color": "#a0a0a0",
}


def _extract_raw_text(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _pdf_color_to_hex(color) -> str | None:
    """Converts a pdfplumber `non_stroking_color` value -- a grayscale float,
    an RGB tuple, or a CMYK tuple, each component in 0..1 -- to "#rrggbb".
    Returns None for shapes we can't confidently interpret rather than
    guessing, so callers can fall back to a safe default."""
    if color is None:
        return None
    if isinstance(color, (int, float)):
        components = (color, color, color)
    elif isinstance(color, (list, tuple)):
        if len(color) == 1:
            g = color[0]
            components = (g, g, g)
        elif len(color) == 3:
            components = tuple(color)
        elif len(color) == 4:
            c, m, y, k = color
            components = (1 - min(1.0, c + k), 1 - min(1.0, m + k), 1 - min(1.0, y + k))
        else:
            return None
    else:
        return None

    def _to_byte(v: float) -> int:
        return max(0, min(255, round(float(v) * 255)))

    r, g, b = (_to_byte(v) for v in components)
    return f"#{r:02x}{g:02x}{b:02x}"


def _dominant_color(chars: list[dict]) -> str | None:
    counts = Counter(_pdf_color_to_hex(c.get("non_stroking_color")) for c in chars)
    counts.pop(None, None)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _group_lines(chars: list[dict]) -> dict[float, list[dict]]:
    lines: dict[float, list[dict]] = {}
    for c in chars:
        lines.setdefault(round(c["top"], 0), []).append(c)
    return lines


def _detect_rule_color(page) -> str | None:
    """Finds thin, wide filled rects (the horizontal divider lines many resumes
    draw under section headings) and returns their dominant color. These are
    vector graphics, not text, so they're invisible to the char-based color
    detection above and need pdfplumber's `rects` instead."""
    candidates = [
        r for r in page.rects
        if r.get("fill") and (r["bottom"] - r["top"]) <= 2.0 and (r["x1"] - r["x0"]) > page.width * 0.3
    ]
    if not candidates:
        return None
    return _dominant_color(candidates)


def _detect_column_count(chars: list[dict], margin_left: float, page_width: float) -> int:
    lines: dict[float, list[dict]] = {}
    for c in chars:
        key = round(c["top"], 0)
        lines.setdefault(key, []).append(c)

    line_starts = [min(c["x0"] for c in cs) for cs in lines.values()]
    if not line_starts:
        return 1

    mid = page_width / 2
    near_left = sum(1 for x in line_starts if x <= margin_left + 15)
    near_mid = sum(1 for x in line_starts if mid - 20 <= x <= mid + 40)
    total = len(line_starts)

    if total and near_left / total > 0.15 and near_mid / total > 0.15:
        return 2
    return 1


def _analyze_layout(content: bytes) -> dict:
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        if not pdf.pages:
            return dict(DEFAULT_LAYOUT)

        page = pdf.pages[0]
        chars = page.chars
        if not chars:
            return dict(DEFAULT_LAYOUT)

        font_names = Counter(c["fontname"] for c in chars)
        font_family = font_names.most_common(1)[0][0]

        sizes = Counter(round(c["size"], 1) for c in chars)
        body_size = sizes.most_common(1)[0][0]

        # Cross-reference the char-geometry data with the same text-based
        # heading heuristic the PDF generator itself uses, so "heading style"
        # is measured from actual section-title lines (EXECUTIVE SUMMARY,
        # TECHNICAL EXPERTISE, ...) rather than just "the biggest font on the
        # page" -- which is often the candidate's name/tagline line instead,
        # and can be a completely different size/weight/color than headings.
        line_groups = _group_lines(chars)
        heading_chars: list[dict] = []
        for line_chars in line_groups.values():
            line_text = "".join(c["text"] for c in line_chars)
            if looks_like_heading(line_text):
                heading_chars.extend(line_chars)

        if heading_chars:
            heading_size_counts = Counter(round(c["size"], 1) for c in heading_chars)
            heading_size = heading_size_counts.most_common(1)[0][0]
            heading_bold = sum("bold" in c["fontname"].lower() for c in heading_chars) > len(heading_chars) / 2
        else:
            heading_sizes = sorted((s for s in sizes if s > body_size + 0.5), reverse=True)
            heading_size = heading_sizes[0] if heading_sizes else round(body_size * 1.3, 1)
            heading_bold = True

        x0s = [c["x0"] for c in chars]
        x1s = [c["x1"] for c in chars]
        tops = [c["top"] for c in chars]
        bottoms = [c["bottom"] for c in chars]

        margin_left = min(x0s)
        margin_right = max(0.0, page.width - max(x1s))
        margin_top = min(tops)
        margin_bottom = max(0.0, page.height - max(bottoms))

        line_tops = sorted(set(round(c["top"], 1) for c in chars))
        gaps = [b - a for a, b in zip(line_tops, line_tops[1:]) if b - a > 1]
        avg_line_gap = (sum(gaps) / len(gaps)) if gaps else body_size * 1.2
        line_spacing_ratio = round(avg_line_gap / body_size, 2) if body_size else 1.25

        # Pull the actual color used for section headings and for the first
        # line on the page (the candidate's name), so the generated PDF can
        # reuse the master CV's real color scheme -- most ATS resumes are
        # plain black, so falling back to near-black rather than guessing a
        # brand color keeps the output honest when no color is detected. If no
        # heading-shaped line was found (fallback branch above), approximate
        # heading chars by font size instead.
        if not heading_chars:
            heading_chars = [c for c in chars if round(c["size"], 1) == heading_size]
        heading_color = _dominant_color(heading_chars) or DEFAULT_LAYOUT["heading_color"]
        name_chars = [c for c in chars if abs(c["top"] - margin_top) < 2.0]
        name_color = _dominant_color(name_chars) or heading_color
        rule_color = _detect_rule_color(page) or DEFAULT_LAYOUT["rule_color"]

        name_font_size = None
        if name_chars:
            name_size_counts = Counter(round(c["size"], 1) for c in name_chars)
            name_font_size = name_size_counts.most_common(1)[0][0]

        return {
            "font_family": font_family,
            "body_font_size": body_size,
            "heading_font_size": heading_size,
            "line_spacing_ratio": max(1.0, min(line_spacing_ratio, 2.2)),
            "margins": {
                "left": round(margin_left, 1),
                "right": round(margin_right, 1),
                "top": round(margin_top, 1),
                "bottom": round(margin_bottom, 1),
            },
            "page_width": round(page.width, 1),
            "page_height": round(page.height, 1),
            "column_count": _detect_column_count(chars, margin_left, page.width),
            "heading_color": heading_color,
            "heading_bold": heading_bold,
            "name_color": name_color,
            "name_font_size": name_font_size if name_font_size is not None else round(heading_size * 1.35, 1),
            "rule_color": rule_color,
        }


def parse_master_cv(content: bytes) -> dict:
    """Returns {"raw_text": str, "sections": [...], "layout": {...}}."""
    raw_text = _extract_raw_text(content).strip()
    if not raw_text:
        raise ValueError("No extractable text found in this PDF.")

    sections = segment_sections(raw_text)

    try:
        layout = _analyze_layout(content)
    except Exception:  # noqa: BLE001 - layout parsing is best-effort, text extraction already succeeded
        layout = dict(DEFAULT_LAYOUT)

    layout["section_order"] = [s["name"] for s in sections]

    return {"raw_text": raw_text, "sections": sections, "layout": layout}
