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

from app.services.text_sections import segment_sections

DEFAULT_LAYOUT = {
    "font_family": "Helvetica",
    "body_font_size": 10.5,
    "heading_font_size": 13.0,
    "line_spacing_ratio": 1.25,
    "margins": {"left": 54.0, "right": 54.0, "top": 54.0, "bottom": 54.0},
    "page_width": 612.0,
    "page_height": 792.0,
    "column_count": 1,
}


def _extract_raw_text(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


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
        heading_sizes = sorted((s for s in sizes if s > body_size + 0.5), reverse=True)
        heading_size = heading_sizes[0] if heading_sizes else round(body_size * 1.3, 1)

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
