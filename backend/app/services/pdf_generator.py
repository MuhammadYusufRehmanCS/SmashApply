"""Render the fixed Master CV HTML with Chromium; never resize or clip content."""
import asyncio
import io
import os
import re
import sys
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from markupsafe import Markup, escape
from pypdf import PdfReader


class CVOverflowError(ValueError):
    """Content cannot fit the approved single-page design."""

    def __init__(self, message: str, *, measurements: dict | None = None):
        super().__init__(message)
        self.measurements = measurements


def _inline(value: str) -> Markup:
    return Markup(re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", str(escape(value))))


def _plain(value: str) -> str:
    return value.replace("**", "")


def _master_emphasis(value: str, section: str = "experience") -> Markup:
    """Restore the reference PDF's emphasis when stored plain text lost it."""
    if "**" in value:
        return _inline(value)
    phrases = (
        "Cloud & DevOps Engineer", "hands-on experience",
        "production operations across AWS, Azure, and GCP",
        "incident response, and SDLC best practices", "scalable, secure cloud solutions",
        "99.9% uptime", "SonarQube quality gates", "< 60s deployments",
        "artifact management", "infrastructure with Terraform, Ansible",
        "improve reliability", "containerized services", "60%+", "Terraform modules",
        "5–10 minutes", "Docker/Kubernetes", "10–30s service readiness",
    )
    phrases = phrases[:5] if section == "summary" else phrases[5:]
    pattern = "|".join(re.escape(phrase) for phrase in sorted(phrases, key=len, reverse=True))
    return _inline(re.sub(pattern, lambda match: f"**{match[0]}**", value))


def _master_label(value: str) -> Markup:
    text = str(escape(_plain(value)))
    for label in ("A.S. Computer Science &amp; Engineering", "Certifications &amp; Tooling:",
                  "Core Strengths:", "Languages:", "Citizenship:"):
        text = text.replace(label, f"<strong>{label}</strong>")
    return Markup(text)


def _employer_heading(value: str, first: bool) -> Markup:
    title, separator, remainder = _plain(value).partition(" | ")
    if first and separator:
        return Markup("<strong>{}</strong>{}{}").format(title, separator, remainder)
    return escape(_plain(value))


def _labeled_row(value: str) -> Markup:
    """Bolds a 'Label: description' prefix (Core Skills rows) with no bullet
    marker -- the label is everything before the first colon, exactly the
    split `tailor._validate` requires to already be present."""
    label, separator, rest = value.partition(":")
    if not separator:
        return _inline(value)
    return Markup("<strong>{}:</strong> {}").format(escape(_plain(label).strip()), _inline(rest.strip()))


def _project_row(value: str) -> Markup:
    """Match the reference emphasis without changing project wording."""
    return _labeled_row(value)


def _education_row(value: str) -> Markup:
    if _plain(value).startswith("A.S. Computer Science & Engineering"):
        return _master_label(value)
    return _labeled_row(value)


def _company_heading(value: str) -> Markup:
    parts = _plain(value).split(" | ")
    return Markup(" | ").join(
        Markup('<span class="company">{}</span>').format(part)
        if part.strip().casefold() in ("arqon consulting", "ventera group") else escape(part)
        for part in parts
    )


_environment = Environment(
    loader=FileSystemLoader(Path(__file__).resolve().parents[1] / "templates"),
    autoescape=select_autoescape(["html"]), undefined=StrictUndefined,
)
_environment.filters["inline"] = _inline
_environment.filters["plain"] = _plain
_environment.filters["master_emphasis"] = _master_emphasis
_environment.filters["master_label"] = _master_label
_environment.filters["employer_heading"] = _employer_heading
_environment.filters["labeled_row"] = _labeled_row
_environment.filters["project_row"] = _project_row
_environment.filters["education_row"] = _education_row
_environment.filters["company_heading"] = _company_heading


def render_cv_html(context: dict) -> str:
    """All entrypoints use one immutable visual shell; only text is variable."""
    from app.services.cv_tailor import short_role_title
    data = dict(context, role_title=short_role_title(context.get("role_title", "")))
    if "core_skills" not in data:
        data["core_skills"] = [label + ": " + text for label, text in
                               zip(data.get("expertise_labels", []), data.get("technical_expertise", []))]
    if "education" not in data:
        data["education"] = [line["text"] for section in data.get("remaining_sections", [])
                             if "education" in section["name"].lower() for line in section["lines"]]
    if "languages" not in data:
        data["languages"] = next((line["text"] for section in data.get("remaining_sections", [])
                                  for line in section["lines"] if line["text"].startswith("Languages:")), "Languages:")
    entries = []
    for index, entry in enumerate(data.get("experience", [])):
        bullets = (data["experience_bullets"][index] if "experience_bullets" in data else entry["bullets"])
        project = bullets[-1] if bullets and _plain(bullets[-1]).startswith("Selected Project: ") else ""
        entries.append(dict(entry, index=index, standard_bullets=bullets[:-1] if project else bullets, project=project))
    data["experience"] = entries
    if data.get("finalized"):
        from app.services.cv_tailor import TailoringError
        if len(data["core_skills"]) != 3 or [len(e["standard_bullets"]) for e in entries] != [3, 2] or any(not e["project"] for e in entries):
            raise TailoringError("Finalized layout requires three Core Skills rows, 3/2 ordinary experience bullets, and one Selected Project per employer.")
    return _environment.get_template("finalized_cv.html").render(**data)


def build_ats_pdf(content: dict | str, layout: dict | None = None) -> bytes:
    """Accept template JSON or legacy cached text. CSS owns the fixed design.

    The layout argument is retained for compatibility and cannot change CSS.
    Async endpoints must invoke this function in a worker thread.
    """
    from app.services.cv_tailor import template_context_from_text

    context = template_context_from_text(content) if isinstance(content, str) else content
    if len(context.get("summary", "").replace("**", "").split()) > 40:
        raise CVOverflowError("Executive Summary exceeds the 40-word limit.")
    html = render_cv_html(context)
    local_browsers = Path(__file__).resolve().parents[2] / ".playwright"
    if local_browsers.is_dir():
        os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(local_browsers))
    # Uvicorn reload uses a Windows selector policy, which cannot launch
    # subprocesses. Use a private loop without changing the server's policy.
    loop_factory = asyncio.ProactorEventLoop if sys.platform == "win32" else asyncio.new_event_loop
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        pdf, measurements = runner.run(_render_pdf(html))
    if len(PdfReader(io.BytesIO(pdf)).pages) != 1:
        raise CVOverflowError(
            "CV exceeds the fixed one-page template. Shorten the summary or bullets and retry.",
            measurements=measurements,
        )
    # Print width differs from the browser viewport. Check the physical PDF,
    # not just DOM line count, before releasing the finalized resume.
    import pdfplumber
    with pdfplumber.open(io.BytesIO(pdf)) as document:
        page = document.pages[0]
        lines = page.extract_text_lines()
        start = next(i for i, line in enumerate(lines) if line["text"] == "EXECUTIVE SUMMARY")
        end = next(i for i, line in enumerate(lines) if line["text"] == "CORE SKILLS")
        if end - start - 1 > 2:
            raise CVOverflowError("Executive Summary exceeds two physical lines in the printed PDF.")
        margin = 36 - 0.5
        if any(char["x0"] < margin or char["x1"] > float(page.width) - margin
               or char["top"] < margin or char["bottom"] > float(page.height) - margin
               for char in page.chars):
            raise CVOverflowError("CV text exceeds the fixed ATS page margins.")
    return pdf


async def _render_pdf(html: str) -> tuple[bytes, dict | None]:
    from playwright.async_api import async_playwright

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        try:
            page = await browser.new_page(viewport={"width": 720, "height": 1056})
            await page.route("**/*", lambda route: route.abort())
            await page.set_content(html, wait_until="load")
            await page.emulate_media(media="print")
            await page.evaluate("document.fonts.ready")
            finalized_error = await page.evaluate("""() => {
                const header = document.querySelector('#header');
                const role = header?.querySelector('[data-header-role]');
                if (role) {
                    const words = role.textContent.trim().split(/\s+/);
                    const range = document.createRange();
                    range.selectNodeContents(header);
                    const available = header.getBoundingClientRect().width;
                    const tooWide = () => range.getBoundingClientRect().width > available + 0.5;
                    // First shrink the header font to keep the full title on one line.
                    // line-height is fixed in CSS, so the rest of the page does not move.
                    const MIN_HEADER_PT = 9;
                    let size = parseFloat(getComputedStyle(header).fontSize) * 0.75;
                    while (tooWide() && size > MIN_HEADER_PT) {
                        size = Math.max(MIN_HEADER_PT, size - 0.1);
                        header.style.setProperty('font-size', size.toFixed(2) + 'pt', 'important');
                    }
                    // Only if it still does not fit at the minimum size, drop role modifiers.
                    while (tooWide() && words.length > 2) {
                        // Keep the final role noun (Engineer, Architect, etc.).
                        const modifier = words.findIndex(w => /^(senior|junior|lead|principal|staff|sr\.?|jr\.?)$/i.test(w));
                        words.splice(words.length > 3 ? words.length - 2 : (modifier >= 0 ? modifier : words.length - 2), 1);
                        role.textContent = words.join(' ');
                    }
                    // If the meaningful title still cannot fit beside the banner,
                    // wrap the header naturally rather than dropping its role noun.
                    if (range.getBoundingClientRect().width > available + 0.5)
                        header.style.whiteSpace = 'normal';
                }
                if (header && header.getBoundingClientRect().height > 2 * parseFloat(getComputedStyle(header).lineHeight) + 0.5)
                    return 'Header exceeds two physical lines even at the minimum header font size. Shorten the role suffix.';
                const summary = document.querySelector('[data-summary]');
                const lineHeight = summary ? parseFloat(getComputedStyle(summary).lineHeight) : 0;
                if (summary && summary.getBoundingClientRect().height > 2 * lineHeight + 0.5)
                    return 'Executive Summary exceeds two physical lines.';
                if (document.documentElement.scrollWidth > document.documentElement.clientWidth)
                    return 'CV content exceeds the printable width.';
                return null;
            }""")
            if finalized_error:
                raise CVOverflowError(finalized_error, measurements=await _measure_wording(page))
            pdf = await page.pdf(prefer_css_page_size=True, print_background=True,
                                 display_header_footer=False)
            # Measure only after exporting. The diagnostic DOM never changes the
            # delivered PDF, font sizes, spacing, margins, or approved template.
            measurements = None
            if len(PdfReader(io.BytesIO(pdf)).pages) != 1:
                measurements = await _measure_wording(page)
            if len(PdfReader(io.BytesIO(pdf)).pages) == 1:
                footer_overflow = await page.evaluate("() => document.querySelector('.cv-page').getBoundingClientRect().height > 704 * 4 / 3 + 0.5")
                if footer_overflow:
                    raise CVOverflowError("CV exceeds the one-page content budget reserved above the footer.",
                                          measurements=await _measure_wording(page))
            return pdf, measurements
        finally:
            await browser.close()


async def _measure_wording(page) -> dict:
    return await page.evaluate("""() => {
        const rule = [...document.styleSheets].flatMap(s => [...s.cssRules])
            .find(r => r.type === CSSRule.PAGE_RULE);
        const px = value => {
            const probe = document.createElement('div');
            probe.style.width = value;
            document.body.appendChild(probe);
            const width = probe.getBoundingClientRect().width;
            probe.remove();
            return width;
        };
        const dimensions = rule.style.getPropertyValue('size').trim().split(/\\s+/);
        const width = px(dimensions[0]) - px(rule.style.marginLeft) - px(rule.style.marginRight);
        const available = document.body.hasAttribute('data-finalized-cv') ? px('704pt') :
            px(dimensions[1]) - px(rule.style.marginTop) - px(rule.style.marginBottom) - px("16pt");
        document.body.style.width = `${width}px`;
        const fields = [];
        const add = (element, path, prefix = '') => {
            const style = getComputedStyle(element);
            const canvas = document.createElement('canvas').getContext('2d');
            canvas.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
            const text = element.textContent.trim();
            const content = prefix ? text.slice(prefix.length).trim() : text;
            const lineHeight = parseFloat(style.lineHeight);
            const height = element.getBoundingClientRect().height;
            fields.push({path, lines: Math.round(height / lineHeight),
                height, line_height: lineHeight, width: element.getBoundingClientRect().width,
                prefix_width: canvas.measureText(prefix + ' ').width,
                average_char_width: canvas.measureText(content).width / Math.max(1, content.length),
                characters: content.length});
        };
        const summary = document.querySelector('[data-summary]');
        if (summary) add(summary, 'summary');
        for (const element of document.querySelectorAll('[data-field]')) {
            const path = element.dataset.field;
            const prefix = path.startsWith('technical_expertise') ? element.querySelector('strong')?.textContent || '' : '';
            add(element, path, prefix);
        }
        const height = document.querySelector('.cv-page').getBoundingClientRect().height;
        return {available_height: available, content_height: height,
            fixed_height: height - fields.reduce((sum, f) => sum + f.height, 0), fields};
    }""")
