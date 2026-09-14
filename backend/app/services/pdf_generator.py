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


_environment = Environment(
    loader=FileSystemLoader(Path(__file__).resolve().parents[1] / "templates"),
    autoescape=select_autoescape(["html"]), undefined=StrictUndefined,
)
_environment.filters["inline"] = _inline
_environment.filters["plain"] = _plain
_environment.filters["master_emphasis"] = _master_emphasis
_environment.filters["master_label"] = _master_label
_environment.filters["employer_heading"] = _employer_heading


def render_cv_html(context: dict) -> str:
    return _environment.get_template("cv_template.html").render(**context)


def build_ats_pdf(content: dict | str, layout: dict | None = None) -> bytes:
    """Accept template JSON or legacy cached text. CSS owns the fixed design.

    The layout argument is retained for compatibility and cannot change CSS.
    Async endpoints must invoke this function in a worker thread.
    """
    from app.services.cv_tailor import template_context_from_text

    context = template_context_from_text(content) if isinstance(content, str) else content
    html = render_cv_html(context)
    local_browsers = Path(__file__).resolve().parents[2] / ".playwright"
    if local_browsers.is_dir():
        os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(local_browsers))
    # Uvicorn reload uses a Windows selector policy, which cannot launch
    # subprocesses. Use a private loop without changing the server's policy.
    loop_factory = asyncio.ProactorEventLoop if sys.platform == "win32" else asyncio.new_event_loop
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        pdf = runner.run(_render_pdf(html))
    if len(PdfReader(io.BytesIO(pdf)).pages) != 1:
        raise CVOverflowError(
            "CV exceeds the fixed one-page template. Shorten the summary or bullets and retry."
        )
    return pdf


async def _render_pdf(html: str) -> bytes:
    from playwright.async_api import async_playwright

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        try:
            page = await browser.new_page(viewport={"width": 720, "height": 1056})
            await page.route("**/*", lambda route: route.abort())
            await page.set_content(html, wait_until="load")
            await page.emulate_media(media="print")
            await page.evaluate("document.fonts.ready")
            await page.evaluate("""() => {
                const headline = document.querySelector('.cv-header');
                if (!headline) return;
                const range = document.createRange();
                range.selectNodeContents(headline);
                const available = headline.getBoundingClientRect().width;
                for (let attempt = 0; attempt < 3; attempt++) {
                    const width = range.getBoundingClientRect().width;
                    if (width <= available) break;
                    const size = parseFloat(getComputedStyle(headline).fontSize);
                    headline.style.fontSize = `${Math.floor(size * available / width * 100) / 100}px`;
                }
            }""")
            return await page.pdf(prefer_css_page_size=True, print_background=True,
                                  display_header_footer=False)
        finally:
            await browser.close()
