"""Follows redirects to a job posting's canonical URL and checks whether it's dead."""
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

from app.config import get_settings

DEAD_KEYWORDS = [
    "job expired",
    "no longer accepting applications",
    "position filled",
    "position has been filled",
    "this job is no longer available",
    "posting has expired",
    "job posting has expired",
    "job has expired",
    "this position is no longer",
    "requisition is no longer available",
    "page not found",
    "404",
]


@dataclass
class LinkValidationResult:
    final_url: str
    status_code: int | None
    is_dead: bool
    reason: str | None


async def validate_link(url: str) -> LinkValidationResult:
    settings = get_settings()
    headers = {"User-Agent": settings.link_user_agent}

    try:
        async with httpx.AsyncClient(
            follow_redirects=True, headers=headers, timeout=settings.link_validation_timeout
        ) as client:
            response = await client.get(url)
    except httpx.HTTPError as exc:
        return LinkValidationResult(final_url=url, status_code=None, is_dead=True, reason=str(exc))

    final_url = str(response.url)

    if response.status_code >= 400:
        return LinkValidationResult(
            final_url=final_url,
            status_code=response.status_code,
            is_dead=True,
            reason=f"HTTP {response.status_code}",
        )

    body_text = _extract_visible_text(response.text).lower()
    for keyword in DEAD_KEYWORDS:
        if keyword in body_text:
            return LinkValidationResult(
                final_url=final_url,
                status_code=response.status_code,
                is_dead=True,
                reason=f"matched dead keyword: '{keyword}'",
            )

    return LinkValidationResult(
        final_url=final_url, status_code=response.status_code, is_dead=False, reason=None
    )


def _extract_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" ")
