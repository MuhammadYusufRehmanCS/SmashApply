"""Live job scraping across JobSpy boards, direct ATS APIs, and BuiltIn.

The service normalizes all providers into the Job model shape used by the
router. Results are US-filtered and deduplicated before persistence so no
single board, including Indeed, dominates the feed.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import math
import re
from collections import defaultdict, deque
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

try:
    import httpx
except ImportError:  # pragma: no cover - exercised only in lean local envs
    httpx = None  # type: ignore[assignment]

try:  # Optional at import time so tests can run without scrape dependencies.
    import pandas as pd
except ImportError:  # pragma: no cover - exercised only in lean local envs
    pd = None  # type: ignore[assignment]

try:
    from jobspy import scrape_jobs as _jobspy_scrape_jobs
except ImportError:  # pragma: no cover - exercised only in lean local envs
    _jobspy_scrape_jobs = None

from app.config import get_settings
from app.roles import ALIGNED_ROLES

logger = logging.getLogger(__name__)

US_SEARCH_LOCATION = "United States"
BUILTIN_BASE_URL = "https://builtin.com"
GREENHOUSE_JOBS_URL = "https://boards-api.greenhouse.io/v1/boards/{board}/jobs"
LEVER_POSTINGS_URL = "https://api.lever.co/v0/postings/{site}"
REMOTIVE_JOBS_URL = "https://remotive.com/api/remote-jobs"
THEMUSE_JOBS_URL = "https://www.themuse.com/api/public/jobs"

EXTRA_SEARCH_ROLES = (
    "DevOps Engineer",
    "Cloud DevOps Engineer",
    "AWS DevOps Engineer",
    "Azure DevOps Engineer",
)
JOBSPY_SITES = {"linkedin", "indeed", "glassdoor", "zip_recruiter", "bayt"}

HTTP_HEADERS = {
    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

TRACKING_QUERY_PARAMS = {
    "gh_src",
    "ref",
    "refid",
    "source",
    "trk",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}

ROLE_TOKEN_ALIASES = {
    "devsecops": {"devsecops", "devops"},
    "systems": {"systems", "system"},
    "site": {"site", "sre"},
    "reliability": {"reliability", "sre"},
}

STATE_NAMES = {
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "new hampshire",
    "new jersey",
    "new mexico",
    "new york",
    "north carolina",
    "north dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhode island",
    "south carolina",
    "south dakota",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west virginia",
    "wisconsin",
    "wyoming",
    "district of columbia",
}
STATE_ABBREVIATIONS = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
}
STATE_ABBREVIATION_RE = re.compile(
    r"(?:^|[\s,(/-])("
    + "|".join(sorted(STATE_ABBREVIATIONS))
    + r")(?:$|[\s,)./-])"
)
STATE_NAME_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(state) for state in sorted(STATE_NAMES, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)
US_COUNTRY_RE = re.compile(r"\b(?:united states(?: of america)?|u\.s\.a?\.?|usa)\b", re.IGNORECASE)
US_ABBREVIATION_RE = re.compile(r"(?:^|[\s,(/-])(?:US|U\.S\.?)(?:$|[\s,)./-])")


class _PlainTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"br", "div", "li", "p", "section"}:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data)


class _JsonLdParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[str] = []
        self._capturing = False
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "script":
            return

        attributes = {key.lower(): value or "" for key, value in attrs}
        if "ld+json" in attributes.get("type", "").lower():
            self._capturing = True
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._capturing:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script" and self._capturing:
            self.scripts.append("".join(self._parts))
            self._capturing = False
            self._parts = []


class _MetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, str] = {}
        self.title = ""
        self._in_title = False
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self._in_title = True
            return

        if tag.lower() != "meta":
            return

        attributes = {key.lower(): value or "" for key, value in attrs}
        key = attributes.get("property") or attributes.get("name")
        content = attributes.get("content")
        if key and content:
            self.meta[key.lower()] = content

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False
            self.title = _clean_text(" ".join(self._title_parts))


class _JobLinkParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.links: list[str] = []
        self._seen: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return

        attributes = {key.lower(): value or "" for key, value in attrs}
        href = attributes.get("href", "")
        if "/job/" not in href:
            return

        absolute_url = urljoin(self.base_url, href)
        parsed = urlparse(absolute_url)
        if not parsed.netloc.endswith("builtin.com"):
            return

        normalized = normalize_job_url(absolute_url)
        if normalized in self._seen:
            return

        self._seen.add(normalized)
        self.links.append(absolute_url)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _html_to_text(value: Any) -> str:
    raw = _clean_text(value)
    if not raw:
        return ""
    if "<" not in raw or ">" not in raw:
        return raw

    parser = _PlainTextParser()
    try:
        parser.feed(raw)
    except Exception:  # noqa: BLE001 - malformed provider HTML should not abort ingestion
        return _clean_text(re.sub(r"<[^>]+>", " ", raw))
    return _clean_text(" ".join(parser.parts))


def _missing_value(value: Any) -> bool:
    if value is None:
        return True

    if pd is not None:
        try:
            missing = pd.isna(value)
        except Exception:  # noqa: BLE001 - pandas may not like arbitrary values
            missing = False
        if isinstance(missing, bool):
            return missing

    return isinstance(value, float) and math.isnan(value)


def _row_value(row: Any, key: str, default: str = "") -> str:
    value = row.get(key, default)
    if _missing_value(value):
        return default
    return _clean_text(value)


def _format_error(exc: BaseException) -> str:
    return str(exc).strip() or exc.__class__.__name__


async def _gather_with_concurrency(
    factories: Iterable[Callable[[], Any]],
    limit: int,
    *,
    return_exceptions: bool = False,
) -> list[Any]:
    semaphore = asyncio.Semaphore(max(1, limit))

    async def run(factory: Callable[[], Any]) -> Any:
        async with semaphore:
            return await factory()

    return await asyncio.gather(
        *(run(factory) for factory in factories),
        return_exceptions=return_exceptions,
    )


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", _clean_text(value).lower())).strip()


def _role_tokens(value: str) -> list[str]:
    return [token for token in _normalize_text(value).split() if token not in {"and", "or", "the"}]


def _role_matches_title(title: str, role: str) -> bool:
    normalized_title = _normalize_text(title)
    normalized_role = _normalize_text(role)
    if not normalized_title or not normalized_role:
        return False
    if normalized_role in normalized_title:
        return True

    title_tokens = set(normalized_title.split())
    if "sre" in title_tokens:
        title_tokens.update({"site", "reliability", "engineer"})

    for token in _role_tokens(role):
        accepted_tokens = ROLE_TOKEN_ALIASES.get(token, {token})
        if not title_tokens.intersection(accepted_tokens):
            return False
    return True


def _match_role(title: str, roles: Iterable[tuple[str, bool]]) -> tuple[str, bool] | None:
    for role, is_primary in roles:
        if _role_matches_title(title, role):
            return role, is_primary
    return None


def _roles_for(primary_role: str) -> list[tuple[str, bool]]:
    primary = (primary_role or "").strip()
    roles: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for role, is_primary in (
        [(primary, True)]
        + [(role, False) for role in ALIGNED_ROLES]
        + [(role, False) for role in EXTRA_SEARCH_ROLES]
    ):
        normalized = _normalize_text(role)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        roles.append((role, is_primary))
    return roles


def scrape_role_names(primary_role: str) -> list[str]:
    return [role for role, _ in _roles_for(primary_role)]


def _has_us_signal(*values: Any) -> bool:
    original_text = " ".join(_clean_text(value) for value in values if _clean_text(value))
    if not original_text:
        return False

    if US_COUNTRY_RE.search(original_text) or US_ABBREVIATION_RE.search(original_text):
        return True
    if STATE_ABBREVIATION_RE.search(original_text):
        return True
    return bool(STATE_NAME_RE.search(original_text))


def _is_generic_remote_location(value: Any) -> bool:
    normalized = _normalize_text(value)
    if not normalized:
        return True
    generic_tokens = {"remote", "telecommute", "virtual", "anywhere", "work", "from", "home"}
    return all(token in generic_tokens for token in normalized.split())


def _has_us_location_signal(location: Any, *fallback_values: Any) -> bool:
    if _has_us_signal(location):
        return True
    if _is_generic_remote_location(location):
        return _has_us_signal(*fallback_values)
    return False


def _coerce_us_search_location(location: str) -> str:
    cleaned = _clean_text(location)
    if not cleaned or "remote" in cleaned.lower():
        return US_SEARCH_LOCATION
    if _has_us_signal(cleaned):
        return cleaned
    return US_SEARCH_LOCATION


def _jobspy_location(site: str, requested_location: str) -> str:
    if site.strip().lower() == "linkedin":
        return US_SEARCH_LOCATION
    return _coerce_us_search_location(requested_location)


def normalize_job_url(url: Any) -> str:
    cleaned = _clean_text(url)
    if not cleaned:
        return ""

    parsed = urlparse(cleaned)
    if not parsed.scheme or not parsed.netloc:
        return _normalize_text(cleaned)

    query_params = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=False):
        normalized_key = key.lower()
        if normalized_key.startswith("utm_") or normalized_key in TRACKING_QUERY_PARAMS:
            continue
        query_params.append((normalized_key, value))

    path = (parsed.path or "").rstrip("/").lower()
    query = urlencode(sorted(query_params))
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", query, ""))


def _dedupe_text(value: Any) -> str:
    normalized = _normalize_text(value)
    return re.sub(r"\b(?:inc|incorporated|llc|ltd|corp|corporation|co|company)\b", "", normalized).strip()


def job_dedupe_keys(job: Mapping[str, Any]) -> set[tuple[str, ...]]:
    """Return duplicate keys for a normalized URL and company/title pair."""
    keys: set[tuple[str, ...]] = set()

    normalized_url = normalize_job_url(job.get("job_url", ""))
    if normalized_url:
        keys.add(("url", normalized_url))

    title = _dedupe_text(job.get("title", ""))
    company = _dedupe_text(job.get("company", ""))
    if title and company:
        keys.add(("company_title", company, title))

    return keys


def _dedupe_jobs(jobs: Iterable[dict]) -> list[dict]:
    seen: set[tuple[str, ...]] = set()
    unique_jobs: list[dict] = []

    for job in jobs:
        keys = job_dedupe_keys(job)
        if keys and seen.intersection(keys):
            continue
        seen.update(keys)
        unique_jobs.append(job)

    return unique_jobs


def _interleave_by_field(jobs: Iterable[dict], field: str) -> list[dict]:
    groups: dict[str, deque[dict]] = defaultdict(deque)
    for job in jobs:
        groups[_clean_text(job.get(field)) or "unknown"].append(job)

    interleaved: list[dict] = []
    while groups:
        for key in list(groups.keys()):
            group = groups[key]
            interleaved.append(group.popleft())
            if not group:
                del groups[key]
    return interleaved


def _order_for_persistence(jobs: Iterable[dict]) -> list[dict]:
    interleaved = _interleave_by_field(jobs, "site")
    jobspy_jobs = [job for job in interleaved if _clean_text(job.get("site")).lower() in JOBSPY_SITES]
    direct_jobs = [job for job in interleaved if _clean_text(job.get("site")).lower() not in JOBSPY_SITES]

    # The list endpoint sorts newest first. Insert JobSpy rows first so direct
    # ATS/public-board rows are visible at the top after a scrape.
    return _interleave_by_field(jobspy_jobs, "site") + _interleave_by_field(direct_jobs, "site")


def _job_dict(
    *,
    title: str,
    company: str,
    location: str,
    job_url: str,
    site: str,
    description: str,
    date_posted: str | None,
    role_category: str,
    is_primary_role: bool,
) -> dict | None:
    title = _clean_text(title)
    job_url = _clean_text(job_url)
    if not title or not job_url:
        return None

    return {
        "title": title,
        "company": _clean_text(company),
        "location": _clean_text(location) or US_SEARCH_LOCATION,
        "job_url": job_url,
        "site": _clean_text(site),
        "description": _clean_text(description),
        "date_posted": _clean_text(date_posted) or None,
        "role_category": role_category,
        "is_primary_role": is_primary_role,
    }


def _scrape_jobspy_one(site: str, search_term: str, location: str) -> Any:
    if _jobspy_scrape_jobs is None:
        raise RuntimeError("python-jobspy is not installed")

    settings = get_settings()
    return _jobspy_scrape_jobs(
        site_name=[site],
        search_term=search_term,
        location=location,
        results_wanted=settings.jobspy_results_wanted,
        hours_old=settings.jobspy_hours_old,
        country_indeed=settings.jobspy_country_indeed,
        linkedin_fetch_description=True,
    )


async def _scrape_jobspy_site_role(
    site: str,
    role: str,
    is_primary: bool,
    requested_location: str,
) -> tuple[list[dict], list[str]]:
    source_location = _jobspy_location(site, requested_location)
    try:
        df = await asyncio.to_thread(_scrape_jobspy_one, site, role, source_location)
    except Exception as exc:  # noqa: BLE001 - one bad source should not abort the scrape
        logger.warning("jobspy scrape failed for site=%s role=%s: %s", site, role, exc)
        return [], [f"jobspy/{site}/{role}: {_format_error(exc)}"]

    if df is None or getattr(df, "empty", True):
        return [], []

    jobs: list[dict] = []
    for _, row in df.iterrows():
        title = _row_value(row, "title")
        job_url = _row_value(row, "job_url")
        raw_location = _row_value(row, "location")
        description = _row_value(row, "description")
        if not title or not job_url:
            continue
        if not _has_us_location_signal(raw_location, description):
            continue

        job = _job_dict(
            title=title,
            company=_row_value(row, "company"),
            location=raw_location or source_location,
            job_url=job_url,
            site=_row_value(row, "site", site) or site,
            description=description,
            date_posted=_row_value(row, "date_posted") or None,
            role_category=role,
            is_primary_role=is_primary,
        )
        if job:
            jobs.append(job)

    return jobs, []


async def _scrape_jobspy_sources(roles: list[tuple[str, bool]], requested_location: str) -> tuple[list[dict], list[str]]:
    if _jobspy_scrape_jobs is None:
        return [], ["jobspy: python-jobspy is not installed"]

    settings = get_settings()
    factories = [
        lambda site=site.strip(), role=role, is_primary=is_primary: _scrape_jobspy_site_role(
            site,
            role,
            is_primary,
            requested_location,
        )
        for site in settings.jobspy_site_list
        if site.strip()
        for role, is_primary in roles
    ]
    provider_results = await _gather_with_concurrency(
        factories,
        getattr(settings, "job_source_concurrency", 4),
    )

    jobs: list[dict] = []
    errors: list[str] = []
    for provider_jobs, provider_errors in provider_results:
        jobs.extend(provider_jobs)
        errors.extend(provider_errors)

    jobs = _interleave_by_field(_dedupe_jobs(jobs), "site")
    return jobs[: getattr(settings, "jobspy_total_results_wanted", 40)], errors


async def _fetch_json(client: httpx.AsyncClient, url: str, params: dict[str, str] | None = None) -> Any:
    response = await client.get(url, params=params)
    response.raise_for_status()
    return response.json()


async def _fetch_text(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(url)
    response.raise_for_status()
    return response.text


def _join_unique(values: Iterable[str]) -> str:
    seen: set[str] = set()
    parts: list[str] = []
    for value in values:
        cleaned = _clean_text(value)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            parts.append(cleaned)
    return ", ".join(parts)


def _parse_date(value: Any) -> str | None:
    cleaned = _clean_text(value)
    if not cleaned:
        return None
    if re.fullmatch(r"\d{13}", cleaned):
        return datetime.fromtimestamp(int(cleaned) / 1000, tz=timezone.utc).date().isoformat()
    if re.fullmatch(r"\d{10}", cleaned):
        return datetime.fromtimestamp(int(cleaned), tz=timezone.utc).date().isoformat()
    return cleaned[:10] if re.match(r"\d{4}-\d{2}-\d{2}", cleaned) else cleaned


def _greenhouse_location(posting: Mapping[str, Any]) -> str:
    locations: list[str] = []

    location = posting.get("location")
    if isinstance(location, Mapping):
        locations.append(_clean_text(location.get("name")))
    else:
        locations.append(_clean_text(location))

    for office in posting.get("offices") or []:
        if not isinstance(office, Mapping):
            continue
        locations.append(_clean_text(office.get("name")))
        locations.append(_clean_text(office.get("location")))

    return _join_unique(locations)


async def _scrape_greenhouse_board(
    client: httpx.AsyncClient,
    board_token: str,
    company_name: str,
    roles: list[tuple[str, bool]],
) -> tuple[list[dict], list[str]]:
    try:
        data = await _fetch_json(
            client,
            GREENHOUSE_JOBS_URL.format(board=board_token),
            params={"content": "true"},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("greenhouse scrape failed for board=%s: %s", board_token, exc)
        return [], [f"greenhouse/{board_token}: {_format_error(exc)}"]

    jobs: list[dict] = []
    for posting in data.get("jobs") or []:
        if not isinstance(posting, Mapping):
            continue

        title = _clean_text(posting.get("title"))
        role_match = _match_role(title, roles)
        if not role_match:
            continue

        location = _greenhouse_location(posting)
        description = _html_to_text(posting.get("content"))
        if not _has_us_location_signal(location, description):
            continue

        role, is_primary = role_match
        job = _job_dict(
            title=title,
            company=company_name,
            location=location,
            job_url=_clean_text(posting.get("absolute_url") or posting.get("url")),
            site="greenhouse",
            description=description,
            date_posted=_parse_date(posting.get("updated_at")),
            role_category=role,
            is_primary_role=is_primary,
        )
        if job:
            jobs.append(job)

    return jobs, []


async def _scrape_greenhouse_sources(roles: list[tuple[str, bool]]) -> tuple[list[dict], list[str]]:
    if httpx is None:
        return [], ["greenhouse: httpx is not installed"]

    settings = get_settings()
    if not settings.greenhouse_boards:
        return [], []

    timeout = httpx.Timeout(settings.job_http_timeout_seconds)
    async with httpx.AsyncClient(headers=HTTP_HEADERS, timeout=timeout, follow_redirects=True) as client:
        factories = [
            lambda board_token=board_token, company_name=company_name: _scrape_greenhouse_board(
                client,
                board_token,
                company_name,
                roles,
            )
            for board_token, company_name in settings.greenhouse_boards
        ]
        provider_results = await _gather_with_concurrency(
            factories,
            getattr(settings, "job_source_concurrency", 4),
        )

    jobs: list[dict] = []
    errors: list[str] = []
    for provider_jobs, provider_errors in provider_results:
        jobs.extend(provider_jobs)
        errors.extend(provider_errors)

    jobs = _interleave_by_field(_dedupe_jobs(jobs), "company")
    return jobs[: settings.ats_results_wanted], errors


def _lever_location(posting: Mapping[str, Any]) -> str:
    locations: list[str] = []
    categories = posting.get("categories")
    if isinstance(categories, Mapping):
        locations.append(_clean_text(categories.get("location")))

    workplace_type = posting.get("workplaceType")
    if workplace_type:
        locations.append(_clean_text(workplace_type))

    return _join_unique(locations)


def _lever_description(posting: Mapping[str, Any]) -> str:
    parts = [
        _clean_text(posting.get("descriptionPlain")),
        _html_to_text(posting.get("description")),
        _clean_text(posting.get("additionalPlain")),
        _html_to_text(posting.get("additional")),
    ]

    for item in posting.get("lists") or []:
        if not isinstance(item, Mapping):
            continue
        parts.append(_clean_text(item.get("content")))
    return _join_unique(parts)


async def _scrape_lever_board(
    client: httpx.AsyncClient,
    site: str,
    company_name: str,
    roles: list[tuple[str, bool]],
) -> tuple[list[dict], list[str]]:
    try:
        data = await _fetch_json(client, LEVER_POSTINGS_URL.format(site=site), params={"mode": "json"})
    except Exception as exc:  # noqa: BLE001
        logger.warning("lever scrape failed for site=%s: %s", site, exc)
        return [], [f"lever/{site}: {_format_error(exc)}"]

    jobs: list[dict] = []
    for posting in data or []:
        if not isinstance(posting, Mapping):
            continue

        title = _clean_text(posting.get("text"))
        role_match = _match_role(title, roles)
        if not role_match:
            continue

        location = _lever_location(posting)
        description = _lever_description(posting)
        if not _has_us_location_signal(location, description):
            continue

        role, is_primary = role_match
        job = _job_dict(
            title=title,
            company=company_name,
            location=location,
            job_url=_clean_text(posting.get("hostedUrl") or posting.get("applyUrl")),
            site="lever",
            description=description,
            date_posted=_parse_date(posting.get("createdAt")),
            role_category=role,
            is_primary_role=is_primary,
        )
        if job:
            jobs.append(job)

    return jobs, []


async def _scrape_lever_sources(roles: list[tuple[str, bool]]) -> tuple[list[dict], list[str]]:
    if httpx is None:
        return [], ["lever: httpx is not installed"]

    settings = get_settings()
    if not settings.lever_boards:
        return [], []

    timeout = httpx.Timeout(settings.job_http_timeout_seconds)
    async with httpx.AsyncClient(headers=HTTP_HEADERS, timeout=timeout, follow_redirects=True) as client:
        factories = [
            lambda site=site, company_name=company_name: _scrape_lever_board(
                client,
                site,
                company_name,
                roles,
            )
            for site, company_name in settings.lever_boards
        ]
        provider_results = await _gather_with_concurrency(
            factories,
            getattr(settings, "job_source_concurrency", 4),
        )

    jobs: list[dict] = []
    errors: list[str] = []
    for provider_jobs, provider_errors in provider_results:
        jobs.extend(provider_jobs)
        errors.extend(provider_errors)

    jobs = _interleave_by_field(_dedupe_jobs(jobs), "company")
    return jobs[: settings.ats_results_wanted], errors


def _slugify_role(role: str) -> str:
    return "-".join(_role_tokens(role))


def _builtin_search_urls(role: str) -> list[str]:
    role_slug = _slugify_role(role)
    if not role_slug:
        return []
    return [
        f"{BUILTIN_BASE_URL}/jobs/dev-engineering/search/{role_slug}",
        f"{BUILTIN_BASE_URL}/jobs/remote/dev-engineering/search/{role_slug}",
    ]


def _json_ld_jobpostings(html_text: str) -> list[Mapping[str, Any]]:
    parser = _JsonLdParser()
    parser.feed(html_text)

    postings: list[Mapping[str, Any]] = []
    for script in parser.scripts:
        try:
            parsed = json.loads(script)
        except json.JSONDecodeError:
            continue
        postings.extend(_walk_jobpostings(parsed))
    return postings


def _walk_jobpostings(value: Any) -> list[Mapping[str, Any]]:
    postings: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        raw_type = value.get("@type") or value.get("type")
        types = raw_type if isinstance(raw_type, list) else [raw_type]
        if any(_clean_text(item).lower() == "jobposting" for item in types):
            postings.append(value)
        for nested_value in value.values():
            postings.extend(_walk_jobpostings(nested_value))
    elif isinstance(value, list):
        for item in value:
            postings.extend(_walk_jobpostings(item))
    return postings


def _schema_text(value: Any) -> str:
    if isinstance(value, Mapping):
        parts = []
        for key in (
            "name",
            "address",
            "streetAddress",
            "addressLocality",
            "addressRegion",
            "addressCountry",
            "postalCode",
        ):
            parts.append(_schema_text(value.get(key)))
        return _join_unique(parts)
    if isinstance(value, list):
        return _join_unique(_schema_text(item) for item in value)
    return _clean_text(value)


def _schema_company(posting: Mapping[str, Any]) -> str:
    organization = posting.get("hiringOrganization")
    if isinstance(organization, Mapping):
        return _clean_text(organization.get("name"))
    return _clean_text(organization)


def _schema_location(posting: Mapping[str, Any]) -> str:
    return _join_unique(
        [
            _schema_text(posting.get("jobLocation")),
            _schema_text(posting.get("applicantLocationRequirements")),
            _schema_text(posting.get("jobLocationType")),
        ]
    )


def _schema_url(posting: Mapping[str, Any], fallback_url: str) -> str:
    return _clean_text(posting.get("url") or posting.get("sameAs") or fallback_url)


def _job_from_schema(
    posting: Mapping[str, Any],
    roles: list[tuple[str, bool]],
    fallback_url: str,
    site: str,
) -> dict | None:
    title = _clean_text(posting.get("title"))
    role_match = _match_role(title, roles)
    if not role_match:
        return None

    location = _schema_location(posting)
    description = _html_to_text(posting.get("description"))
    if not _has_us_location_signal(location, description):
        return None

    role, is_primary = role_match
    return _job_dict(
        title=title,
        company=_schema_company(posting),
        location=location,
        job_url=_schema_url(posting, fallback_url),
        site=site,
        description=description,
        date_posted=_parse_date(posting.get("datePosted")),
        role_category=role,
        is_primary_role=is_primary,
    )


def _builtin_links(html_text: str, page_url: str) -> list[str]:
    parser = _JobLinkParser(page_url)
    parser.feed(html_text)
    return parser.links


def _metadata_job(html_text: str, page_url: str, roles: list[tuple[str, bool]], site: str) -> dict | None:
    parser = _MetadataParser()
    parser.feed(html_text)
    raw_title = _clean_text(parser.meta.get("og:title") or parser.title)
    if not raw_title:
        return None

    raw_title = re.sub(r"\s+\|\s+Built In.*$", "", raw_title, flags=re.IGNORECASE)
    company = ""
    title = raw_title
    at_match = re.match(r"(.+?)\s+at\s+(.+)$", raw_title, flags=re.IGNORECASE)
    if at_match:
        title = at_match.group(1)
        company = at_match.group(2)
    elif " - " in raw_title:
        title, company = [part.strip() for part in raw_title.split(" - ", 1)]

    role_match = _match_role(title, roles)
    if not role_match:
        return None

    description = _clean_text(parser.meta.get("og:description") or parser.meta.get("description"))
    page_text = _html_to_text(html_text)
    if not _has_us_location_signal(US_SEARCH_LOCATION, description, page_text[:5000]):
        return None

    role, is_primary = role_match
    return _job_dict(
        title=title,
        company=company,
        location=US_SEARCH_LOCATION,
        job_url=page_url,
        site=site,
        description=description or page_text[:4000],
        date_posted=None,
        role_category=role,
        is_primary_role=is_primary,
    )


async def _scrape_builtin_detail(
    client: httpx.AsyncClient,
    job_url: str,
    roles: list[tuple[str, bool]],
) -> tuple[list[dict], list[str]]:
    try:
        html_text = await _fetch_text(client, job_url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("builtin detail scrape failed for url=%s: %s", job_url, exc)
        return [], [f"builtin/detail: {_format_error(exc)}"]

    jobs = [
        job
        for posting in _json_ld_jobpostings(html_text)
        if (job := _job_from_schema(posting, roles, job_url, "builtin"))
    ]
    if jobs:
        return jobs, []

    fallback_job = _metadata_job(html_text, job_url, roles, "builtin")
    return ([fallback_job] if fallback_job else []), []


async def _scrape_builtin_sources(roles: list[tuple[str, bool]]) -> tuple[list[dict], list[str]]:
    if httpx is None:
        return [], ["builtin: httpx is not installed"]

    settings = get_settings()
    timeout = httpx.Timeout(settings.job_http_timeout_seconds)

    jobs: list[dict] = []
    errors: list[str] = []
    detail_links: list[str] = []
    seen_links: set[str] = set()

    async with httpx.AsyncClient(headers=HTTP_HEADERS, timeout=timeout, follow_redirects=True) as client:
        page_urls = [
            page_url
            for role, _ in roles
            for page_url in _builtin_search_urls(role)
        ]
        page_results = await _gather_with_concurrency(
            [lambda page_url=page_url: _fetch_text(client, page_url) for page_url in page_urls],
            getattr(settings, "job_source_concurrency", 4),
            return_exceptions=True,
        )

        for page_url, result in zip(page_urls, page_results, strict=False):
            if isinstance(result, Exception):
                logger.warning("builtin page scrape failed for url=%s: %s", page_url, result)
                errors.append(f"builtin/{page_url}: {_format_error(result)}")
                continue

            for posting in _json_ld_jobpostings(result):
                job = _job_from_schema(posting, roles, page_url, "builtin")
                if job:
                    jobs.append(job)

            for link in _builtin_links(result, page_url):
                normalized = normalize_job_url(link)
                if normalized in seen_links:
                    continue
                seen_links.add(normalized)
                detail_links.append(link)
                if len(detail_links) >= settings.ats_results_wanted:
                    break

            if len(detail_links) >= settings.ats_results_wanted:
                break

        if len(jobs) < settings.ats_results_wanted and detail_links:
            detail_results = await _gather_with_concurrency(
                [lambda link=link: _scrape_builtin_detail(client, link, roles) for link in detail_links],
                getattr(settings, "job_source_concurrency", 4),
            )
            for detail_jobs, detail_errors in detail_results:
                jobs.extend(detail_jobs)
                errors.extend(detail_errors)

    jobs = _interleave_by_field(_dedupe_jobs(jobs), "company")
    return jobs[: settings.ats_results_wanted], errors


async def _scrape_remotive_role(
    client: httpx.AsyncClient,
    role: str,
    is_primary: bool,
) -> tuple[list[dict], list[str]]:
    try:
        data = await _fetch_json(client, REMOTIVE_JOBS_URL, params={"search": role})
    except Exception as exc:  # noqa: BLE001
        logger.warning("remotive scrape failed for role=%s: %s", role, exc)
        return [], [f"remotive/{role}: {_format_error(exc)}"]

    jobs: list[dict] = []
    for posting in data.get("jobs") or []:
        if not isinstance(posting, Mapping):
            continue

        title = _clean_text(posting.get("title"))
        role_match = _match_role(title, [(role, is_primary)]) or _match_role(title, _roles_for(role))
        if not role_match:
            continue

        location = _clean_text(posting.get("candidate_required_location"))
        description = _html_to_text(posting.get("description"))
        if not _has_us_location_signal(location, description):
            continue

        matched_role, matched_is_primary = role_match
        job = _job_dict(
            title=title,
            company=_clean_text(posting.get("company_name")),
            location=location or US_SEARCH_LOCATION,
            job_url=_clean_text(posting.get("url")),
            site="remotive",
            description=description,
            date_posted=_parse_date(posting.get("publication_date")),
            role_category=matched_role,
            is_primary_role=is_primary and matched_is_primary,
        )
        if job:
            jobs.append(job)

    return jobs, []


async def _scrape_remotive_sources(roles: list[tuple[str, bool]]) -> tuple[list[dict], list[str]]:
    if httpx is None:
        return [], ["remotive: httpx is not installed"]

    settings = get_settings()
    timeout = httpx.Timeout(settings.job_http_timeout_seconds)
    async with httpx.AsyncClient(headers=HTTP_HEADERS, timeout=timeout, follow_redirects=True) as client:
        provider_results = await _gather_with_concurrency(
            [
                lambda role=role, is_primary=is_primary: _scrape_remotive_role(
                    client,
                    role,
                    is_primary,
                )
                for role, is_primary in roles
            ],
            getattr(settings, "job_source_concurrency", 4),
        )

    jobs: list[dict] = []
    errors: list[str] = []
    for provider_jobs, provider_errors in provider_results:
        jobs.extend(provider_jobs)
        errors.extend(provider_errors)

    jobs = _interleave_by_field(_dedupe_jobs(jobs), "company")
    return jobs[: settings.ats_results_wanted], errors


def _themuse_location(posting: Mapping[str, Any]) -> str:
    locations = posting.get("locations") or []
    if not isinstance(locations, list):
        return _clean_text(locations)
    return _join_unique(_clean_text(location.get("name")) for location in locations if isinstance(location, Mapping))


async def _scrape_themuse_page(
    client: httpx.AsyncClient,
    category: str,
    page: int,
    roles: list[tuple[str, bool]],
) -> tuple[list[dict], list[str]]:
    try:
        data = await _fetch_json(
            client,
            THEMUSE_JOBS_URL,
            params={"category": category, "location": US_SEARCH_LOCATION, "page": str(page)},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("themuse scrape failed for category=%s page=%s: %s", category, page, exc)
        return [], [f"themuse/{category}/page-{page}: {_format_error(exc)}"]

    jobs: list[dict] = []
    for posting in data.get("results") or []:
        if not isinstance(posting, Mapping):
            continue

        title = _clean_text(posting.get("name"))
        role_match = _match_role(title, roles)
        if not role_match:
            continue

        location = _themuse_location(posting)
        description = _html_to_text(posting.get("contents"))
        if not _has_us_location_signal(location, description):
            continue

        company = posting.get("company")
        refs = posting.get("refs")
        role, is_primary = role_match
        job = _job_dict(
            title=title,
            company=_clean_text(company.get("name") if isinstance(company, Mapping) else company),
            location=location,
            job_url=_clean_text(refs.get("landing_page") if isinstance(refs, Mapping) else ""),
            site="themuse",
            description=description,
            date_posted=_parse_date(posting.get("publication_date")),
            role_category=role,
            is_primary_role=is_primary,
        )
        if job:
            jobs.append(job)

    return jobs, []


async def _scrape_themuse_sources(roles: list[tuple[str, bool]]) -> tuple[list[dict], list[str]]:
    if httpx is None:
        return [], ["themuse: httpx is not installed"]

    settings = get_settings()
    timeout = httpx.Timeout(settings.job_http_timeout_seconds)
    async with httpx.AsyncClient(headers=HTTP_HEADERS, timeout=timeout, follow_redirects=True) as client:
        provider_results = await _gather_with_concurrency(
            [
                lambda category=category, page=page: _scrape_themuse_page(client, category, page, roles)
                for category in settings.themuse_category_list
                for page in range(1, settings.themuse_pages + 1)
            ],
            getattr(settings, "job_source_concurrency", 4),
        )

    jobs: list[dict] = []
    errors: list[str] = []
    for provider_jobs, provider_errors in provider_results:
        jobs.extend(provider_jobs)
        errors.extend(provider_errors)

    jobs = _interleave_by_field(_dedupe_jobs(jobs), "company")
    return jobs[: settings.ats_results_wanted], errors


async def scrape_for_roles(primary_role: str, location: str) -> tuple[list[dict], list[str]]:
    """Return normalized job dictionaries plus per-source error messages."""
    roles = _roles_for(primary_role)
    requested_location = _coerce_us_search_location(location)
    settings = get_settings()

    direct_provider_tasks = []
    jobspy_task = None
    for source in settings.job_source_list:
        if source == "greenhouse":
            direct_provider_tasks.append(_scrape_greenhouse_sources(roles))
        elif source == "lever":
            direct_provider_tasks.append(_scrape_lever_sources(roles))
        elif source == "builtin":
            direct_provider_tasks.append(_scrape_builtin_sources(roles))
        elif source == "remotive":
            direct_provider_tasks.append(_scrape_remotive_sources(roles))
        elif source == "themuse":
            direct_provider_tasks.append(_scrape_themuse_sources(roles))
        elif source == "jobspy":
            jobspy_task = _scrape_jobspy_sources(roles, requested_location)
        else:
            logger.warning("unknown job ingestion source configured: %s", source)

    if not direct_provider_tasks and jobspy_task is None:
        return [], ["no active job ingestion sources configured"]

    provider_results = []
    if direct_provider_tasks:
        provider_results.extend(await asyncio.gather(*direct_provider_tasks))
    if jobspy_task is not None:
        provider_results.append(await jobspy_task)

    results: list[dict] = []
    errors: list[str] = []
    for provider_jobs, provider_errors in provider_results:
        results.extend(provider_jobs)
        errors.extend(provider_errors)

    return _order_for_persistence(_dedupe_jobs(results)), errors
