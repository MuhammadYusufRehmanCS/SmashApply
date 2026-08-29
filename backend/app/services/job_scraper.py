"""Live job board scraping via python-jobspy. Queries LinkedIn, Indeed,
Glassdoor, and ZipRecruiter for a primary role plus the 10 aligned
Cloud/DevOps titles in app.roles.ALIGNED_ROLES.
"""
import asyncio
import logging

import pandas as pd
from jobspy import scrape_jobs

from app.config import get_settings
from app.roles import ALIGNED_ROLES

logger = logging.getLogger(__name__)


def _scrape_one(search_term: str, location: str) -> pd.DataFrame:
    settings = get_settings()
    return scrape_jobs(
        site_name=settings.jobspy_site_list,
        search_term=search_term,
        location=location,
        results_wanted=settings.jobspy_results_wanted,
        hours_old=settings.jobspy_hours_old,
        country_indeed=settings.jobspy_country_indeed,
        linkedin_fetch_description=True,
    )


def _row_value(row: pd.Series, key: str, default: str = "") -> str:
    value = row.get(key, default)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return str(value).strip()


async def scrape_for_roles(primary_role: str, location: str) -> tuple[list[dict], list[str]]:
    """Returns (job_dicts, per-role error messages)."""
    roles = [(primary_role, True)] + [(role, False) for role in ALIGNED_ROLES if role != primary_role]

    results: list[dict] = []
    errors: list[str] = []

    for role, is_primary in roles:
        try:
            df = await asyncio.to_thread(_scrape_one, role, location)
        except Exception as exc:  # noqa: BLE001 - one bad role/site shouldn't abort the whole scrape
            logger.warning("jobspy scrape failed for role=%s: %s", role, exc)
            errors.append(f"{role}: {exc}")
            continue

        if df is None or df.empty:
            continue

        for _, row in df.iterrows():
            title = _row_value(row, "title")
            job_url = _row_value(row, "job_url")
            if not title or not job_url:
                continue
            results.append(
                {
                    "title": title,
                    "company": _row_value(row, "company"),
                    "location": _row_value(row, "location", location),
                    "job_url": job_url,
                    "site": _row_value(row, "site"),
                    "description": _row_value(row, "description"),
                    "date_posted": _row_value(row, "date_posted") or None,
                    "role_category": role,
                    "is_primary_role": is_primary,
                }
            )

    return results, errors
