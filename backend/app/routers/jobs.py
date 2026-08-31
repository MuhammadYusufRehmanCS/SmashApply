import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Job, MasterCV
from app.roles import ALIGNED_ROLES, PRIMARY_ROLE_DEFAULT
from app.schemas import JobOut, ScrapeRequest, ScrapeResult, TailorResult
from app.services.cv_tailor import TailoringError, compute_match_score, tailor_cv
from app.services.job_scraper import scrape_for_roles
from app.services.pdf_generator import build_ats_pdf

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

# Default initials prefix for generated CV filenames (e.g. "MYR_Google.pdf").
DEFAULT_USER_INITIALS = "MYR"


@router.get("", response_model=list[JobOut])
def list_jobs(db: Session = Depends(get_db)):
    stmt = select(Job).order_by(Job.created_at.desc())
    return db.execute(stmt).scalars().all()


@router.post("/scrape", response_model=ScrapeResult)
async def scrape_jobs_endpoint(payload: ScrapeRequest, db: Session = Depends(get_db)):
    primary_role = (payload.primary_role or PRIMARY_ROLE_DEFAULT).strip() or PRIMARY_ROLE_DEFAULT
    location = (payload.location or "Remote").strip() or "Remote"

    found, site_errors = await scrape_for_roles(primary_role, location)

    # Dedupe against both existing DB rows and duplicates within this batch (the same
    # posting can surface under more than one of the 11 queried role titles). Checking
    # the DB per-row without tracking in-batch keys would let a same-batch duplicate slip
    # past the check and hit the UNIQUE constraint on commit, rolling back the whole batch.
    existing_keys = {
        (title, company, job_url)
        for title, company, job_url in db.execute(select(Job.title, Job.company, Job.job_url)).all()
    }

    created = 0
    skipped = 0
    for item in found:
        key = (item["title"], item["company"], item["job_url"])
        if key in existing_keys:
            skipped += 1
            continue
        existing_keys.add(key)

        db.add(
            Job(
                title=item["title"],
                company=item["company"],
                location=item["location"],
                job_url=item["job_url"],
                site=item["site"],
                description=item["description"],
                role_category=item["role_category"],
                is_primary_role=item["is_primary_role"],
                date_posted=item["date_posted"],
            )
        )
        created += 1

    db.commit()

    return ScrapeResult(
        primary_role=primary_role,
        location=location,
        roles_queried=[primary_role] + [r for r in ALIGNED_ROLES if r != primary_role],
        total_found=len(found),
        created=created,
        skipped=skipped,
        site_errors=site_errors,
    )


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


class MasterCvUnavailableError(Exception):
    """Raised when tailoring is attempted but there's no usable Master CV on
    record. This app never writes the uploaded PDF's bytes to disk -- upload
    parses it into raw_text/sections_json/layout_json and stores only that in
    the MasterCV row -- so "the file is missing" here means either no MasterCV
    row exists yet, or its raw_text is empty. Handled globally in main.py."""


def _get_master_cv_or_400(db: Session) -> MasterCV:
    cv = db.execute(select(MasterCV).order_by(MasterCV.id.desc())).scalars().first()
    if not cv or not (cv.raw_text or "").strip():
        raise MasterCvUnavailableError()
    return cv


async def _run_tailor(job: Job, cv: MasterCV, db: Session) -> tuple[list[str], str]:
    try:
        keywords, tailored_text = await tailor_cv(
            cv, job.title, job.company, job.description or job.title
        )
        job.tailored_cv = tailored_text
        job.tailored_keywords = ", ".join(keywords)
        job.tailored_at = datetime.now(timezone.utc)
        job.match_score = compute_match_score(keywords, tailored_text)
        db.commit()
        db.refresh(job)
    except TailoringError as exc:
        # Ollama unreachable, returned invalid/empty JSON, or its response
        # failed shape/structural validation -- tailor_cv() never returns a
        # partially-usable result in these cases, so there is nothing here to
        # fall back to rendering (never the job description, never a raw or
        # corrupted generation). Log the real cause for debugging, but return
        # a clean, generic failure to the client.
        logging.exception("Tailoring failed")
        db.rollback()
        raise HTTPException(status_code=500, detail="CV Tailoring failed. Please retry.") from exc
    except Exception as exc:
        # Anything else (a bug in our own reconstruction code, a bad DB
        # write, ...) would otherwise propagate as an unhandled crash, which
        # browsers surface as a bare "TypeError: Failed to fetch" instead of
        # a real error message.
        logging.exception("Tailoring failed")
        db.rollback()
        raise HTTPException(status_code=500, detail="CV Tailoring failed. Please retry.") from exc

    return keywords, tailored_text


@router.post("/{job_id}/tailor", response_model=TailorResult)
async def tailor_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    cv = _get_master_cv_or_400(db)
    keywords, tailored_text = await _run_tailor(job, cv, db)

    return TailorResult(job_id=job.id, keywords=keywords, tailored_cv=tailored_text)


@router.patch("/{job_id}/toggle-applied", response_model=JobOut)
def toggle_applied(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job.applied = not job.applied
    db.commit()
    db.refresh(job)
    return job


@router.get("/{job_id}/download-cv")
async def download_cv(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    cv = _get_master_cv_or_400(db)
    if not job.tailored_cv:
        await _run_tailor(job, cv, db)

    layout = json.loads(cv.layout_json)
    pdf_bytes = build_ats_pdf(job.tailored_cv, layout)

    # Dynamically fetch and clean this job's company name so every download is
    # named for its target company (e.g. MYR_Google.pdf, MYR_Amazon.pdf) instead
    # of a generic filename.
    clean_company_name = "".join(c for c in (job.company or "") if c.isalnum() or c in (" ", "-", "_"))
    clean_company_name = clean_company_name.strip().replace(" ", "_")[:80] or "Company"
    clean_company_name = clean_company_name.encode("ascii", "ignore").decode("ascii") or "Company"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={DEFAULT_USER_INITIALS}_{clean_company_name}.pdf"},
    )
