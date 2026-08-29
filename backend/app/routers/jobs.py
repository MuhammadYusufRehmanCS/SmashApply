import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Job, MasterCV
from app.roles import ALIGNED_ROLES, PRIMARY_ROLE_DEFAULT
from app.schemas import JobOut, ScrapeRequest, ScrapeResult, TailorResult
from app.services.cv_tailor import tailor_cv
from app.services.job_scraper import scrape_for_roles
from app.services.pdf_generator import build_ats_pdf

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


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


def _get_master_cv_or_400(db: Session) -> MasterCV:
    cv = db.execute(select(MasterCV).order_by(MasterCV.id.desc())).scalars().first()
    if not cv:
        raise HTTPException(
            status_code=400, detail="No master CV uploaded yet. Upload one via /api/cv/upload first."
        )
    return cv


async def _run_tailor(job: Job, cv: MasterCV, db: Session) -> tuple[list[str], str]:
    try:
        keywords, tailored_text = await tailor_cv(cv.raw_text, job.description or job.title)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    job.tailored_cv = tailored_text
    job.tailored_keywords = ", ".join(keywords)
    job.tailored_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(job)
    return keywords, tailored_text


@router.post("/{job_id}/tailor", response_model=TailorResult)
async def tailor_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    cv = _get_master_cv_or_400(db)
    keywords, tailored_text = await _run_tailor(job, cv, db)

    return TailorResult(job_id=job.id, keywords=keywords, tailored_cv=tailored_text)


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

    safe = "".join(c for c in f"{job.company}_{job.title}" if c.isalnum() or c in (" ", "-", "_"))
    safe = safe.strip().replace(" ", "_")[:80] or "tailored_cv"
    filename = f"CV_{safe}.pdf".encode("ascii", "ignore").decode("ascii") or "tailored_cv.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
