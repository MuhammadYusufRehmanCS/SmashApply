from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import CVMaster, Job, JobStatus
from app.schemas import IngestResult, JobOut, RevalidateResult
from app.services.email_ingestion import fetch_new_jobs
from app.services.link_validator import validate_link
from app.services.vector_matcher import match_score

router = APIRouter(prefix="/api/ingest", tags=["ingest"])


@router.post("/gmail", response_model=IngestResult)
async def ingest_gmail(db: Session = Depends(get_db)):
    try:
        candidates = fetch_new_jobs()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface a clear JSON error instead of an unhandled 500
        raise HTTPException(status_code=502, detail=f"Gmail sync failed: {exc}") from exc

    cv = db.execute(select(CVMaster).order_by(CVMaster.id.desc())).scalars().first()

    created, skipped, errors = 0, 0, []

    for candidate in candidates:
        exists = db.execute(select(Job).where(Job.raw_url == candidate.raw_url)).scalars().first()
        if exists:
            skipped += 1
            continue

        try:
            validation = await validate_link(candidate.raw_url)
        except Exception as exc:  # noqa: BLE001 - surface per-link failures without aborting the batch
            errors.append(f"{candidate.raw_url}: {exc}")
            continue

        description = ""  # full description scraping is left as a follow-up; title/company suffice to start
        score = match_score(candidate.title, cv.embedding) if cv else 0.0

        job = Job(
            title=candidate.title,
            company=candidate.company,
            raw_url=candidate.raw_url,
            final_url=validation.final_url,
            description=description,
            match_score=score,
            status=JobStatus.EXPIRED.value if validation.is_dead else JobStatus.UNSEEN.value,
            validation_note=validation.reason,
            gmail_message_id=candidate.message_id,
        )
        db.add(job)
        created += 1

    db.commit()
    return IngestResult(fetched=len(candidates), created=created, skipped=skipped, errors=errors)


@router.post("/validate/{job_id}", response_model=JobOut)
async def validate_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    validation = await validate_link(job.raw_url)
    job.final_url = validation.final_url
    job.validation_note = validation.reason
    if validation.is_dead:
        job.status = JobStatus.EXPIRED.value
    db.commit()
    db.refresh(job)
    return job


@router.post("/revalidate", response_model=RevalidateResult)
async def revalidate_active(db: Session = Depends(get_db)):
    active_jobs = (
        db.execute(
            select(Job).where(Job.status.in_([JobStatus.UNSEEN.value, JobStatus.SMASHING.value]))
        )
        .scalars()
        .all()
    )

    expired = 0
    for job in active_jobs:
        validation = await validate_link(job.final_url or job.raw_url)
        job.final_url = validation.final_url
        job.validation_note = validation.reason
        if validation.is_dead:
            job.status = JobStatus.EXPIRED.value
            expired += 1

    db.commit()
    return RevalidateResult(checked=len(active_jobs), expired=expired)
