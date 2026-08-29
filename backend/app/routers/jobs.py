from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import CVMaster, Job, JobStatus
from app.schemas import JobOut, MetricsOut, SmashResponse
from app.services.cv_tailor import tailor_cv

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("", response_model=list[JobOut])
def list_jobs(status: str | None = None, db: Session = Depends(get_db)):
    stmt = select(Job)
    if status:
        statuses = [s.strip() for s in status.split(",") if s.strip()]
        stmt = stmt.where(Job.status.in_(statuses))
    stmt = stmt.order_by(Job.match_score.desc())
    return db.execute(stmt).scalars().all()


@router.get("/metrics", response_model=MetricsOut)
def get_metrics(db: Session = Depends(get_db)):
    rows = db.execute(select(Job.status, func.count(Job.id)).group_by(Job.status)).all()
    counts = {status: count for status, count in rows}
    smashed = counts.get(JobStatus.SMASHED.value, 0)
    passed = counts.get(JobStatus.PASSED.value, 0)
    active = counts.get(JobStatus.UNSEEN.value, 0) + counts.get(JobStatus.SMASHING.value, 0)
    total = sum(counts.values())
    return MetricsOut(smashed=smashed, passed=passed, active_pipeline=active, total=total)


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/{job_id}/pass", response_model=JobOut)
def pass_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.status = JobStatus.PASSED.value
    db.commit()
    db.refresh(job)
    return job


@router.post("/{job_id}/smash", response_model=SmashResponse)
async def smash_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    cv = db.execute(select(CVMaster).order_by(CVMaster.id.desc())).scalars().first()
    if not cv:
        raise HTTPException(
            status_code=400, detail="No master CV uploaded yet. Upload one via /api/cv first."
        )

    job.status = JobStatus.SMASHING.value
    db.commit()

    try:
        tailored = await tailor_cv(cv.raw_text, job.description)
    except RuntimeError as exc:
        job.status = JobStatus.UNSEEN.value
        db.commit()
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    job.tailored_cv = tailored
    job.status = JobStatus.SMASHED.value
    db.commit()
    db.refresh(job)

    apply_url = job.final_url or job.raw_url
    return SmashResponse(job=job, tailored_cv=tailored, apply_url=apply_url)
