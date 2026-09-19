from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.database import get_db
from app.routers.jobs import _get_master_cv_or_400, _tailoring_failure_detail
from app.services.cv_tailor import TailoringError
from app.services.inbox_scraper import (
    InboxError, WORKFLOW_LOCK, fetch_latest_referral_jd,
)
from app.services.tailor import generate_tailored_resume

router = APIRouter(prefix="/api/inbox", tags=["inbox"])


class Referral(BaseModel):
    job_title: str = Field(min_length=1)
    sender: str = Field(min_length=1)
    jd_text: str
    duration: str = ""


def _invalid_referral(referral):
    if referral is None or len(referral.jd_text.strip()) < 50:
        return JSONResponse(status_code=422, content={"detail": (
            "No unread referral emails were found in your inbox. Please try again."
            if referral is None else
            "No valid job description text found in the latest unread referral email. Marked as read. Please try again."
        )})
    return None


async def _extract():
    if not WORKFLOW_LOCK.acquire(blocking=False):
        raise HTTPException(409, "An email referral is already being processed. Please wait.")
    try:
        data = await run_in_threadpool(fetch_latest_referral_jd)
        return Referral(**data) if data is not None else None
    except InboxError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        WORKFLOW_LOCK.release()


@router.post("/extract-latest", response_model=Referral)
async def extract_latest_email(db: Session = Depends(get_db)):
    _get_master_cv_or_400(db)
    referral = await _extract()
    return _invalid_referral(referral) or referral


async def _generate(referral, master):
    invalid = _invalid_referral(referral)
    if invalid is not None:
        return invalid
    try:
        pdf = await generate_tailored_resume(referral.jd_text, master, job_title=referral.job_title)
        return Response(pdf, media_type="application/pdf", headers={
            "Content-Disposition": 'inline; filename="Tailored_Email_Referral.pdf"',
            "Cache-Control": "no-store",
        })
    except TailoringError as exc:
        raise HTTPException(502, _tailoring_failure_detail(exc) + " The email was marked as read. Mark it unread in Gmail to retry.") from exc


@router.post("/tailor")
async def tailor_extracted_email(referral: Referral, db: Session = Depends(get_db)):
    return await _generate(referral, _get_master_cv_or_400(db))


@router.post("/tailor-latest")
async def tailor_latest_email(db: Session = Depends(get_db)):
    """Compatibility endpoint; the UI uses two requests to show extraction first."""
    master = _get_master_cv_or_400(db)
    return await _generate(await _extract(), master)
