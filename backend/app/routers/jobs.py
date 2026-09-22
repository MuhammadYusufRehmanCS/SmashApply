import json
import logging
from uuid import uuid4
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Job, MasterCV
from app.roles import PRIMARY_ROLE_DEFAULT
from app.schemas import JobOut, ScrapeRequest, ScrapeResult, TailorResult
from app.services.cv_tailor import TailorCVResult, TailoringError, LLMExecutionError, compute_match_score, tailor_cv, has_reframed_experience
from app.services.job_scraper import job_dedupe_keys, scrape_for_roles, scrape_role_names, cv_search_roles
from app.services.pdf_generator import CVOverflowError, build_ats_pdf
from app.services.cv_fitting import fit_tailored_cv
from starlette.concurrency import run_in_threadpool

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

# Default initials prefix for generated CV filenames (e.g. "MYR_Google.pdf").
DEFAULT_USER_INITIALS = "MYR"


@router.get("", response_model=list[JobOut])
def list_jobs(db: Session = Depends(get_db)):
    stmt = select(Job).order_by(Job.created_at.desc(), Job.id.desc())
    return db.execute(stmt).scalars().all()


@router.post("/scrape", response_model=ScrapeResult)
async def scrape_jobs_endpoint(payload: ScrapeRequest, db: Session = Depends(get_db)):
    primary_role = (payload.primary_role or PRIMARY_ROLE_DEFAULT).strip() or PRIMARY_ROLE_DEFAULT
    location = (payload.location or "Remote").strip() or "Remote"

    master = _get_master_cv_or_400(db)

    # Dedupe against existing DB rows and same-batch duplicates. Provider URLs
    # can differ for the same posting, so treat a normalized URL OR normalized
    # company/title pair as the duplicate key.
    existing_keys = set()
    for title, company, job_url in db.execute(select(Job.title, Job.company, Job.job_url)).all():
        existing_keys.update(job_dedupe_keys({"title": title, "company": company, "job_url": job_url}))

    found, site_errors = await scrape_for_roles(
        primary_role, location, master_text=master.raw_text, existing_keys=existing_keys,
    )

    created = 0
    skipped = 0
    for item in found:
        item_keys = job_dedupe_keys(item)
        if item_keys and existing_keys.intersection(item_keys):
            skipped += 1
            continue
        existing_keys.update(item_keys)

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
        roles_queried=cv_search_roles(master.raw_text, primary_role),
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


def _has_cached_tailoring(job: Job, cv: MasterCV | None = None) -> bool:
    present = bool((job.tailored_cv or "").strip() and (job.tailored_keywords or "").strip())
    if present and cv is not None:
        from app.services.tailor import is_finalized_master, _validate
        from app.services.cv_tailor import template_context_from_text, _TailoredPayload
        from pydantic import ValidationError
        if is_finalized_master(cv):
            try:
                context = template_context_from_text(job.tailored_cv)
                _validate(_TailoredPayload(role_title=context["role_title"], summary=context["summary"],
                                           core_skills=context["core_skills"],
                                           experience_bullets=context["experience_bullets"]), cv.raw_text, job.description or job.title)
            except (TailoringError, ValidationError, KeyError):
                return False
    return present and (cv is None or has_reframed_experience(job.tailored_cv, json.loads(cv.sections_json)))


def _tailoring_failure_detail(exc: Exception) -> str:
    """Report the failed stage without exposing provider bodies or CV text."""
    chain = []
    current = exc
    while current is not None and all(current is not item for item in chain):
        chain.append(current)
        current = current.__cause__
    if any(isinstance(item, LLMExecutionError) for item in chain):
        from openai import APIConnectionError, APITimeoutError
        if any(isinstance(item, APITimeoutError) for item in chain):
            return "The OpenAI request timed out. Please retry when the connection is stable."
        if any(isinstance(item, APIConnectionError) for item in chain):
            return ("The backend could not connect to OpenAI. Run it with normal network access "
                    "and check firewall, proxy or VPN settings. No model response was received.")
        status = next((getattr(item, "status_code", None) for item in chain
                       if getattr(item, "status_code", None)), None)
        if status == 401:
            return "OpenAI rejected the API key. Check OPENAI_API_KEY in backend/.env."
        if status == 429:
            from app.services.openai_retry import is_quota_error
            if any(is_quota_error(item) for item in chain):
                return "OpenAI reported insufficient API quota. Check the API project's credit balance and spending limit."
            return ("OpenAI's temporary request/token rate limit persisted after waiting and retrying. "
                    "This is not a billing rejection. Please wait briefly before trying again.")
        if status in (400, 403, 404):
            return "OpenAI rejected the configured model or request. Check model access and the backend error log."
        return "The OpenAI generation request failed or returned no usable response. Check the backend error log."
    from app.services.cv_tailor import PayloadFormatError
    from app.services.tailor import FinalizedValidationError
    for item in reversed(chain):
        if isinstance(item, (FinalizedValidationError, PayloadFormatError)):
            return "Generated CV validation failed: " + str(item)
    if any(isinstance(item, CVOverflowError) for item in chain):
        return "Generated wording still exceeds one page in the fixed PDF layout after the repair budget was exhausted."
    # These messages originate in our structural validator, never in model prose.
    for item in reversed(chain):
        message = str(item)
        if message.startswith(("Model returned ", "Model did not return ",
                               "Could not parse Professional Experience", "Master CV has no parsed sections",
                               "Upload the finalized Master CV", "Master CV has no identity header",
                               "Master CV has no Languages line", "Finalized CV did not pass")):
            return message
    return "Generated CV failed structure validation. Check the backend error log for the rejected field."


async def _run_tailor(job: Job, cv: MasterCV, db: Session, *, allow_fallback: bool = False,
                      candidate: TailorCVResult | None = None) -> TailorCVResult:
    try:
        result = candidate or await tailor_cv(
            cv,
            job.title,
            job.company,
            job.description or job.title,
            allow_fallback=allow_fallback,
        )
        result, _ = await fit_tailored_cv(result, cv, job.title, job.company, job.description or job.title)
        if result.cacheable:
            job.tailored_cv = result.text
            job.tailored_keywords = ", ".join(result.keywords)
            job.tailored_at = datetime.now(timezone.utc)
            job.match_score = compute_match_score(result.keywords, result.text)
            db.commit()
            db.refresh(job)
        elif not _has_cached_tailoring(job):
            # Do not let a fallback Master CV get cached as if it were a
            # tailored result. This also clears fallback rows written by older
            # versions where tailored_cv was populated but tailored_keywords
            # stayed blank.
            job.tailored_cv = None
            job.tailored_keywords = None
            job.tailored_at = None
            job.match_score = None
            db.commit()
            db.refresh(job)
    except TailoringError as exc:
        # Never cache or deliver a CV that failed the wording or page-fit checks.
        error_id = uuid4().hex[:12]
        logging.exception("Tailoring failed for job %s [error %s]", job.id, error_id)
        db.rollback()
        # A failed regeneration must not take away a previously completed CV.
        # Only reuse this job's own validated, reframed, one-page version.
        if _has_cached_tailoring(job, cv):
            try:
                await run_in_threadpool(build_ats_pdf, job.tailored_cv)
            except CVOverflowError:
                pass
            else:
                logging.warning("Returning the previously completed CV for job %s after refresh failed", job.id)
                return TailorCVResult(
                    keywords=[k.strip() for k in job.tailored_keywords.split(",") if k.strip()],
                    text=job.tailored_cv, cacheable=False, used_fallback=True,
                    warning=f"{_tailoring_failure_detail(exc)} No new resume was generated; your saved resume is unchanged. Error reference: {error_id}.",
                )
        raise HTTPException(
            status_code=502,
            detail=f"{_tailoring_failure_detail(exc)} Your saved CV and design were preserved. Error reference: {error_id}.",
        ) from exc
    except Exception as exc:
        # Anything else (a bug in our own reconstruction code, a bad DB
        # write, ...) would otherwise propagate as an unhandled crash, which
        # browsers surface as a bare "TypeError: Failed to fetch" instead of
        # a real error message.
        logging.exception("Tailoring failed")
        db.rollback()
        raise HTTPException(status_code=500, detail="CV Tailoring failed. Please retry.") from exc

    return result


@router.post("/{job_id}/tailor", response_model=TailorResult)
async def tailor_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    cv = _get_master_cv_or_400(db)
    result = await _run_tailor(job, cv, db, allow_fallback=True)

    return TailorResult(job_id=job.id, keywords=result.keywords, tailored_cv=result.text,
                        used_fallback=result.used_fallback,
                        warning=(result.warning or "New tailoring failed. Your previously saved resume is still available, but no new resume was generated."
                                 if result.used_fallback else None))


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
    cv_text = job.tailored_cv
    template_data = None
    if not _has_cached_tailoring(job, cv):
        result = await _run_tailor(job, cv, db, allow_fallback=True)
        cv_text = result.text
        template_data = result.template_data

    try:
        pdf_bytes = await run_in_threadpool(build_ats_pdf, template_data or cv_text or cv.raw_text)
    except CVOverflowError:
        # Existing cached CVs may predate the page-fit check. Revise their
        # wording automatically, and commit only after a successful render.
        candidate = TailorCVResult(
            keywords=[k.strip() for k in (job.tailored_keywords or "").split(",") if k.strip()],
            text=cv_text, cacheable=True, used_fallback=False,
        )
        result = await _run_tailor(job, cv, db, candidate=candidate)
        pdf_bytes = await run_in_threadpool(build_ats_pdf, result.text)

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
