"""Email entry point into the existing validated resume/PDF pipeline."""
from app.models import MasterCV
from app.services.cv_tailor import tailor_cv
from app.services.cv_fitting import fit_tailored_cv


async def generate_tailored_resume(raw_email: str, master: MasterCV, job_title: str = "Email referral") -> bytes:
    title, company = job_title, ""
    result = await tailor_cv(master, title, company, raw_email, allow_fallback=False)
    _, pdf = await fit_tailored_cv(result, master, title, company, raw_email)
    return pdf
