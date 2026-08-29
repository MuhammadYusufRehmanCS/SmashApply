"""Rewords CV bullet points to mirror a job description's keywords via a local
Ollama model. The prompt explicitly forbids inventing or altering facts
(employers, dates, titles, degrees) -- only phrasing/emphasis may change.
"""
import httpx

from app.config import get_settings

SYSTEM_PROMPT = """You are a careful resume editor. You will be given a candidate's \
master CV and a target job description. Rewrite the CV's bullet points so their \
phrasing and emphasis mirror the job description's key terms and priorities.

Hard rules:
- Do NOT invent, remove, or change any factual detail: employers, job titles, \
dates, degrees, certifications, or metrics must stay exactly as given.
- Do NOT fabricate new experience, skills, or achievements the candidate didn't list.
- You MAY reorder bullet points, tighten wording, and swap in synonyms/keywords \
from the job description where they truthfully describe existing experience.
- Preserve the overall structure and section headers of the input CV.
- Output only the tailored CV text -- no commentary, no markdown fences.
"""


async def tailor_cv(raw_cv_text: str, job_description: str) -> str:
    settings = get_settings()
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- JOB DESCRIPTION ---\n{job_description.strip()}\n\n"
        f"--- MASTER CV ---\n{raw_cv_text.strip()}\n\n"
        f"--- TAILORED CV ---\n"
    )

    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"{settings.ollama_base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            tailored = (data.get("response") or "").strip()
            return tailored or raw_cv_text
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Could not reach local Ollama instance at {settings.ollama_base_url}: {exc}. "
            "Is `ollama serve` running?"
        ) from exc
