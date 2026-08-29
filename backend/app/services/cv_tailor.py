"""Extracts target keywords from a job description and rewords the Master
CV's bullet points to mirror them, via a local Ollama model. The prompt
forbids inventing or altering facts (employers, dates, titles, metrics) --
only phrasing/emphasis/ordering may change.
"""
import re

import httpx

from app.config import get_settings

SYSTEM_PROMPT = """You are a careful ATS resume editor. You will be given a target job \
description and a candidate's master CV. Do two things:

1. Extract the specific cloud/DevOps tools, platforms, and keywords the job description \
is looking for (e.g. AWS, GCP, Azure, Terraform, Kubernetes, Docker, Jenkins, GitHub Actions, \
CI/CD, Ansible, Prometheus, IAM, VPC, etc.) -- only ones actually implied by the job description.
2. Rewrite the CV's bullet points so their phrasing and emphasis mirror those keywords and the \
job's priorities.

Hard rules:
- Do NOT invent, remove, or change any factual detail: employers, job titles, dates, degrees, \
certifications, or metrics must stay exactly as given.
- Do NOT fabricate new experience, skills, tools, or achievements the candidate didn't list.
- You MAY reorder bullet points, tighten wording, and swap in synonyms/keywords from the job \
description where they truthfully describe existing experience.
- Preserve the CV's original section headers exactly (same text, same order) so the layout can \
be reproduced.
- Keep every bullet point on its own line with a leading "-" marker (convert any other bullet \
symbol like "•" to "-"), so the layout can be reproduced.

Respond in EXACTLY this format, with no extra commentary and no markdown fences:

KEYWORDS: comma, separated, list, of, extracted, keywords
---TAILORED CV---
<full tailored CV text here, section headers included>
"""


def _parse_response(text: str, fallback_cv: str) -> tuple[list[str], str]:
    match = re.search(r"KEYWORDS:\s*(.*?)\n---TAILORED CV---\n(.*)", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return [], text.strip() or fallback_cv

    keywords_raw, tailored = match.group(1), match.group(2)
    keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]
    tailored = tailored.strip() or fallback_cv
    return keywords, tailored


async def tailor_cv(raw_cv_text: str, job_description: str) -> tuple[list[str], str]:
    """Returns (extracted_keywords, tailored_cv_text)."""
    settings = get_settings()
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- JOB DESCRIPTION ---\n{job_description.strip()}\n\n"
        f"--- MASTER CV ---\n{raw_cv_text.strip()}\n"
    )

    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(f"{settings.ollama_base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            raw_output = (data.get("response") or "").strip()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Could not reach local Ollama instance at {settings.ollama_base_url}: {exc}. "
            f"Is `ollama serve` running with the '{settings.ollama_model}' model pulled?"
        ) from exc

    if not raw_output:
        return [], raw_cv_text

    return _parse_response(raw_output, raw_cv_text)
