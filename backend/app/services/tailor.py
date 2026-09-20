"""Finalized Master CV contract for email-to-resume generation.

Only editable prose goes to the model. Identity, employers, education,
certifications and languages come from the uploaded master, not the JD.
"""
import asyncio
import json
import logging
import re
from difflib import SequenceMatcher

from app.config import get_settings
from app.models import MasterCV
from app.services.cv_tailor import (
    LLMExecutionError, TailoringError, _request_tailored_payload,
    _split_experience_entries, SYSTEM_PROMPT, TailorCVResult, short_role_title,
)
from app.services.pdf_generator import CVOverflowError, build_ats_pdf
from app.services.text_sections import strip_bullet

PROJECTS = (
    "Selected Project: Release Automation System -",
    "Selected Project: Automated Infrastructure Provisioning -",
)
LEADERSHIP = "Leadership & Cross-Functional Collaboration:"
RULEBOOK = SYSTEM_PROMPT

# Aliases share a canonical identity so AWS/Amazon Web Services and K8s/Kubernetes
# cannot bypass duplicate checks. Metadata keywords supplement this vocabulary.
TECHNOLOGIES = {
    "AWS": ("AWS", "Amazon Web Services"), "Azure": ("Azure", "Microsoft Azure"),
    "GCP": ("GCP", "Google Cloud", "Google Cloud Platform"),
    "Kubernetes": ("Kubernetes", "K8s"), "Terraform": ("Terraform",),
    "Docker": ("Docker",), "Jenkins": ("Jenkins",), "Ansible": ("Ansible",),
    "Python": ("Python",), "Bash": ("Bash",), "Helm": ("Helm",),
    "GitHub Actions": ("GitHub Actions",), "SonarQube": ("SonarQube",),
    "FluxCD": ("FluxCD", "Flux CD"), "ArgoCD": ("ArgoCD", "Argo CD"),
    "ECR": ("ECR",), "ACR": ("ACR",), "GitLab": ("GitLab",),
    "Prometheus": ("Prometheus",), "Grafana": ("Grafana",),
    "PowerShell": ("PowerShell",), "SQL": ("SQL",), "Linux": ("Linux",),
    "Snowflake": ("Snowflake",), "Databricks": ("Databricks",),
    "Kafka": ("Kafka", "Apache Kafka"), "Airflow": ("Airflow", "Apache Airflow"),
    "Splunk": ("Splunk",), "Sentinel": ("Microsoft Sentinel", "Azure Sentinel"),
    "SOC 2": ("SOC 2", "SOC2"), "ISO 27001": ("ISO 27001", "ISO-27001"),
    "NIST": ("NIST",), "HIPAA": ("HIPAA",), "PCI DSS": ("PCI DSS", "PCI-DSS"),
}


def _clean(text):
    return re.sub(r"\s+", " ", text.replace("**", "").replace("\u2013", "-").replace("\u2014", "-")).strip()


def _source_context(master):
    sections = json.loads(master.sections_json)
    def section(hint):
        return next((s["content"] for s in sections if hint in s["name"].lower()), "")
    entries = _split_experience_entries(section("experience"))
    if not entries or len(entries) != 2 or any(
        name not in entry["header_line"].lower()
        for name, entry in zip(("arqon consulting", "ventera group"), entries)
    ):
        raise TailoringError("Upload the finalized Master CV with Arqon Consulting followed by Ventera Group.")
    header = [line.strip() for line in section("header").splitlines()
              if line.strip() and not re.fullmatch(r"\d+/\d+", line.strip())]
    if not header:
        raise TailoringError("Master CV has no identity header.")
    # The old header contained a technology banner and duplicate education/citizenship.
    contact = [" | ".join(part.strip() for part in line.split("|")
                if not re.search(r"A\.S\.?|Citizenship:|Work Authorization:", part, re.I))
               for line in header[1:]]
    education = section("education")
    lines = [_clean(strip_bullet(line)) for line in education.splitlines() if line.strip()]
    degree = next((line for line in lines if re.search(r"A\.S\.?\s+Computer Science", line, re.I)), "")
    degree = re.sub(r"\s*[,|(-]*\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+)?(?:19|20)\d{2}.*$", "", degree, flags=re.I).rstrip(" ,|(-")
    degree = re.sub(r"^A\.S\.?\s+Computer Science\s*(?:&|and)\s*Engineering", "A.S. Computer Science & Engineering", degree, flags=re.I)
    cert_index = next((i for i, line in enumerate(lines) if line.startswith("Certifications:")), None)
    if not degree or cert_index is None:
        raise TailoringError("Upload the finalized Master CV with the A.S. degree and a Certifications line (without a tooling list).")
    certifications = lines[cert_index]
    for line in lines[cert_index + 1:]:
        if re.match(r"(?:Tools?|Tooling|Technologies|Core Strengths)\s*:", line, re.I):
            break
        certifications += " " + line
    certifications = re.split(r"\s*[|;]\s*(?:Tools?|Tooling|Technologies)\s*:", certifications, flags=re.I)[0]
    languages = next((_clean(strip_bullet(line)) for line in section("additional").splitlines()
                      if strip_bullet(line).strip().startswith("Languages:")), "")
    if not languages:
        raise TailoringError("Master CV has no Languages line.")
    return dict(finalized=True, name=_clean(header[0].split("|")[0]),
                contact=[_clean(line) for line in contact if line],
                experience=[dict(header_line=_clean(e["header_line"])) for e in entries],
                education=[degree, certifications], languages=languages)


def _mentions(text, aliases):
    return list(re.finditer(r"(?<![\w])(?:" + "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True)) + r")(?![\w])", text, re.I))


class FinalizedValidationError(TailoringError):
    """A locally generated, safe-to-display validation reason."""


def _normalized_payload(payload):
    # The prompt permits Markdown emphasis. Labels and project separators are
    # structural text; typography must not make an otherwise valid JSON fail.
    def label(text):
        head, sep, rest = text.partition(":")
        if not sep:
            return text
        return head.replace("**", "").strip() + sep + " " + rest.lstrip("* ")

    groups = []
    for group in payload.experience_bullets:
        bullets = []
        for text in group:
            if text.replace("**", "").strip().startswith("Selected Project:"):
                text = text.replace("**", "").strip()
                for prefix in PROJECTS:
                    name = prefix[:-1].rstrip()
                    text = re.sub(r"^" + re.escape(name) + r"\s*[-\u2013\u2014]\s*", prefix + " ", text)
            bullets.append(text)
        groups.append(bullets)
    return payload.model_copy(update={
        "core_skills": [label(s) for s in payload.core_skills],
        "experience_bullets": groups,
    })


def _validate(payload, master_text: str | None = None):
    payload = _normalized_payload(payload)
    if not payload.role_title.strip() or len(payload.role_title.split()) > 4 or len(payload.role_title) > 32 or "\n" in payload.role_title:
        raise FinalizedValidationError("Role title must be 1-4 words and at most 32 characters on one line.")
    if not 1 <= len(payload.summary.split()) <= 25 or "\n" in payload.summary:
        raise FinalizedValidationError("Executive Summary must be one paragraph of at most 25 words.")
    skills = payload.core_skills
    if len(skills) != 3 or any(":" not in s or not s.split(":", 1)[1].strip() for s in skills):
        raise FinalizedValidationError("Core Skills must contain exactly three labeled, nonempty bullets.")
    if not skills[2].startswith(LEADERSHIP):
        raise FinalizedValidationError("Core Skills bullet 3 must cover Leadership & Cross-Functional Collaboration.")
    if skills[0].split(":", 1)[0].casefold() == skills[1].split(":", 1)[0].casefold():
        raise FinalizedValidationError("Core Skills domain labels must be distinct.")
    if any(len(_clean(skill).split()) > 28 for skill in skills):
        raise FinalizedValidationError("Core Skills bullets must be at most 28 words each.")
    groups = payload.experience_bullets
    if len(groups) != 2 or [len(g) for g in groups] != [4, 3]:
        raise FinalizedValidationError("Experience must have exactly four Arqon and three Ventera bullets.")
    for group, prefix in zip(groups, PROJECTS):
        if not group[-1].startswith(prefix) or not group[-1][len(prefix):].strip():
            raise FinalizedValidationError("Selected Project prefix or project description is missing: " + prefix)
        if any("selected project:" in bullet.lower() for bullet in group[:-1]):
            raise FinalizedValidationError("Selected Projects must occur only in the final employer bullet.")
    for group in groups:
        for index, bullet in enumerate(group):
            if len(_clean(bullet).split()) > (35 if index == len(group) - 1 else 30):
                raise FinalizedValidationError("Experience word budget exceeded: general bullets 30 words, projects 35.")
    prose = [payload.summary, *skills, *groups[0], *groups[1]]
    if any(not text.strip() or "\n" in text for text in prose):
        raise FinalizedValidationError("Every bullet must be nonempty, single-paragraph text.")
    # Exact and near-duplicate claim checks remove category/project labels first.
    claims = [re.sub(r"[^a-z0-9 ]", "", _clean(text.split(":", 1)[-1]).lower()) for text in prose]
    for index, claim in enumerate(claims):
        for other in claims[:index]:
            if SequenceMatcher(None, claim, other).ratio() >= 0.82:
                raise FinalizedValidationError("Duplicate or near-duplicate claim across resume sections.")
    vocabulary = dict(TECHNOLOGIES)
    repeated = [name for name, aliases in vocabulary.items()
                if sum(len(_mentions(text, aliases)) for text in [payload.role_title, *prose]) > 1]
    if repeated:
        raise FinalizedValidationError(
            "Repeated technology; keep each in its Selected Project when applicable: " + ", ".join(repeated))
    if master_text is not None:
        generated = " ".join(prose)
        source = _clean(master_text)
        for name, aliases in vocabulary.items():
            if _mentions(generated, aliases) and not _mentions(source, aliases):
                raise FinalizedValidationError("Technology claim is not supported by the Master CV: " + name)
        # Deterministically block newly invented quantitative claims. Semantic
        # equivalence still relies on the grounded prompt, not an ATS score.
        numbers = r"(?<![\w])\d+(?:\.\d+)?"
        if set(re.findall(numbers, generated)) - set(re.findall(numbers, source)):
            raise FinalizedValidationError("Generated metrics contain numbers absent from the Master CV.")
        standards = r"\b(?:SOC\s*2|ISO[ -]*27001|NIST(?:[ -]+(?:\d+-\d+|CSF|RMF))?|HIPAA|PCI[ -]*DSS|FedRAMP)\b"
        for match in re.finditer(standards, generated, re.I):
            if not _mentions(source, (match[0],)):
                raise FinalizedValidationError("Compliance/framework claim is not supported by the Master CV: " + match[0])
    metrics = re.findall(r"\b\d+(?:\.\d+)?\s*%", " ".join(prose))
    if len(metrics) != len(set(metrics)):
        raise FinalizedValidationError("Repeated quantified accomplishment across sections.")


def _context(source, payload, master_text=None):
    payload = _normalized_payload(payload).model_copy(update={"role_title": short_role_title(payload.role_title)})
    _validate(payload, master_text)
    return dict(source, role_title=_clean(payload.role_title), summary=_clean(payload.summary),
                core_skills=[re.sub(r"\s+", " ", s).strip() for s in payload.core_skills],
                experience=[dict(entry, bullets=[re.sub(r"\s+", " ", b).strip() for b in bullets])
                            for entry, bullets in zip(source["experience"], payload.experience_bullets)])


def is_finalized_master(master):
    sections = json.loads(master.sections_json)
    experience = " ".join(s["content"] for s in sections if "experience" in s["name"].lower()).lower()
    return "arqon consulting" in experience and "ventera group" in experience


def _serialize(context):
    blocks = [context["name"] + " | " + context["role_title"] + "\n" + "\n".join(context["contact"]),
              "EXECUTIVE SUMMARY\n" + context["summary"],
              "CORE SKILLS\n" + "\n".join("- " + s for s in context["core_skills"])]
    blocks.append("PROFESSIONAL EXPERIENCE\n" + "\n".join(
        e["header_line"] + "\n" + "\n".join("- " + b for b in e["bullets"]) for e in context["experience"]))
    blocks.append("EDUCATION & CERTIFICATIONS\n" + "\n".join("- " + line for line in context["education"]))
    blocks.append("ADDITIONAL INFORMATION\n- " + context["languages"] +
                  "\n- Work Authorization: United States Citizen (No sponsorship required)")
    return "\n\n".join(blocks)


async def generate_tailored_result(raw_email: str, master: MasterCV, job_title: str = "Email referral"):
    """Shared generation path for email and job-list tailoring."""
    source = _source_context(master)
    prompt = json.dumps({"target_role": job_title, "job_description": raw_email,
                         "verified_master_cv": master.raw_text}, ensure_ascii=False)
    base_prompt = prompt
    last_error = None
    for attempt in range(3):
        payload = None
        try:
            payload = await _request_tailored_payload(get_settings(), prompt, system_prompt=RULEBOOK)
            context = _context(source, payload, master.raw_text)
            pdf = await asyncio.to_thread(build_ats_pdf, context)
            from app.services.cv_tailor import jd_keywords
            keywords = jd_keywords(raw_email, payload.keywords)
            return TailorCVResult(keywords, _serialize(context), True, False), pdf
        except LLMExecutionError:
            raise
        except (TailoringError, CVOverflowError) as exc:
            last_error = exc
            logging.getLogger(__name__).warning("Finalized CV attempt %s/3 rejected (%s): %s", attempt + 1, type(exc).__name__, exc)
            if attempt < 2:
                prompt = base_prompt + "\nCorrect the previous validation failure: " + str(exc)
                if payload is not None:
                    prompt += "\nPrevious candidate: " + payload.model_dump_json()
                prompt += "\nReturn complete corrected JSON. Shorten wording as needed; retain every required bullet and all facts."
    raise TailoringError("Finalized CV did not pass structure, deduplication or one-page validation after three attempts.") from last_error


async def generate_tailored_resume(raw_email: str, master: MasterCV, job_title: str = "Email referral") -> bytes:
    _, pdf = await generate_tailored_result(raw_email, master, job_title)
    return pdf
