"""Extracts target keywords from a job description and rewords the Master
CV's Executive Summary, Technical Expertise, and Professional Experience
bullets to mirror them, via a local Ollama model.

Works for ANY scraped role -- DevOps, Platform Engineering, Cloud Engineering,
SRE, Data Engineering, etc. -- by having the model dynamically infer the role
category and required tools from that specific job's title/company/description
rather than assuming a fixed role type. The prompt forbids inventing or
altering facts (employers, dates, titles, metrics, certifications) -- only
phrasing/emphasis/ordering may change.
"""
import re

import httpx

from app.config import get_settings
from app.services.text_sections import is_bullet_line, strip_bullet

SYSTEM_PROMPT = """You are an expert ATS resume strategist for infrastructure, cloud, and data \
careers -- DevOps, Platform Engineering, Site Reliability Engineering, Cloud Engineering, Data/\
Database Engineering, and adjacent roles. You will be given a specific target job (its title, \
hiring company, and full job description) and a candidate's master CV. You must tailor the CV so \
the candidate reads as a top-tier, purpose-built match for THAT exact job -- not a generic \
reshuffle applied to every job the same way.

Silently reason through two internal steps before you write your final answer -- these steps are \
for your own thinking only; NEVER write their names, numbers, or any part of your analysis into \
your final answer. Your final answer contains ONLY the exact "KEYWORDS: ... / ---TAILORED CV---" \
format shown at the very end of this prompt, nothing else, no matter how you reasoned to get there.

Internal step one -- analyze the target job description on its own merits:
- Determine the actual role category this specific posting is for (it may be DevOps, Platform \
Engineering, Cloud Engineering, SRE, Data Engineering, or something else entirely -- infer it \
fresh from this job's title and description, never assume).
- Extract the specific tools, languages, cloud platforms, and methodologies THIS job description \
actually names or clearly implies (e.g. AWS/GCP/Azure, Terraform, Kubernetes, Docker, Jenkins, \
GitHub Actions/GitLab CI, CI/CD, Ansible, Prometheus/Grafana, Python/Go/Bash, SQL, Airflow, \
Snowflake, IAM, VPC, incident response/on-call, internal developer platforms, etc.) -- only ones \
actually implied by THIS job description, not a generic list reused across postings.

Internal step two -- rewrite the CV to mirror that analysis:
- Executive Summary / Professional Summary: rewrite it so it explicitly positions the candidate \
as an ideal, top-tier fit for THIS role category at THIS company, naturally working in the 3-6 \
most important tools/keywords you extracted in Pass 1.
- Technical Expertise / Skills section: reorder and rephrase the EXISTING lines so the tools/\
keywords most relevant to this job description are foregrounded first. Keep the exact same number \
of lines/categories as the master CV -- do not add a new category line (even a truthful one built \
from real skills) and do not remove one. You may only surface skills the candidate already lists \
elsewhere in the master CV -- never introduce a tool the candidate doesn't have.
- Professional Experience bullet points, under EVERY employer listed in the CV (rewrite the \
bullets for all of them, not just one -- whichever companies those happen to be for this \
candidate): reframe each EXISTING bullet's phrasing and emphasis around the specific tools/\
keywords from this job description wherever it genuinely, truthfully applies. Match the emphasis \
to the inferred role category, e.g. lean into IaC and CI/CD pipelines for a DevOps posting, \
Kubernetes and internal developer platforms for a Platform Engineering posting, monitoring/SLOs/\
incident response for an SRE posting, or database/query tuning and data pipelines for a Data/\
Cloud posting.
- Emphasis formatting: wrap 2-4 of the most scannable keywords/tools/metrics PER bullet in \
**double asterisks** (e.g. "Automated infrastructure using **Terraform**, reducing manual \
deployment steps by **80%**"), the way a well-formatted, ATS-safe resume bolds the phrases a \
recruiter's eye should catch first. Bold short phrases only (a tool name, a platform, a \
percentage/metric, a 2-4 word impact phrase) -- never bold an entire sentence or bullet. Bold \
words that are ALREADY part of the bullet's existing sentence -- do NOT grow the bullet by \
tacking a new trailing clause onto the end just to have something to bold (e.g. never append \
", focusing on **X** and **Y**" or ", leveraging **X**, **Y**, and **Z**" or ", demonstrating \
**X** expertise" after the original sentence -- if the bullet doesn't already mention a keyword, \
leave it out rather than appending a clause that names it). Every bullet's sentence structure and \
length should stay close to the master CV's original -- only word choice and what gets bolded \
should change. Also \
bold the short category label at the start of each Technical Expertise line (e.g. "**Cloud & \
Infrastructure:**") but leave the tool list after the colon unbolded. Do NOT use **bold** \
markers anywhere in the Education/Certifications section. Never append a parenthetical keyword \
recap to the end of a Professional Experience bullet (e.g. never end a bullet in something like \
"(Kubernetes, Docker, CI/CD)") -- if a keyword belongs in the bullet, put it directly in the \
sentence and bold it there, don't tag a list onto the end.

Hard rules:
- Do NOT invent, remove, or change any factual detail: employers, job titles, dates, degrees, \
or metrics must stay exactly as given.
- STRICT CONSTRAINT: Do NOT add, remove, hallucinate, or alter anything in the Education/\
Certifications section. Every certification and credential must be reproduced EXACTLY as it \
appears in the master CV, in the same section, completely unchanged -- tailoring never touches \
this section, no matter how well a certification would seem to fit the target job.
- Do NOT fabricate new experience, skills, tools, or achievements the candidate didn't list.
- CRITICAL -- never confuse the job description's responsibilities with the candidate's own \
history: the job description describes what the EMPLOYER wants someone to do (often phrased as \
"you will...", "responsibilities include...", or imperative instructions like "collaborate with \
X" or "stay current on Y"). NONE of that language may be copied, paraphrased into a first bullet, \
or inserted as a new line under Professional Experience. Every single bullet in the tailored \
Professional Experience section must trace back to a specific bullet that already exists in the \
master CV for that same employer -- you are only allowed to reword that existing bullet's phrasing \
and keyword emphasis, never author a new one from the job posting's text.
- The tailored CV must have the EXACT SAME NUMBER of Professional Experience bullets per employer \
as the master CV -- never add extra bullets, never drop any, never merge or split one bullet into \
several. In particular, never tack on an extra closing/summary bullet at the end of a job's list \
(e.g. a generic line about "collaborating with cross-functional teams" or "delivering business \
value") that isn't a reworded version of a bullet the master CV already has for that employer.

Worked example of the ONLY kind of change allowed on a bullet (notice the count stays at 2, \
nothing is added, and only wording/emphasis changed to fit a hypothetical Kubernetes-heavy \
job description):
  Master CV had exactly these 2 bullets for an employer:
    - Automated infrastructure provisioning with Terraform across 40+ AWS accounts
    - Deployed containerized workloads to production with zero downtime
  Correct tailored output for that same employer (still exactly 2 bullets):
    - Automated Infrastructure-as-Code provisioning with Terraform across 40+ AWS accounts
    - Deployed and orchestrated containerized workloads on Kubernetes with zero-downtime releases
  Incorrect (do NOT do this -- adds a 3rd bullet that wasn't in the master CV):
    - Automated Infrastructure-as-Code provisioning with Terraform across 40+ AWS accounts
    - Deployed and orchestrated containerized workloads on Kubernetes with zero-downtime releases
    - Collaborated with cross-functional teams to support platform reliability goals
  Also incorrect (do NOT do this -- never append a parenthetical explaining your own edit; a \
bullet is ONLY the resume line itself, never commentary about how or why you changed it):
    - Automated Infrastructure-as-Code provisioning with Terraform across 40+ AWS accounts (reworded \
to emphasize IaC for this job)
  Also incorrect (do NOT do this -- never tack a recap keyword list onto the end of a bullet in \
parentheses; weave keywords into the sentence itself instead, don't tag them on afterward):
    - Deployed and orchestrated containerized workloads on Kubernetes with zero-downtime releases \
(Kubernetes, Docker, orchestration)

- You MAY reorder bullet points, tighten wording, and swap in synonyms/keywords from the job \
description where they truthfully describe existing experience.
- Preserve the CV's original section headers exactly (same text, same order) so the layout can \
be reproduced.
- Keep every bullet point on its own line with a leading "-" marker (convert any other bullet \
symbol like "•" to "-"), so the layout can be reproduced.
- Do NOT truncate, summarize, or skip any part of the CV. Reproduce the ENTIRE resume line-by-line \
in full, including every single bullet point under Professional Experience for every employer, \
every Education entry, every Certification, and every line of any Additional/Other section -- even \
if that makes the output long. Never shorten a section "for brevity."
- Do NOT add meta-commentary, notes, disclaimers, or placeholders of any kind, whether as their \
own line OR appended to the end of a bullet in parentheses -- for example never write things like \
"Note:", "[rest of CV remains the same]", "the remaining bullets are unchanged", "as an AI \
language model", or a bullet ending in "(reworded to...)", "(emphasizes...)", "(mirrors existing \
bullet...)", "(rewords existing bullet to highlight...)". A bullet is ONLY the resume line itself. \
Do NOT restate the extracted keyword list a second time inside the tailored CV body.
- Do NOT wrap any part of your response in markdown code fences (no ``` anywhere) and do not add \
any explanatory text before or after the required format.
- Your response starts IMMEDIATELY with the literal text "KEYWORDS:" as its very first characters \
-- not "Step one", not "Analysis:", not a blank line, not any other lead-in. If you catch yourself \
about to write anything else first, stop and start over with "KEYWORDS:" instead.

Respond in EXACTLY this format, with no extra commentary and no markdown fences:

KEYWORDS: comma, separated, list, of, extracted, keywords
---TAILORED CV---
<full tailored CV text here, section headers included, every line reproduced in full>
"""

# Lines the model sometimes adds despite the prompt (meta-commentary, disclaimers,
# truncation notices) -- stripped before the text ever reaches the PDF generator.
_CODE_FENCE_LINE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*$")
_META_LINE_PREFIX_RE = re.compile(
    r"(?i)^\s*[\[\(]?\s*(note|notes|disclaimer|reminder|important|keywords)\s*[:\-]"
)
_META_PHRASES = (
    "rest of the cv remains",
    "rest of cv remains",
    "remainder of the cv",
    "remains the same",
    "remains unchanged",
    "unchanged from",
    "not repeated here",
    "same as above",
    "omitted for brevity",
    "for brevity",
    "as an ai language model",
    "as an ai, i",
    "i cannot provide",
    "i can't provide",
    "here is the tailored",
    "here's the tailored",
)

# The model sometimes obeys "reword the bullet" but then appends a parenthetical
# explaining what it did, e.g. "...MTTR. (rewords existing bullet to highlight
# Terraform)". Despite explicit prompt instructions not to, this happens often
# enough on smaller local models that a regex safety net is needed -- strips a
# trailing parenthetical only when it reads as commentary about the edit itself
# (not a legitimate resume parenthetical like "(AWS, Azure, GCP)" or "(2023-2025)").
_INLINE_META_PAREN_RE = re.compile(
    r"\s*\((?:[^()]*\b(?:reword\w*|emphasiz\w*|highlight\w*|mirror\w*|tailor\w*|"
    r"existing\s+bullet|original\s+bullet|keyword\s+emphasis|align\w*\s+with|"
    r"per\s+the\s+job|for\s+this\s+(?:role|job|position))\b[^()]*)\)\s*$",
    re.IGNORECASE,
)


def _strip_inline_meta_commentary(line: str) -> str:
    while True:
        stripped_line = _INLINE_META_PAREN_RE.sub("", line)
        if stripped_line == line:
            return line.rstrip()
        line = stripped_line


# The system prompt requires every bullet to use a leading "-" (and
# explicitly says to convert any other bullet glyph like "•" to it), but
# smaller local models don't always comply -- and separately, Ollama
# occasionally emits U+FFFD (the Unicode replacement character) in place of
# the bullet glyph entirely, a tokenizer artifact since nothing in this
# pipeline itself performs a lossy decode (the bytes are already corrupted
# by the time httpx reads the response). Both cases are unambiguously "this
# line is a bullet using the wrong marker" -- normalize them to a clean "-"
# before the text reaches ReportLab or the API, rather than leaving a stray
# glyph (or a literal replacement-character box) in the resume/PDF/preview.
# "*" is deliberately excluded: a line can legitimately start with "**" for
# bold markdown (e.g. a bolded Technical Expertise category label), and
# treating that as a bullet marker would eat the opening ** and break the
# bold span instead of just fixing a bullet.
_NON_DASH_BULLET_CHARS = "•◦▪●‣·�"
_LEADING_BULLET_RE = re.compile(rf"^(\s*)[{_NON_DASH_BULLET_CHARS}]+\s*")


def _normalize_bullet_marker(line: str) -> str:
    return _LEADING_BULLET_RE.sub(r"\1- ", line)


def _dedupe_bullets(lines: list[str]) -> list[str]:
    """Drops a bullet line if its normalized text exactly duplicates an
    earlier bullet anywhere in the document. Small local models occasionally
    repeat a line verbatim during generation -- a resume should never
    legitimately have two identical bullets, so this is always safe to strip."""
    seen: set[str] = set()
    result = []
    for line in lines:
        if is_bullet_line(line):
            normalized = re.sub(r"\s+", " ", strip_bullet(line).replace("*", "").strip().lower())
            if normalized in seen:
                continue
            seen.add(normalized)
        result.append(line)
    return result


def _sanitize_tailored_text(text: str) -> str:
    """Strips stray markdown code fences and LLM conversational notes the model
    sometimes adds despite the system prompt, so only the resume body reaches
    ReportLab."""
    lines = []
    for line in text.splitlines():
        line = _normalize_bullet_marker(line)
        stripped = line.strip()
        if not stripped:
            lines.append(line)
            continue
        if _CODE_FENCE_LINE_RE.match(stripped):
            continue
        if _META_LINE_PREFIX_RE.match(stripped):
            continue
        lowered = stripped.lower()
        if any(phrase in lowered for phrase in _META_PHRASES):
            continue
        lines.append(_strip_inline_meta_commentary(line))

    lines = _dedupe_bullets(lines)
    cleaned = "\n".join(lines).replace("```", "").replace("`", "")
    return cleaned.strip()


def _keyword_present(keyword: str, tailored_text_lower: str) -> bool:
    """Whole-word match for plain-text keywords (so "Go" doesn't match inside
    "Google"), falling back to a plain substring check for keywords containing
    punctuation a word-boundary regex can't handle (e.g. "CI/CD", "Node.js")."""
    keyword_lower = keyword.strip().lower()
    if not keyword_lower:
        return False
    if re.fullmatch(r"[\w\s\-\+#]+", keyword_lower):
        return re.search(rf"\b{re.escape(keyword_lower)}\b", tailored_text_lower) is not None
    return keyword_lower in tailored_text_lower


def compute_match_score(keywords: list[str], tailored_text: str) -> int | None:
    """Returns 0-100: the percentage of the JD's extracted technical keywords
    that show up in the tailored CV -- a simple, deterministic proxy for how
    well the tailoring pass actually worked the target job's terms into the
    resume. Returns None when there are no keywords to score against."""
    if not keywords:
        return None
    tailored_text_lower = tailored_text.lower()
    matched = sum(1 for kw in keywords if _keyword_present(kw, tailored_text_lower))
    return round(100 * matched / len(keywords))


_KEYWORDS_LINE_RE = re.compile(r"(?im)^\s*keywords\s*:\s*(.+)$")
# Looser fallback anchor for when the model doesn't hit the exact "KEYWORDS:
# ...\n---TAILORED CV---\n" format (small local models drift on this despite
# the prompt) -- finds "tailored cv" as a standalone marker line, with any
# amount of surrounding dashes/colons, and takes everything after it. This
# still won't match a stray in-sentence mention like "the tailored CV should
# look professional" because that isn't followed by a line break.
_TAILORED_CV_MARKER_RE = re.compile(r"(?is)tailored\s*cv\s*[-:]*\s*\n+(.*)")


def _parse_response(text: str, fallback_cv: str) -> tuple[list[str], str]:
    match = re.search(r"KEYWORDS:\s*(.*?)\n---TAILORED CV---\n(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        keywords_raw, tailored = match.group(1), match.group(2)
        keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]
        tailored = _sanitize_tailored_text(tailored) or fallback_cv
        return keywords, tailored

    # Strict format didn't match -- the model likely added preamble/reasoning
    # before its answer. Recover the keyword list and the CV body independently
    # rather than treating the whole (preamble-polluted) response as the CV.
    keywords = []
    kw_match = _KEYWORDS_LINE_RE.search(text)
    if kw_match:
        keywords = [k.strip() for k in kw_match.group(1).split(",") if k.strip()]

    cv_match = _TAILORED_CV_MARKER_RE.search(text)
    if cv_match:
        return keywords, _sanitize_tailored_text(cv_match.group(1)) or fallback_cv

    return keywords, _sanitize_tailored_text(text) or fallback_cv


async def tailor_cv(
    raw_cv_text: str, job_title: str, company_name: str, job_description_text: str
) -> tuple[list[str], str]:
    """Returns (extracted_keywords, tailored_cv_text).

    job_title/company_name/job_description_text are passed as distinct fields
    (rather than one blob) so the prompt can hand Ollama a clearly labeled,
    per-job target to analyze -- this is what lets the same prompt tailor
    correctly for a DevOps role at one company and a Data Engineering role at
    the next, without any role-specific logic on the Python side.
    """
    settings = get_settings()
    job_title = (job_title or "").strip() or "the target role"
    company_name = (company_name or "").strip() or "the hiring company"
    job_description_text = (job_description_text or "").strip() or job_title

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- TARGET JOB ---\n"
        f"Job Title: {job_title}\n"
        f"Company: {company_name}\n"
        f"Job Description:\n{job_description_text}\n\n"
        f"--- MASTER CV ---\n{raw_cv_text.strip()}\n"
    )

    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 4096,
            "num_predict": 1024,
        },
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
