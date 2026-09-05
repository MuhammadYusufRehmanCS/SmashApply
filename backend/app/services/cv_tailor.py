"""Rewrites a Master CV's Executive Summary, Technical Expertise, and
Professional Experience bullets to mirror a target job description, via
OpenAI Chat Completions. Every other section of the CV -- the Header/contact
line apart from the top resume headline title, Education, Certifications,
Additional, and anything else -- is reproduced byte-for-byte from the master
CV and is never sent to, or returned by, the model.

Anti-hallucination design: rather than relying on prompt instructions alone
to stop the model from inventing/altering employers, dates, titles, degrees,
or the surrounding document structure, those facts are structurally kept out
of the model's hands entirely --

  - The Professional Experience section is split (in Python, before any
    LLM call) into per-employer entries. Only each entry's BULLET TEXT is
    sent to the model; the entry's title/company/location/dates line is
    never sent and never comes back from the model -- it's spliced back in
    verbatim when reconstructing the tailored CV.
  - Technical Expertise is split the same way, into per-category entries.
    Only each category's tool-list TEXT is sent; the category LABEL (e.g.
    "Cloud & Infrastructure:") is immutable and spliced back in verbatim --
    the model sees it for context (so it tailors the right kind of tools
    into the right category) but never has to echo, and can never alter, it.
  - Any section that couldn't be confidently split this way (an atypically
    formatted resume), or that isn't Summary/Skills/Experience at all
    (Education, Certifications, Additional, ...), is passthrough, always,
    unconditionally -- reproduced byte-for-byte from the master CV. The only
    header exception is the top resume headline title, which may be replaced
    with the target job title while the name/contact line stays locked.
  - OpenAI is called with JSON mode, and the response is parsed and
    shape-validated (Pydantic) before it's allowed anywhere near the PDF
    renderer. Missing category rewrites, mismatched employer/bullet counts,
    unchanged bullets, leaked schema keys, and cliche summary starters are
    rejected or cleaned before reconstruction. A failed attempt gets one
    retry; if the model still cannot produce strict output,
    the service either salvages usable tailored JSON as a non-cacheable
    response or returns clean Master CV fallback text for PDF rendering.

Works for ANY scraped role -- DevOps, Platform Engineering, Cloud Engineering,
SRE, Data Engineering, etc. -- by having the model dynamically infer the role
category and required tools from that specific job's title/company/description
rather than assuming a fixed role type.
"""
import json
import logging
import re
from typing import Annotated, NamedTuple

from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, BeforeValidator, Field, ValidationError

from app.config import get_settings
from app.models import MasterCV
from app.services.text_sections import BULLET_PREFIXES, is_bullet_line, looks_like_entry_header, strip_bullet

_SUMMARY_HINTS = ("summary", "objective", "profile")
_SKILLS_HINTS = ("skill", "expertise", "competenc", "tools", "technolog")
_EXPERIENCE_HINTS = ("experience", "employment", "work history")


def _is_summary_section(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _SUMMARY_HINTS)


def _is_skills_section(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _SKILLS_HINTS)


def _is_experience_section(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _EXPERIENCE_HINTS)


def _split_experience_entries(content: str) -> list[dict] | None:
    """Splits a Professional Experience section's raw text into per-employer
    entries: {"header_line", "tagline", "bullets"}. header_line/tagline are
    the immutable title/company/location/dates line (and the optional
    one-line role blurb some resumes put before the bullets); bullets are
    the ONLY part that ever reaches the model.

    Returns None if no entry-header-shaped line was found at all -- an
    atypically-formatted section this heuristic can't confidently split.
    Callers must treat None as "don't touch this section" rather than
    guessing, since a wrong split could scramble which bullets belong to
    which employer.
    """
    lines = [ln for ln in content.splitlines() if ln.strip()]
    entries: list[dict] = []
    current: dict | None = None

    for line in lines:
        if looks_like_entry_header(line):
            current = {"header_line": line.strip(), "tagline": None, "bullets": []}
            entries.append(current)
            continue
        if current is None:
            # Content before any recognized entry header -- the format isn't
            # what this heuristic expects, so don't guess at a split.
            return None
        if is_bullet_line(line):
            current["bullets"].append(strip_bullet(line))
        elif current["tagline"] is None and not current["bullets"]:
            current["tagline"] = line.strip()
        elif current["bullets"]:
            current["bullets"][-1] = f"{current['bullets'][-1]} {line.strip()}".strip()
        else:
            current["tagline"] = f"{current['tagline']} {line.strip()}".strip()

    return entries or None


# "Cloud & Infrastructure: AWS, Azure, ..." -- label is everything before the
# first colon. Deliberately requires the label to be short (a category name,
# not a whole sentence that happens to contain a colon somewhere).
_CATEGORY_LINE_RE = re.compile(r"^(.{2,60}?):\s*(.+)$")


def _split_technical_expertise(content: str) -> list[dict] | None:
    """Splits a Technical Expertise section into per-category entries:
    {"prefix", "label", "items"}. prefix+label are the immutable bullet
    marker and category name; items is the ONLY part that ever reaches the
    model.

    A category's item list is often word-wrapped across several physical
    lines by the PDF's own text extraction (long tool lists commonly spill
    onto a second or third line) -- a non-bulleted line is treated as a
    continuation of the PREVIOUS category's items, not a new category.

    Returns None if a line can't be attributed to any category at all (no
    bulleted "Label: items" line has been seen yet, or a bulleted line
    doesn't contain a colon) -- an atypically-formatted section this
    heuristic can't confidently split, in which case the whole section
    stays untouched passthrough.
    """
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    entries: list[dict] = []
    for line in lines:
        if is_bullet_line(line):
            prefix = next((c for c in line if c in BULLET_PREFIXES), "-")
            match = _CATEGORY_LINE_RE.match(strip_bullet(line))
            if not match:
                return None
            label, items = match.groups()
            entries.append({"prefix": prefix, "label": label.strip(), "items": items.strip()})
        elif entries:
            entries[-1]["items"] = f"{entries[-1]['items']} {line}".strip()
        else:
            # Content before any recognized category line -- don't guess.
            return None
    return entries or None


# The model's own schema key names -- if any of these show up as a bare
# bullet string (e.g. the model emitted a stray 5th list item that's just
# "experience_bullets" instead of real bullet text, which is exactly the bug
# this guards against), it's unambiguously not real resume content.
_SCHEMA_KEY_NAMES = {
    "keywords",
    "summary",
    "executive_summary",
    "technical_expertise",
    "technical_expertise_items",
    "experience_bullets",
    "bullets",
    "items",
}
_BARE_IDENTIFIER_RE = re.compile(r"^[a-z][a-z_]{2,}$")


def _looks_like_leaked_key(text: str) -> bool:
    """True for anything that isn't real resume prose: a bare schema-key
    token, or (a real observed failure mode) a stringified JSON/Python list
    or dict leaking through as a field's literal text value -- e.g. the
    model returning "['bullet 1', 'bullet 2']" as a plain string for a field
    that should just be prose, rather than (or in addition to) the coercion
    in `_coerce_list_to_str` catching it as an actual JSON array. Used for
    every field the model returns: the summary, each Technical Expertise
    category's items, and every bullet."""
    raw_stripped = text.strip()
    if raw_stripped.startswith(("[", "{")) and raw_stripped.endswith(("]", "}")):
        return True
    stripped = raw_stripped.strip('"').strip().lower()
    if stripped in _SCHEMA_KEY_NAMES:
        return True
    # A real bullet is always prose -- spaces, capitals, punctuation. A bare
    # lowercase_and_underscores-only token with no spaces is unambiguously
    # not a sentence, regardless of whether it happens to match a key name.
    return bool(_BARE_IDENTIFIER_RE.fullmatch(stripped))


def _coerce_list_to_str(separator: str):
    """A small local model asked for a string field quite reasonably
    sometimes returns a JSON list instead (one item per sentence/tool/
    phrase) -- both are semantically fine, so accept either shape rather
    than failing validation over it. The right join separator depends on
    what the field represents: prose paragraphs read naturally joined by
    newlines, a tool list by ", ", and a single bullet's fragments by a
    plain space (it should read as one continuous sentence)."""

    def _coerce(value: object) -> object:
        if isinstance(value, list):
            return separator.join(str(item) for item in value)
        return value

    return _coerce


class _TailoredPayload(BaseModel):
    keywords: list[str] = Field(default_factory=list)
    summary: Annotated[str, BeforeValidator(_coerce_list_to_str("\n"))] = ""
    # One reworded items-string per Technical Expertise category, same order
    # as the categories this job was built from.
    technical_expertise: list[Annotated[str, BeforeValidator(_coerce_list_to_str(", "))]] = Field(
        default_factory=list
    )
    # One inner list of reworded bullets per employer, same order as the
    # employer entries this job was built from.
    experience_bullets: list[list[Annotated[str, BeforeValidator(_coerce_list_to_str(" "))]]] = Field(
        default_factory=list
    )


class TailoringError(RuntimeError):
    """Raised for local CV issues or for a single failed LLM attempt.
    `tailor_cv()` catches per-attempt model failures, retries once, and only
    returns non-cacheable fallback text after both attempts fail."""


class LLMExecutionError(TailoringError):
    """Raised when the OpenAI client cannot execute the tailoring request."""


class TailorCVResult(NamedTuple):
    keywords: list[str]
    text: str
    cacheable: bool
    used_fallback: bool


SYSTEM_PROMPT = """You are an expert ATS resume strategist for infrastructure, cloud, DevOps,
platform engineering, SRE, and data-adjacent engineering roles. You will be given one target job
posting, the candidate's editable resume summary, editable Technical Expertise category item
lists, and editable bullet text grouped by employer. Your task is ATS-aware, recruiter-readable
tailoring: rewrite every editable field so the resume reads like it was purpose-built for this
specific job while preserving real facts and still sounding natural to a human reviewer.

ATS AND HUMAN READABILITY:
- Optimize for both ATS parsing and recruiter review. Use exact JD keywords where they truthfully
fit, but keep the writing natural, concrete, and easy to scan.
- Do not keyword-stuff, stack buzzwords, or turn bullets into tool lists. Each bullet must read as
one clear achievement or responsibility with a recognizable action, scope, and outcome.
- Keep phrasing concise and plain-spoken. Prefer strong specific verbs and concrete workstreams
over generic claims such as "leveraged skills", "worked on various tasks", or "responsible for".

NON-NEGOTIABLE TAILORING DEPTH:
- Modifying ONLY the Executive Summary is an automatic failure. A valid response MUST materially
tailor the Executive Summary, every Technical Expertise / Skills category, and the Professional
Experience bullets for every role you receive.
- Rewrite the content using strong action verbs, target keywords, and tailored phrasing. Retain
factual accuracy regarding experience, but rephrase bullets and skills so they directly reflect
the requirements of the job description.
- The resume must not read like the Master CV with a new summary. If Technical Expertise or
Professional Experience still read substantially like the originals, your output is invalid.
- Treat the candidate as qualified only through the real roles, projects, tools, and workstreams
supported by the Master CV. Use exact JD terminology aggressively when it fits those verified
workstreams, even if the Master CV used broader or older wording.
- Invent experience to satisfy the posting. Never add employers, titles, degrees, dates,
certifications, business domains.
- When a JD requirement is absent, emphasize the adjacent experience directly.
- Do not attach JD keywords mechanically with phrases like "-aligned", "-backed", or repeated
technology-name modifiers. Reframe the full sentence so keywords describe a real workstream
naturally.
- Do NOT mention total years of experience anywhere in generated resume text. Phrases like
"8 years of experience", "8+ years experience", "over 8 years", or "8-year background" are
forbidden. The locked employment date ranges remain separate and are preserved by code.
- Do NOT treat job-posting qualifiers as candidate resume facts. Terms such as "clearance
required", "must hold clearance", "remote", "onsite", "contract", or eligibility requirements must
not be written into the resume headline, summary, skills, or bullets unless they already describe
the candidate in the master resume facts you were given.

IMMUTABLE FIELD LOCK:
- You are NOT given the candidate's name, contact information, degree/school section, company
names, historical employer role titles, locations, or date ranges.
- Never write, infer, paraphrase, or invent any name, contact detail, employer, historical role title, location,
degree, school, certification date, or date range. Those fields are spliced back by code exactly as
they appeared in the master resume.
- The only title-like field allowed to align to the JD is the top resume headline/profile title
after the candidate name; the code updates that field outside your JSON. Never change historical
employer role titles.

JD ANALYSIS YOU MUST PERFORM:
- Infer the real role category from THIS job title and description. Do not reuse a generic cloud
or DevOps template.
- Extract the target job's core mission in plain terms: what the hire is expected to improve,
operate, automate, secure, migrate, scale, or deliver.
- Extract 10-18 exact JD keywords, ordered by importance. Put the top 3-4 primary technical
requirements first. Use exact named tools, cloud services, methodologies, languages, compliance
terms, and operational practices from the JD when they appear.

EXECUTIVE SUMMARY REQUIREMENTS:
- Write a fresh 2-4 sentence Executive Summary aligned to the target job title, core mission, and
top 3-4 primary technical requirements from the JD. And don't say you've worked for them.
- Make the summary recruiter-readable, not a dense keyword inventory. Use one compact phrase for
the most important matching tools, then explain the practical infrastructure, delivery, reliability,
security, or automation value behind them.
- The first words must be role-specific or mission-specific, for example "Cloud automation
engineer..." or "Platform-focused engineer...". NEVER start with generic resume cliches.
- Strictly forbidden summary starter phrases: "Results-driven", "Result-driven",
"Results-oriented", "Proven track record", "Seasoned professional", "Highly skilled
professional", "Experienced professional", "Dynamic professional", "Dedicated professional", or
minor variations of those phrases.
- Naturally include the strongest JD terms for cloud, DevOps, platform, SRE, automation,
monitoring, security, and scripting workstreams. If the Master CV uses generic wording for a
workstream that the JD names precisely, replace the generic wording with the JD's stronger exact
terminology when it remains factually plausible.

TECHNICAL EXPERTISE REQUIREMENTS:
- Return exactly one tailored item string per category you are given, in the same category order.
Do not return the category label itself.
- Technical Expertise / Skills MUST be reorganized, reordered, and updated to highlight
technologies requested in the Job Description, using the JD's exact terminology when it matches
the candidate's cloud, DevOps, platform, SRE, automation, monitoring, security, or scripting
workstreams.
- You may dynamically swap, reorder, and insert JD-named tools across categories when the tool is
aligned to the target role and belongs in that category. Replace weaker generic terms with stronger
JD terms where appropriate, such as "pipelines" -> "CI/CD pipelines", "monitoring" ->
"observability", "containers" -> "Docker/Kubernetes", or "infrastructure automation" ->
"Terraform/IaC" when those JD terms appear.
- Put JD-matching tools first, then adjacent tools. Remove obvious duplicates across a single
category string. Do not repeat the same keyword, acronym, or obvious alias twice inside one
category.
- Every Technical Expertise category must be materially changed from the original when it is
returned: reorder the tools around the JD, remove low-signal filler when needed, and add
role-aligned JD-matching tools when they fit the category. Do not return the
original category item list unchanged.
- Category contracts are strict:
  Cloud & Infrastructure = cloud providers, cloud services, networking, identity, storage,
  compute, serverless, databases, DNS, TLS, and cloud architecture.
  DevOps & Platforms = ONLY CI/CD, IaC, release automation, configuration management,
  containerization, orchestration, registries, GitOps, and platform delivery tools. Examples:
  Terraform, Helm, Docker, Kubernetes, GitOps, FluxCD, ArgoCD, GitHub Actions, Jenkins, GitLab CI,
  Azure DevOps, SonarQube, Ansible, CloudFormation, ARM/Bicep, Pulumi.
  Monitoring & Security = observability, metrics, logging, tracing, alerting, SLO/error budgets,
  incident response, vulnerability management, IAM/security operations, compliance controls, and
  production reliability practices.
  Languages & Tools = programming/scripting languages, CLIs, SDKs, source control, operating
  systems, administration tools, APIs, frameworks, and developer utilities that are not CI/CD, IaC,
  container, cloud-service, or monitoring/security tools.
- Do NOT place duplicated AWS/Azure/GCP service lists inside DevOps & Platforms. For example,
AWS Lambda, IAM, VPC, API Gateway, KMS, EventBridge, ALB, CloudWatch, Azure VNet, Entra ID, Azure
Monitor, GCP IAM, GKE, and Cloud Storage belong outside DevOps & Platforms unless paired with a
true CI/CD, IaC, container, or GitOps tool.

EXPERIENCE BULLET REQUIREMENTS:
- Rewrite EVERY original bullet substantially. Do not merely swap one or two words.
- Keep bullets readable for a recruiter skimming quickly: lead with the action, include the most
relevant JD terms naturally, and close with the real outcome or operational purpose.
- For every role with 3 or more bullets, at least 3 bullets MUST seamlessly integrate exact
high-value keywords, tools, methodologies, or phrasing from the target Job Description. For roles
with 2 bullets, both bullets must do this; for roles with 1 bullet, that bullet must do this.
- Across each role, spread 4-8 distinct JD keywords across the bullets when the workstreams allow
it. Do not repeat the same JD keyword in adjacent bullets when an equivalent high-value JD term can
be used instead.
- The Professional Experience section must carry the strongest JD keyword density. Do not leave JD
terms only in the Executive Summary or Technical Expertise. If a JD names tools, methodologies,
platform practices, programming languages, SDLC practices, observability, security, automation, or
cloud services that fit a bullet's workstream, rewrite that bullet to include those exact terms.
- Every returned experience bullet must have different wording from the original. Keep the same
truthful fact, but rewrite the action verb, emphasis, keyword placement, and impact framing so it
aligns to the JD.
- Preserve the original achievement, scope, and exact numeric metrics. You may foreground or move
existing metrics, but never invent a new number, percentage, uptime, team size, budget, SLA, date,
or volume.
- Incorporate exact JD technologies, methodologies, and impact keywords wherever truthful. A bullet
should usually include 2-4 JD keywords when the bullet workstream can plausibly support them.
- Do not pad by repeating the same keyword twice in a bullet. Prefer a compact sequence of distinct
JD-matching tools, methodologies, and outcome terms.
- You may bring a JD keyword/tool into a bullet when the original bullet supports the same
workstream or tool family, even if the original used generic wording. Example: a CI/CD bullet can
foreground GitHub Actions, Jenkins, quality gates, rollback automation, release automation, SDLC,
or DevSecOps. An infrastructure bullet can foreground Terraform, IaC, cloud networking, identity,
or automation. A monitoring/reliability bullet can foreground observability, alerting, SLOs,
incident response, or production reliability. Do not claim unrelated domains, migrations,
certifications, compliance regimes, or metrics that are not present.
- Replace generic baseline wording with high-impact JD language: automate, harden, scale, migrate,
orchestrate, monitor, secure, standardize, troubleshoot, optimize, productionize, govern,
instrument, remediate, and deliver.
- Never copy the job posting's future-tense responsibilities into a resume bullet. The bullet must
describe candidate work already done, not what the employer wants someone to do.

HARD OUTPUT RULES:
- Every employer's "experience_bullets" entry MUST contain EXACTLY the same number of bullets as
that employer's original bullet list you were given: same order, one rewritten bullet per original
bullet. Never merge, split, add, drop, or summarize bullets.
- "technical_expertise" MUST contain exactly one string per category you were given, in the same
order. Never add, remove, rename, or merge categories.
- Preserve structural layout exactly: do not add or remove job roles, section headers, category
names, historical employer role titles, company names, locations, dates, or total bullet counts.
Return only the
editable field strings requested by the JSON schema so ReportLab can render the locked layout.
- Use **double asterisks** around 2-4 short scannable keywords/tools/metrics per bullet and per
technical-expertise category. Bold tool names, platforms, metrics, or 2-4 word impact phrases only.
Never bold an entire sentence.
- Do not add meta-commentary, notes, disclaimers, placeholders, or parenthetical explanations
about the rewrite. Resume fields must contain resume text only.
- Do not calculate, infer, or state total tenure/years of experience. Keep JD keyword alignment in
tools, workstreams, responsibilities, and outcomes instead.
- Respond with ONLY a single JSON object, no markdown fences and no extra keys, matching exactly:
{"keywords": ["...", "..."], "summary": "...", "technical_expertise": ["category 1 tool list",
"category 2 tool list"], "experience_bullets": [["bullet 1 for employer 1",
"bullet 2 for employer 1"], ["bullet 1 for employer 2"]]}
- Every string inside "summary", "technical_expertise", and "experience_bullets" must be real
resume content, never a bare schema key, field name, note, or placeholder.
"""


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

_META_PHRASES = (
    "as an ai language model",
    "as an ai, i",
    "i cannot provide",
    "i can't provide",
    "here is the tailored",
    "here's the tailored",
    "note:",
    "[unchanged]",
    "remains unchanged",
    "same as original",
)

_BANNED_SUMMARY_STARTER_RE = re.compile(
    r"^\s*(?:"
    r"results?[-\s]?driven|"
    r"results?[-\s]?oriented|"
    r"proven\s+track\s+record(?:\s+(?:of|in|for|with))?|"
    r"seasoned\s+professional(?:\s+with)?|"
    r"highly\s+skilled(?:\s+professional)?(?:\s+with)?|"
    r"experienced\s+professional(?:\s+with)?|"
    r"dynamic\s+professional(?:\s+with)?|"
    r"dedicated\s+professional(?:\s+with)?"
    r")\b[\s,:;\-]*",
    re.IGNORECASE,
)

_YEARS_VALUE_RE = (
    r"(?:\d{1,2}\+?|\d{1,2}\s*[-\u2013]\s*\d{1,2}|one|two|three|four|five|six|"
    r"seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen)"
)
_YEARS_QUALIFIER_RE = r"(?:(?:over|more than|nearly|about|around|approximately|at least)\s+)?"
_YEARS_EXPERIENCE_RE = re.compile(
    rf"\b(?P<prefix>with|bringing|offering|including)?\s*"
    rf"{_YEARS_QUALIFIER_RE}{_YEARS_VALUE_RE}\s*\+?\s+years?"
    r"(?:['\u2019]\s*)?(?:\s+of)?\s+"
    r"(?P<descriptor>(?:(?:hands-on|professional|relevant|cloud|devops|platform|sre|"
    r"infrastructure|software|engineering|technical)\s+)*)"
    r"experience\b",
    re.IGNORECASE,
)
_YEARS_WORK_RE = re.compile(
    rf"\b{_YEARS_QUALIFIER_RE}{_YEARS_VALUE_RE}\s*\+?\s+years?\s+"
    r"(?=(?:in|with|across|building|supporting|managing|leading|operating|automating)\b)",
    re.IGNORECASE,
)
_YEARS_BACKGROUND_RE = re.compile(
    rf"\b{_YEARS_QUALIFIER_RE}{_YEARS_VALUE_RE}\s*[-\s]+years?\s+"
    r"(?P<descriptor>(?:cloud|devops|platform|sre|infrastructure|software|engineering|technical)\s+)?"
    r"(?:background|track record|career|history|tenure)\b",
    re.IGNORECASE,
)
_JOB_REQUIREMENT_PHRASE_RE = re.compile(
    r"\s*(?:[-\u2013|,/]\s*)?\b(?:"
    r"clearance\s+required|"
    r"required\s+clearance|"
    r"requires?\s+(?:an?\s+)?(?:active\s+)?(?:public\s+trust\s+|secret\s+|top\s+secret\s+|"
    r"ts/sci\s+|security\s+)?clearance|"
    r"must\s+(?:have|hold|obtain|maintain|be\s+eligible\s+for).{0,60}\bclearance|"
    r"eligible\s+for\s+(?:public\s+trust\s+|secret\s+|top\s+secret\s+|ts/sci\s+|security\s+)?clearance"
    r")\b",
    re.IGNORECASE,
)
_JOB_TITLE_PAREN_QUALIFIER_RE = re.compile(
    r"\s*[\(\[][^\)\]]*(?:clearance|remote|hybrid|onsite|on-site|contract|w2|c2c|visa|"
    r"citizen|citizenship)[^\)\]]*[\)\]]",
    re.IGNORECASE,
)
_JOB_TITLE_TRAILING_QUALIFIER_RE = re.compile(
    r"\s*[-\u2013|:/]\s*(?:"
    r".*\bclearance\b.*|"
    r"remote|hybrid|onsite|on-site|contract|contractor|temporary|temp|w2|c2c|"
    r"full[-\s]?time|part[-\s]?time|"
    r"(?:u\.?s\.?\s+)?citizen(?:ship)?\s+required|"
    r"visa\s+sponsorship.*"
    r")$",
    re.IGNORECASE,
)
_UNVERIFIED_STATUS_KEYWORD_RE = re.compile(
    r"\b(?:clearance|polygraph|public\s+trust|ts/sci)\b",
    re.IGNORECASE,
)

# The system prompt requires every bullet to use a leading "-" (and
# explicitly says to convert any other bullet glyph like "•" to it), but
# model responses do not always comply -- and separately, extracted PDFs can
# include U+FFFD (the Unicode replacement character) in place of the bullet
# glyph entirely, a text-extraction artifact since nothing in this
# pipeline itself performs a lossy decode. Both cases are unambiguously "this
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


def _remove_total_years_experience_claims(text: str) -> str:
    def replace_experience(match: re.Match) -> str:
        prefix = (match.group("prefix") or "").strip()
        descriptor = (match.group("descriptor") or "").strip()
        descriptor_words = [
            word
            for word in descriptor.split()
            if word.lower() not in {"hands-on", "professional", "relevant"}
        ]
        descriptor_prefix = " ".join(descriptor_words)
        replacement = (
            f"hands-on {descriptor_prefix} experience"
            if descriptor_prefix
            else "hands-on experience"
        )
        return f"{prefix} {replacement}" if prefix else replacement

    def replace_background(match: re.Match) -> str:
        descriptor = (match.group("descriptor") or "").strip()
        return f"hands-on {descriptor}background" if descriptor else "hands-on background"

    cleaned = _YEARS_EXPERIENCE_RE.sub(replace_experience, text or "")
    cleaned = _YEARS_WORK_RE.sub("hands-on work ", cleaned)
    cleaned = _YEARS_BACKGROUND_RE.sub(replace_background, cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return cleaned.strip()


def _remove_job_requirement_phrases(text: str) -> str:
    cleaned = _JOB_REQUIREMENT_PHRASE_RE.sub("", text or "")
    cleaned = re.sub(r"\s+[-\u2013|,/]\s*([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return cleaned.strip(" -\u2013|,/")


def _sanitize_resume_headline_title(job_title: str) -> str:
    title = _remove_job_requirement_phrases(_clean_field_text(job_title))
    previous = None
    while title and title != previous:
        previous = title
        title = _JOB_TITLE_PAREN_QUALIFIER_RE.sub("", title)
        title = _JOB_TITLE_TRAILING_QUALIFIER_RE.sub("", title)
        title = _remove_job_requirement_phrases(title)
        title = re.sub(r"\s{2,}", " ", title).strip(" -\u2013|:/,")
    return title


def _looks_like_unverified_status_keyword(keyword: str) -> bool:
    return _UNVERIFIED_STATUS_KEYWORD_RE.search(keyword or "") is not None


def _has_total_years_experience_claim(text: str) -> bool:
    return any(
        pattern.search(text or "")
        for pattern in (_YEARS_EXPERIENCE_RE, _YEARS_WORK_RE, _YEARS_BACKGROUND_RE)
    )


def _clean_field_text(text: str) -> str:
    """Strips stray markdown code fences and LLM conversational notes from a
    single returned string field (a bullet, the summary, ...)."""
    lines = []
    for line in (text or "").splitlines():
        line = _normalize_bullet_marker(line)
        stripped = line.strip()
        if not stripped:
            lines.append(line)
            continue
        if stripped.startswith("```"):
            continue
        lowered = stripped.lower()
        if any(phrase in lowered for phrase in _META_PHRASES):
            continue
        line = _INLINE_META_PAREN_RE.sub("", line).rstrip()
        while True:
            stripped_line = _INLINE_META_PAREN_RE.sub("", line)
            if stripped_line == line:
                break
            line = stripped_line
        line = _remove_total_years_experience_claims(line)
        line = _remove_job_requirement_phrases(line)
        lines.append(line)
    return _remove_job_requirement_phrases(
        _remove_total_years_experience_claims("\n".join(lines).replace("```", ""))
    ).strip()


def _capitalize_first_alpha(text: str) -> str:
    for i, char in enumerate(text):
        if char.isalpha():
            return f"{text[:i]}{char.upper()}{text[i + 1:]}"
    return text


def _strip_banned_summary_starter(text: str) -> str:
    """Removes generic resume-cliche openings from an otherwise usable
    summary. This is a deterministic safety net for the prompt's stricter
    instruction: if the model writes "Results-driven Cloud Engineer...",
    the rendered resume starts with "Cloud Engineer..." instead."""
    cleaned = text.strip()
    previous = None
    while cleaned and cleaned != previous:
        previous = cleaned
        cleaned = _BANNED_SUMMARY_STARTER_RE.sub("", cleaned, count=1).lstrip(" ,:;-")
    return _capitalize_first_alpha(cleaned.strip())


def _prepare_summary_text(text: str) -> str:
    return _strip_banned_summary_starter(_clean_field_text(text))


def _render_experience_entries(entries: list[dict], tailored_bullets: list[list[str]]) -> str:
    blocks = []
    for i, entry in enumerate(entries):
        lines = [entry["header_line"]]
        if entry["tagline"]:
            lines.append(entry["tagline"])

        original_bullets = entry["bullets"]
        reworded_raw = tailored_bullets[i] if i < len(tailored_bullets) else []
        # Drop anything that's unambiguously not a real bullet (a leaked
        # schema key like "experience_bullets") before even comparing
        # counts -- a dropped item naturally causes a count mismatch below,
        # which is exactly what should trigger falling back to the originals.
        reworded = [b for b in reworded_raw if not _looks_like_leaked_key(b)]
        # Only trust the model's reworded bullets for this employer if the
        # count matches exactly -- otherwise fall back to the untouched
        # originals rather than risk a merged/dropped/fabricated bullet.
        bullets = reworded if len(reworded) == len(original_bullets) else original_bullets

        for bullet in bullets:
            cleaned = _clean_field_text(bullet).strip()
            if cleaned:
                lines.append(f"- {cleaned}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _strip_stray_category_prefix(candidate: str) -> str:
    """A real observed failure mode: the model prefixes its rewording with a
    category-label-shaped "Label: " fragment -- sometimes its own category's
    label, sometimes (apparent cross-contamination between categories in the
    same generation) a DIFFERENT category's. The items text should never
    start with a label at all, since the real label is always supplied
    separately on reconstruction -- so unwrap one if present rather than
    either keeping a visibly wrong "Label: Label: real content" duplicate or
    discarding an otherwise-good rewording over it."""
    match = _CATEGORY_LINE_RE.match(candidate)
    return match.group(2).strip() if match else candidate


_DEVOPS_CATEGORY_HINTS = ("devops", "platform")
_DEVOPS_ALLOWED_HINTS = (
    "ansible",
    "argo",
    "argocd",
    "argo cd",
    "artifactory",
    "arm/bicep",
    "aws cdk",
    "azure devops",
    "bicep",
    "bitbucket pipelines",
    "buildkite",
    "circleci",
    "ci/cd",
    "cloudformation",
    "cloud development kit",
    "codebuild",
    "codecommit",
    "codedeploy",
    "codepipeline",
    "configuration management",
    "consul",
    "container",
    "container orchestration",
    "cluster autoscaler",
    "docker",
    "docker compose",
    "ecr",
    "flux",
    "fluxcd",
    "git",
    "github actions",
    "gitlab ci",
    "gitops",
    "harbor",
    "helm",
    "hpa",
    "iac",
    "infrastructure as code",
    "ingress",
    "istio",
    "jenkins",
    "keda",
    "kubernetes",
    "kustomize",
    "gradle",
    "maven",
    "nexus",
    "nomad",
    "npm",
    "openshift",
    "orchestration",
    "packer",
    "pipeline",
    "platform",
    "pulumi",
    "rancher",
    "release",
    "release automation",
    "blue/green",
    "canary",
    "service mesh",
    "sonarqube",
    "terragrunt",
    "terraform",
    "vault",
    "zero-downtime",
)
_CLOUD_SERVICE_ONLY_HINTS = (
    "alb",
    "api gateway",
    "app service",
    "aws",
    "aws config",
    "aws organizations",
    "azure",
    "azure monitor",
    "azure sql",
    "bigquery",
    "control tower",
    "cloud build",
    "cloud monitoring",
    "cloud storage",
    "cloudfront",
    "cloud run",
    "cloudwatch",
    "dynamodb",
    "ec2",
    "eks",
    "elb",
    "entra id",
    "eventbridge",
    "fargate",
    "finops",
    "functions",
    "gcp",
    "gke",
    "guardduty",
    "iam",
    "kms",
    "landing zone",
    "lambda",
    "log analytics",
    "nlb",
    "nosql",
    "okta",
    "postgresql",
    "pub/sub",
    "mysql",
    "kafka",
    "rabbitmq",
    "rds",
    "redis",
    "route 53",
    "s3",
    "secrets manager",
    "security hub",
    "servicecatalog",
    "sns",
    "sqs",
    "step functions",
    "storage accounts",
    "systems manager",
    "vnet",
    "vpc",
    "waf",
)
_CLOUD_CATEGORY_FORBIDDEN_HINTS = (
    "ansible",
    "argo",
    "argocd",
    "aws cdk",
    "azure devops",
    "bitbucket pipelines",
    "circleci",
    "ci/cd",
    "cloud development kit",
    "cloudformation",
    "codebuild",
    "codecommit",
    "codedeploy",
    "codepipeline",
    "cluster autoscaler",
    "docker",
    "docker compose",
    "envoy",
    "flux",
    "fluxcd",
    "github actions",
    "gitlab ci",
    "gitops",
    "helm",
    "hpa",
    "ingress",
    "istio",
    "jenkins",
    "keda",
    "kubernetes",
    "kustomize",
    "nomad",
    "openshift",
    "packer",
    "pipeline",
    "policy as code",
    "pulumi",
    "service mesh",
    "sonarqube",
    "terragrunt",
    "terraform",
    "tfsec",
    "trivy",
    "vault",
)
_MONITORING_CATEGORY_FORBIDDEN_HINTS = (
    "argo",
    "argocd",
    "ci/cd",
    "docker",
    "github actions",
    "gitops",
    "helm",
    "jenkins",
    "kubernetes",
    "kustomize",
    "pipeline",
    "pulumi",
    "sonarqube",
    "terraform",
)
_LANGUAGES_CATEGORY_FORBIDDEN_HINTS = (
    "ansible",
    "argo",
    "argocd",
    "aws",
    "azure",
    "azure monitor",
    "ci/cd",
    "cloud monitoring",
    "cloudwatch",
    "docker",
    "gcp",
    "github actions",
    "gitops",
    "grafana",
    "helm",
    "hpa",
    "ingress",
    "istio",
    "jenkins",
    "keda",
    "kubernetes",
    "prometheus",
    "pulumi",
    "service mesh",
    "sonarqube",
    "terraform",
    "trivy",
)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _split_tool_items(items: str) -> list[str]:
    """Splits a comma-delimited tool list without breaking parenthesized
    groups such as "AWS (Lambda, IAM)"."""
    parts: list[str] = []
    start = 0
    depth = 0
    for i, char in enumerate(items):
        if char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            part = items[start:i].strip()
            if part:
                parts.append(part)
            start = i + 1

    last = items[start:].strip()
    if last:
        parts.append(last)
    return parts


def _join_tool_items(parts: list[str]) -> str:
    return ", ".join(part.strip().rstrip(".,;") for part in parts if part.strip())


def _dedupe_tool_items(parts: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for part in parts:
        key = _keyword_canonical_key(part.strip().strip("."))
        if key and key not in seen:
            seen.add(key)
            deduped.append(part.strip())
    return deduped


def _enforce_category_contract(label: str, candidate: str, original_items: str) -> str:
    """Keeps category-specific cleanup deterministic after the model returns.
    The high-risk case is DevOps & Platforms: local models often duplicate
    AWS/Azure/GCP service clusters there after seeing them in a JD, even
    though that line should stay focused on CI/CD, IaC, GitOps, and
    containers."""
    parts = _dedupe_tool_items(_split_tool_items(candidate))
    if not parts:
        return candidate

    lowered_label = label.lower()
    if any(hint in lowered_label for hint in _DEVOPS_CATEGORY_HINTS):
        parts = [
            part for part in parts
            if not (
                _contains_any(part, _CLOUD_SERVICE_ONLY_HINTS)
                and not _contains_any(part, _DEVOPS_ALLOWED_HINTS)
            )
        ]
        if not parts:
            return original_items
    elif "cloud" in lowered_label or "infrastructure" in lowered_label:
        parts = [
            part for part in parts
            if _provider_for_item(part) is not None or not _contains_any(part, _CLOUD_CATEGORY_FORBIDDEN_HINTS)
        ]
    elif "monitor" in lowered_label or "security" in lowered_label:
        parts = [
            part for part in parts
            if not _contains_any(part, _MONITORING_CATEGORY_FORBIDDEN_HINTS)
        ]
    elif "language" in lowered_label or "tools" in lowered_label:
        parts = [
            part for part in parts
            if not _contains_any(part, _LANGUAGES_CATEGORY_FORBIDDEN_HINTS)
        ]

    return _join_tool_items(parts) if parts else original_items


def _is_valid_category_rewording(candidate: str, entry: dict) -> bool:
    """Guards against a real observed failure mode: the model echoing the
    category LABEL back as if it were the items ("Cloud & Infrastructure:
    Cloud & Infrastructure"), silently destroying that category's actual
    tool list. Neither `_looks_like_leaked_key` (candidate isn't a bare
    schema-key token) nor the count check used for bullets (there's no count
    here, it's one string) catches this, so it needs its own check."""
    if _looks_like_leaked_key(candidate):
        return False
    if candidate.strip().lower() == entry["label"].strip().lower():
        return False
    # A genuine rewording of a tool list shouldn't collapse to a fraction of
    # the original's length -- that's a sign of truncation or the model
    # substituting something degenerate rather than actually rephrasing.
    if len(candidate) < 0.4 * len(entry["items"]):
        return False
    return True


def _render_technical_expertise(entries: list[dict], tailored_items: list[str]) -> str:
    lines = []
    for i, entry in enumerate(entries):
        items = entry["items"]
        # Per-category fallback: only trust the model's rewording for THIS
        # category if it was actually returned and passes validation --
        # otherwise keep that one category's original items rather than
        # discarding the whole section over one bad entry.
        if i < len(tailored_items):
            candidate = _strip_stray_category_prefix(_clean_field_text(tailored_items[i]).strip())
            candidate = _enforce_category_contract(entry["label"], candidate, entry["items"])
            if candidate and _is_valid_category_rewording(candidate, entry):
                items = candidate
        lines.append(f"{entry['prefix']} {entry['label']}: {items}")
    return "\n".join(lines)


def _tailor_header_title(header_content: str, job_title: str) -> str:
    target_title = re.sub(r"\s+", " ", _sanitize_resume_headline_title(job_title).strip())
    if not target_title:
        return header_content

    lines = header_content.splitlines()
    if not lines or "|" not in lines[0]:
        return header_content

    name_part, title_part = lines[0].split("|", 1)
    current_title = title_part.strip()
    if not current_title or "@" in current_title:
        return header_content

    lines[0] = f"{name_part.rstrip()} | {target_title.upper()}"
    return "\n".join(lines)


def _reconstruct_tailored_text(
    sections: list[dict],
    payload: _TailoredPayload,
    *,
    target_job_title: str = "",
) -> str:
    experience_idx = next((i for i, s in enumerate(sections) if _is_experience_section(s["name"])), None)
    summary_idx = next((i for i, s in enumerate(sections) if _is_summary_section(s["name"])), None)
    skills_idx = next((i for i, s in enumerate(sections) if _is_skills_section(s["name"])), None)

    experience_entries = None
    if experience_idx is not None:
        experience_entries = _split_experience_entries(sections[experience_idx]["content"])

    skills_entries = None
    if skills_idx is not None:
        skills_entries = _split_technical_expertise(sections[skills_idx]["content"])

    output_blocks = []
    for i, section in enumerate(sections):
        if i == experience_idx and experience_entries is not None:
            content = _render_experience_entries(experience_entries, payload.experience_bullets)
        elif i == skills_idx and skills_entries is not None:
            content = _render_technical_expertise(skills_entries, payload.technical_expertise)
        elif i == summary_idx and payload.summary.strip() and not _looks_like_leaked_key(payload.summary):
            content = _prepare_summary_text(payload.summary)
        else:
            # Passthrough for literally everything else -- Header, Education,
            # Certifications, Additional, and a Summary/Skills/Experience
            # section this heuristic couldn't confidently split or that the
            # model left empty -- reproduced byte-for-byte from the master
            # CV, exactly matching its original template 1:1 except for the
            # top resume headline title when a target job title is provided.
            content = section["content"]

        if section["name"].lower() == "header" and i == 0:
            content = _tailor_header_title(content, target_job_title)
            output_blocks.append(content)
        else:
            output_blocks.append(f"{section['name'].upper()}\n{content}")

    return "\n\n".join(output_blocks).strip()


def _build_prompt(
    sections: list[dict], job_title: str, company_name: str, job_description_text: str
) -> tuple[str, list[dict] | None, list[dict] | None]:
    experience_idx = next((i for i, s in enumerate(sections) if _is_experience_section(s["name"])), None)
    experience_entries: list[dict] | None = None
    employers_block = "(no Professional Experience section could be confidently parsed -- do not \
return any experience_bullets entries)"
    if experience_idx is not None:
        experience_entries = _split_experience_entries(sections[experience_idx]["content"])
        if experience_entries:
            parts = []
            for i, entry in enumerate(experience_entries):
                bullet_lines = "\n".join(f"  - {b}" for b in entry["bullets"])
                parts.append(
                    f"Employer {i + 1} ({len(entry['bullets'])} original bullets; "
                    f"return exactly {len(entry['bullets'])} rewritten bullets):\n{bullet_lines}"
                )
            employers_block = "\n\n".join(parts)

    summary_idx = next((i for i, s in enumerate(sections) if _is_summary_section(s["name"])), None)
    summary_block = sections[summary_idx]["content"] if summary_idx is not None else "(none)"

    skills_idx = next((i for i, s in enumerate(sections) if _is_skills_section(s["name"])), None)
    skills_entries: list[dict] | None = None
    skills_block = "(no Technical Expertise section could be confidently parsed -- do not return \
any technical_expertise entries)"
    if skills_idx is not None:
        skills_entries = _split_technical_expertise(sections[skills_idx]["content"])
        if skills_entries:
            parts = []
            for i, entry in enumerate(skills_entries):
                parts.append(
                    f'Category {i + 1} ("{entry["label"]}" - materially reorder/rewrite this list): '
                    f'{entry["items"]}'
                )
            skills_block = "\n".join(parts)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- TARGET JOB ---\n"
        f"Job Title: {job_title}\n"
        f"Company: {company_name}\n"
        f"Job Description:\n{job_description_text}\n\n"
        f"--- EXISTING SUMMARY (reframe this) ---\n{summary_block}\n\n"
        f"--- EXISTING TECHNICAL EXPERTISE CATEGORIES (reorder/rephrase each category's tool list \
only -- the category name itself, shown here only for context, is fixed and you do not return it) \
---\n{skills_block}\n\n"
        f"--- EXISTING BULLETS PER EMPLOYER (reword each employer's bullets; you are NOT told \
which company/title/dates these belong to, and must not guess or reference one) ---\n"
        f"{employers_block}\n"
    )
    return prompt, experience_entries, skills_entries


def _comparison_key(text: str) -> str:
    normalized = text.replace("**", "").lower()
    normalized = re.sub(r"[^a-z0-9+#/]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _keyword_canonical_key(keyword: str) -> str:
    normalized = (keyword or "").replace("**", "").lower()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[\._]+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9+#/]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    alias_groups = {
        "argocd": {"argo cd", "argocd"},
        "arm/bicep": {"arm bicep", "arm/bicep", "bicep"},
        "aws cdk": {"aws cdk", "cloud development kit"},
        "ci/cd": {"ci cd", "ci/cd", "continuous integration", "continuous delivery"},
        "c#": {"c#", "c sharp"},
        ".net": {".net", "asp.net", "asp net", "dotnet", "net"},
        "fluxcd": {"flux cd", "fluxcd"},
        "golang": {"go", "golang"},
        "iac": {"iac", "infrastructure as code"},
        "node.js": {"express js", "express.js", "javascript/node js", "node js", "node.js"},
        "react": {"react", "react js", "react.js"},
        "rest api": {"api development", "rest api", "rest apis", "restful api", "restful apis"},
        "sql server": {"microsoft sql server", "ms sql", "mssql", "sql server"},
        "entity framework": {"entity framework", "entity framework core"},
        "service catalog": {"service catalog", "servicecatalog"},
        "slo/error budgeting": {
            "error budget",
            "error budgeting",
            "error budgets",
            "slo",
            "slo error budgeting",
            "slo/error budgeting",
            "slos",
        },
        "aws ec2": {"amazon ec2", "ec2"},
        "aws rds": {"amazon rds", "rds"},
        "aws s3": {"amazon s3", "s3"},
        "azure functions": {"azure functions", "functions"},
        "google pub/sub": {"google pub/sub", "pub/sub"},
    }
    for canonical, aliases in alias_groups.items():
        if normalized in aliases:
            return canonical
    return normalized


_ROLE_TITLE_KEYWORD_WORDS = frozenset(
    {
        "administrator",
        "analyst",
        "architect",
        "consultant",
        "developer",
        "director",
        "engineer",
        "lead",
        "manager",
        "principal",
        "senior",
        "specialist",
    }
)


def _looks_like_role_title_keyword(keyword: str) -> bool:
    words = set(_comparison_key(keyword).split())
    return bool(words & _ROLE_TITLE_KEYWORD_WORDS)


def _keyword_list(keywords: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for keyword in keywords:
        cleaned = re.sub(r"\s+", " ", (keyword or "").replace("**", "").strip())
        if _has_total_years_experience_claim(cleaned) or _looks_like_unverified_status_keyword(cleaned):
            continue
        cleaned = _remove_total_years_experience_claims(cleaned)
        cleaned = _remove_job_requirement_phrases(cleaned)
        key = _keyword_canonical_key(cleaned)
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def _content_keyword_list(keywords: list[str]) -> list[str]:
    return [keyword for keyword in _keyword_list(keywords) if not _looks_like_role_title_keyword(keyword)]


_KNOWN_RESUME_KEYWORDS = (
    "Infrastructure as Code",
    "Platform Engineering",
    "Site Reliability Engineering",
    "Production Reliability",
    "Release Automation",
    "Artifact Management",
    "Configuration Management",
    "Vulnerability Management",
    "Incident Management",
    "Incident Response",
    "Root Cause Analysis",
    "Change Management",
    "Service Delivery",
    "Cloud Migration",
    "Landing Zone",
    "Control Tower",
    "AWS Organizations",
    "FinOps",
    "Cost Optimization",
    "Disaster Recovery",
    "High Availability",
    "Load Balancing",
    "Auto Scaling",
    "Blue/Green Deployments",
    "Canary Deployments",
    "Zero-Downtime Deployments",
    "Container Orchestration",
    "Service Mesh",
    "Istio",
    "Envoy",
    "NGINX",
    "Ingress",
    "KEDA",
    "HPA",
    "Cluster Autoscaler",
    "Microservices",
    "Serverless",
    "Observability",
    "Monitoring",
    "Logging",
    "Alerting",
    "Tracing",
    "SLO/Error Budgeting",
    "Error Budgeting",
    "Error Budgets",
    "SLOs",
    "SLIs",
    "SLAs",
    "On-Call",
    "Runbooks",
    "Playbooks",
    "DevSecOps",
    "GitHub Actions",
    "Azure DevOps",
    "GitLab CI",
    "Bitbucket Pipelines",
    "CircleCI",
    "Buildkite",
    "CodePipeline",
    "CodeBuild",
    "CodeDeploy",
    "CodeCommit",
    "SQS",
    "SNS",
    "CloudFormation",
    "Cloud Development Kit",
    "AWS CDK",
    "Terragrunt",
    "Pulumi",
    "ServiceCatalog",
    "Service Catalog",
    "Azure Monitor",
    "Log Analytics",
    "Application Insights",
    "Cloud Monitoring",
    "Cloud Logging",
    "Cloud Storage",
    "JavaScript/Node.js",
    "TypeScript",
    "JavaScript",
    "Node.js",
    "React",
    "React.js",
    "Next.js",
    "Angular",
    "Vue.js",
    "HTML",
    "CSS",
    "Tailwind CSS",
    "Redux",
    "Express.js",
    "FastAPI",
    "Django",
    "Flask",
    "Spring Boot",
    ".NET",
    "C#",
    "ASP.NET",
    "SQL Server",
    "Microsoft SQL Server",
    "Entity Framework",
    "Entity Framework Core",
    "LINQ",
    "Unit Testing",
    "xUnit",
    "NUnit",
    "Jest",
    "Cypress",
    "Selenium",
    "ARM/Bicep",
    "API Gateway",
    "CloudWatch",
    "CloudTrail",
    "CloudFront",
    "Route 53",
    "Security Hub",
    "GuardDuty",
    "AWS Config",
    "Secrets Manager",
    "Systems Manager",
    "Step Functions",
    "DynamoDB",
    "PostgreSQL",
    "MySQL",
    "NoSQL",
    "Redis",
    "Kafka",
    "RabbitMQ",
    "Amazon RDS",
    "RDS",
    "Amazon S3",
    "S3",
    "Amazon EC2",
    "EC2",
    "ECS",
    "Fargate",
    "ECR",
    "ELB",
    "NLB",
    "WAF",
    "Prometheus",
    "SonarQube",
    "OpenTelemetry",
    "Jaeger",
    "Fluent Bit",
    "ELK",
    "Elasticsearch",
    "Logstash",
    "Kibana",
    "Datadog",
    "Splunk",
    "New Relic",
    "PagerDuty",
    "Opsgenie",
    "SIEM",
    "Cloud Security",
    "Network Security",
    "Policy as Code",
    "SAST",
    "DAST",
    "SBOM",
    "OPA",
    "Open Policy Agent",
    "Sentinel",
    "Checkov",
    "tfsec",
    "Trivy",
    "Prisma Cloud",
    "Wiz",
    "SOC 2",
    "SOX",
    "ISO 27001",
    "HIPAA",
    "PCI DSS",
    "FedRAMP",
    "NIST",
    "CIS Benchmarks",
    "RBAC",
    "SSO",
    "MFA",
    "OAuth",
    "OIDC",
    "SAML",
    "Terraform",
    "Kubernetes",
    "Docker",
    "OpenShift",
    "Rancher",
    "Kustomize",
    "Docker Compose",
    "Jenkins",
    "Maven",
    "Gradle",
    "npm",
    "GitOps",
    "FluxCD",
    "ArgoCD",
    "Argo CD",
    "Ansible",
    "Chef",
    "Puppet",
    "SaltStack",
    "Packer",
    "HashiCorp Vault",
    "Vault",
    "Consul",
    "Nomad",
    "Python",
    "Bash",
    "PowerShell",
    "Shell Scripting",
    "Go",
    "Golang",
    "Java",
    "SQL",
    "MongoDB",
    "YAML",
    "JSON",
    "REST APIs",
    "RESTful APIs",
    "API Development",
    "GraphQL",
    "CLI",
    "SDKs",
    "Git",
    "ServiceNow",
    "Jira",
    "Confluence",
    "Okta",
    "Linux",
    "Windows Server",
    "Agile",
    "Scrum",
    "Kanban",
    "ITIL",
    "SDLC",
    "CI/CD",
    "IaC",
    "Helm",
    "Grafana",
    "Lambda",
    "EventBridge",
    "Entra ID",
    "Azure SQL",
    "AKS",
    "ACR",
    "Azure Key Vault",
    "App Service",
    "Storage Accounts",
    "Functions",
    "Cloud Run",
    "Cloud Build",
    "Pub/Sub",
    "BigQuery",
    "DNS",
    "SSL/TLS",
    "EKS",
    "GKE",
    "ACR/ECR",
    "Nexus",
    "Artifactory",
    "Harbor",
    "AWS",
    "Azure",
    "GCP",
    "IAM",
    "VPC",
    "ALB",
    "KMS",
)


def _keyword_position(text_lower: str, keyword: str) -> int | None:
    keyword_lower = keyword.strip().lower()
    if not keyword_lower:
        return None
    if re.fullmatch(r"[\w\s\-]+", keyword_lower):
        match = re.search(rf"\b{re.escape(keyword_lower)}\b", text_lower)
        return match.start() if match else None
    index = text_lower.find(keyword_lower)
    return index if index >= 0 else None


def _keywords_from_text(text: str) -> list[str]:
    text_lower = text.lower()
    matches = []
    for order, keyword in enumerate(_KNOWN_RESUME_KEYWORDS):
        position = _keyword_position(text_lower, keyword)
        if position is not None:
            matches.append((position, order, keyword))
    return _keyword_list([keyword for _, _, keyword in sorted(matches)])


def _supported_target_keywords(target_keywords: list[str], source_text: str) -> list[str]:
    source_lower = source_text.replace("**", "").lower()
    return [keyword for keyword in _keyword_list(target_keywords) if _keyword_present(keyword, source_lower)]


def _matched_keywords(text: str, keywords: list[str]) -> list[str]:
    text_lower = text.replace("**", "").lower()
    matched: list[str] = []
    seen: set[str] = set()
    for keyword in _keyword_list(keywords):
        key = _keyword_canonical_key(keyword)
        if key and key not in seen and _keyword_present(keyword, text_lower):
            seen.add(key)
            matched.append(keyword)
    return matched


def _keyword_match_count(text: str, keywords: list[str]) -> int:
    return len(_matched_keywords(text, keywords))


def _keyword_occurrence_count(text: str, keyword: str) -> int:
    text_lower = text.replace("**", "").lower()
    keyword_lower = keyword.strip().lower()
    if not keyword_lower:
        return 0
    if re.fullmatch(r"[\w\s\-]+", keyword_lower):
        return len(re.findall(rf"\b{re.escape(keyword_lower)}\b", text_lower))
    return text_lower.count(keyword_lower)


def _repeated_long_keywords(text: str, keywords: list[str]) -> list[str]:
    repeated: list[str] = []
    seen_counts: dict[str, int] = {}
    for keyword in _keyword_list(keywords):
        if len(keyword.replace("/", "").replace("+", "").replace("#", "").strip()) <= 3:
            continue
        key = _keyword_canonical_key(keyword)
        count = _keyword_occurrence_count(text, keyword)
        if not count:
            continue
        seen_counts[key] = seen_counts.get(key, 0) + count
        if seen_counts[key] > 1 and key not in {_keyword_canonical_key(item) for item in repeated}:
            repeated.append(keyword)
    return repeated


def _keyword_matches_allowed_terms(keyword: str, allowed_terms: tuple[str, ...]) -> bool:
    keyword_key = _comparison_key(keyword)
    keyword_canonical = _keyword_canonical_key(keyword)
    for term in allowed_terms:
        term_key = _comparison_key(term)
        term_canonical = _keyword_canonical_key(term)
        if not term_key:
            continue
        if keyword_key == term_key or keyword_canonical == term_canonical:
            return True
        if len(term_key) <= 3:
            if re.search(rf"\b{re.escape(term_key)}\b", keyword_key):
                return True
        elif re.search(rf"\b{re.escape(term_key)}\b", keyword_key):
            return True
    return False


def _bullet_keyword_candidates(original: str, keywords: list[str]) -> list[str]:
    """Returns target JD keywords that can be truthfully worked into a bullet's
    existing workstream. This keeps deterministic repair from forcing a cloud
    tool into an unrelated sentence just to satisfy keyword density."""
    keywords = _content_keyword_list(keywords)
    lowered = original.lower()
    candidates = _supported_target_keywords(keywords, original)

    def add_when(triggers: tuple[str, ...], allowed_terms: tuple[str, ...]) -> None:
        if any(trigger in lowered for trigger in triggers):
            candidates.extend(
                keyword
                for keyword in keywords
                if _keyword_matches_allowed_terms(keyword, allowed_terms)
            )

    add_when(
        ("pipeline", "deployment", "deploy", "release", "rollback", "quality gate", "sonarqube"),
        (
            "Bitbucket Pipelines",
            "Blue/Green Deployments",
            "Canary Deployments",
            "CI/CD",
            "CodeBuild",
            "CodeDeploy",
            "CodePipeline",
            "GitOps",
            "ArgoCD",
            "Argo CD",
            "FluxCD",
            "GitHub Actions",
            "GitLab CI",
            "Jenkins",
            "Helm",
            "Kustomize",
            "Maven",
            "Gradle",
            "npm",
            "JavaScript/Node.js",
            "JavaScript",
            "Node.js",
            "TypeScript",
            "React",
            "React.js",
            "Next.js",
            "Angular",
            "Vue.js",
            "REST APIs",
            "RESTful APIs",
            "API Development",
            "GraphQL",
            "Microservices",
            "SAST",
            "DAST",
            "SBOM",
            "Azure DevOps",
            "SonarQube",
            "Release Automation",
            "DevSecOps",
            "SDLC",
            "Zero-Downtime Deployments",
        ),
    )
    add_when(
        ("infrastructure", "provision", "terraform", "ansible", "automated", "automation", "codified"),
        (
            "AWS CDK",
            "Terraform",
            "Terragrunt",
            "Infrastructure as Code",
            "IaC",
            "Ansible",
            "CloudFormation",
            "ARM/Bicep",
            "Pulumi",
            "Packer",
            "HashiCorp Vault",
            "Vault",
            "Consul",
            "Nomad",
            "Configuration Management",
            "Python",
            "Bash",
            "PowerShell",
            "Shell Scripting",
        ),
    )
    add_when(
        ("container", "docker", "kubernetes", "workload", "ingress", "readiness", "orchestrat"),
        (
            "Docker",
            "Kubernetes",
            "Container Orchestration",
            "Service Mesh",
            "Istio",
            "Envoy",
            "NGINX",
            "Ingress",
            "KEDA",
            "HPA",
            "Cluster Autoscaler",
            "EKS",
            "GKE",
            "AKS",
            "ECS",
            "Fargate",
            "OpenShift",
            "Rancher",
            "Docker Compose",
            "Kustomize",
            "Helm",
            "ACR/ECR",
            "ECR",
            "Nexus",
            "Artifactory",
            "Harbor",
            "Orchestration",
        ),
    )
    add_when(
        (
            "cloud",
            "aws",
            "azure",
            "gcp",
            "architecture",
            "architect",
            "serverless",
            "network",
            "migration",
            "availability",
            "scale",
            "scalable",
            "cost",
            "govern",
            "identity",
        ),
        (
            "AWS",
            "Azure",
            "GCP",
            "EKS",
            "GKE",
            "AKS",
            "ECS",
            "Fargate",
            "Lambda",
            "EventBridge",
            "API Gateway",
            "KMS",
            "IAM",
            "VPC",
            "ALB",
            "Entra ID",
            "Azure SQL",
            "Functions",
            "Cloud Storage",
            "Cloud Run",
            "App Service",
            "Storage Accounts",
            "S3",
            "RDS",
            "DynamoDB",
            "EC2",
            "CloudFront",
            "Route 53",
            "WAF",
            "High Availability",
            "Load Balancing",
            "Auto Scaling",
            "Landing Zone",
            "Control Tower",
            "AWS Organizations",
            "FinOps",
            "Cost Optimization",
            "Disaster Recovery",
            "Cloud Migration",
            "Serverless",
            "SQS",
            "SNS",
            "Kafka",
            "RabbitMQ",
            "Redis",
            "PostgreSQL",
            "MySQL",
            "NoSQL",
            "DNS",
            "SSL/TLS",
        ),
    )
    add_when(
        (
            "monitor",
            "observability",
            "logging",
            "alert",
            "incident",
            "slo",
            "error budget",
            "uptime",
            "mttr",
            "security",
            "secure",
            "compliance",
            "vulnerab",
            "audit",
            "remed",
            "reliability",
        ),
        (
            "Observability",
            "Monitoring",
            "Logging",
            "Alerting",
            "Tracing",
            "CloudWatch",
            "CloudTrail",
            "Prometheus",
            "Grafana",
            "Datadog",
            "Splunk",
            "New Relic",
            "OpenTelemetry",
            "Jaeger",
            "ELK",
            "Elasticsearch",
            "Logstash",
            "Kibana",
            "Azure Monitor",
            "Log Analytics",
            "Application Insights",
            "Cloud Monitoring",
            "Incident Response",
            "Incident Management",
            "Root Cause Analysis",
            "SLO/Error Budgeting",
            "Error Budgeting",
            "Error Budgets",
            "SLOs",
            "SLIs",
            "SLAs",
            "On-Call",
            "Runbooks",
            "Playbooks",
            "Production Reliability",
            "Vulnerability Management",
            "Cloud Security",
            "Network Security",
            "Policy as Code",
            "OPA",
            "Open Policy Agent",
            "Sentinel",
            "Checkov",
            "tfsec",
            "Trivy",
            "Prisma Cloud",
            "Wiz",
            "SOC 2",
            "SOX",
            "ISO 27001",
            "NIST",
            "CIS Benchmarks",
            "RBAC",
            "SSO",
            "MFA",
            "Okta",
            "OAuth",
            "OIDC",
            "SAML",
            "SIEM",
        ),
    )
    add_when(
        ("artifact", "registry", "registries", "acr", "ecr", "nexus"),
        ("ACR/ECR", "ECR", "Nexus", "Artifactory", "Harbor", "Artifact Management", "Release Automation"),
    )
    add_when(
        (
            "application",
            "applications",
            "api",
            "apis",
            "service",
            "services",
            "microservice",
            "microservices",
            "software",
            "developer",
            "development",
            "sdlc",
            "enterprise",
        ),
        (
            "JavaScript/Node.js",
            "JavaScript",
            "Node.js",
            "TypeScript",
            "React",
            "React.js",
            "Next.js",
            "Angular",
            "Vue.js",
            "HTML",
            "CSS",
            "Tailwind CSS",
            "Redux",
            "Express.js",
            "FastAPI",
            "Django",
            "Flask",
            "Spring Boot",
            ".NET",
            "C#",
            "ASP.NET",
            "SQL Server",
            "Microsoft SQL Server",
            "Entity Framework",
            "Entity Framework Core",
            "LINQ",
            "Java",
            "REST APIs",
            "RESTful APIs",
            "API Development",
            "GraphQL",
            "SQL",
            "PostgreSQL",
            "MySQL",
            "MongoDB",
            "NoSQL",
            "Microservices",
            "SDLC",
            "Agile",
            "Scrum",
            "Kanban",
            "Unit Testing",
            "xUnit",
            "NUnit",
            "Jest",
            "Cypress",
            "Selenium",
        ),
    )
    ordered = _keyword_list(candidates)
    if any(
        trigger in lowered
        for trigger in ("container", "docker", "kubernetes", "workload", "ingress", "readiness")
    ):
        ordered = [
            keyword
            for _, keyword in sorted(
                enumerate(ordered),
                key=lambda pair: (
                    0 if _provider_for_item(pair[1]) is not None else 1,
                    1 if _is_database_keyword(pair[1]) else 0,
                    pair[0],
                ),
            )
        ]
    return ordered


def _required_keyword_bullet_count(original_bullets: list[str], target_keywords: list[str]) -> int:
    capable_bullets = sum(
        1 for original in original_bullets if _bullet_keyword_candidates(original, target_keywords)
    )
    if not original_bullets or not target_keywords or not capable_bullets:
        return 0
    return min(3, len(original_bullets), capable_bullets)


def _required_role_keyword_count(original_bullets: list[str], target_keywords: list[str]) -> int:
    role_candidates: list[str] = []
    for original in original_bullets:
        role_candidates.extend(_bullet_keyword_candidates(original, target_keywords))

    distinct_candidates = len(_keyword_list(role_candidates))
    if not original_bullets or not distinct_candidates:
        return 0

    return min(8, distinct_candidates, max(4, len(original_bullets) * 2))


def _experience_keyword_candidates(
    experience_entries: list[dict],
    target_keywords: list[str],
) -> list[str]:
    return _keyword_list([
        keyword
        for entry in experience_entries
        for original in entry["bullets"]
        for keyword in _bullet_keyword_candidates(original, target_keywords)
    ])


def _required_experience_keyword_count(
    experience_entries: list[dict],
    target_keywords: list[str],
) -> int:
    candidates = _experience_keyword_candidates(experience_entries, target_keywords)
    total_bullets = sum(len(entry["bullets"]) for entry in experience_entries)
    if not candidates or not total_bullets:
        return 0
    return min(16, len(candidates), max(6, total_bullets * 2))


def _editable_keyword_source(
    sections: list[dict],
    experience_entries: list[dict] | None,
    skills_entries: list[dict] | None,
) -> str:
    parts: list[str] = []
    for section in sections:
        if _is_summary_section(section["name"]):
            parts.append(section["content"])
    if skills_entries is not None:
        parts.extend(entry["items"] for entry in skills_entries)
    if experience_entries is not None:
        for entry in experience_entries:
            parts.extend(entry["bullets"])
    return "\n".join(parts)


def _keyword_rank(text: str, keywords: list[str]) -> int | None:
    text_lower = text.replace("**", "").lower()
    matches = [
        i for i, keyword in enumerate(_keyword_list(keywords))
        if _keyword_present(keyword, text_lower)
    ]
    return min(matches) if matches else None


def _bold_keywords(text: str, keywords: list[str], limit: int = 4) -> str:
    result = text
    applied = 0
    ordered_keywords = sorted(_keyword_list(keywords), key=lambda item: len(item), reverse=True)
    for keyword in ordered_keywords:
        if applied >= limit:
            break
        if re.fullmatch(r"[\w\s\-]+", keyword.strip()):
            pattern = re.compile(rf"(?<!\*)\b({re.escape(keyword)})\b(?!\*)", re.IGNORECASE)
        else:
            pattern = re.compile(rf"(?<!\*)({re.escape(keyword)})(?!\*)", re.IGNORECASE)
        result, count = pattern.subn(r"**\1**", result, count=1)
        if count:
            applied += 1
    return result


def _keyword_series_items(series: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", (series or "").replace("**", "").strip(" ,;"))
    if not normalized:
        return []
    normalized = re.sub(r",?\s+and\s+", ",", normalized, flags=re.IGNORECASE)
    return [item.strip(" ,;") for item in normalized.split(",") if item.strip(" ,;")]


def _format_keyword_series(items: list[str]) -> str:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = re.sub(r"\s+", " ", (item or "").replace("**", "").strip(" ,;"))
        if not cleaned:
            continue
        key = _keyword_canonical_key(cleaned)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    if len(deduped) <= 1:
        return deduped[0] if deduped else ""
    if len(deduped) == 2:
        return f"{deduped[0]} and {deduped[1]}"
    return f"{', '.join(deduped[:-1])}, and {deduped[-1]}"


def _add_keyword_to_series(series: str, keyword: str) -> str:
    return _format_keyword_series(_keyword_series_items(series) + [keyword])


def _normalize_stacked_keyword_phrases(text: str) -> str:
    def collapse_application_delivery(match: re.Match) -> str:
        raw_items = re.split(r"\s+and\s+", match.group(1))
        items = [item.strip() for item in raw_items if item.strip()]
        if len(items) < 3:
            return match.group(0)
        return f"across {_format_keyword_series(items)} application delivery"

    keyword_token = r"(?:\*\*)?[A-Za-z0-9+#./]+(?:\s+[A-Za-z0-9+#./]+){0,3}(?:\*\*)?"

    def collapse_aligned_stack(match: re.Match) -> str:
        items = re.findall(rf"({keyword_token})-aligned", match.group(1))
        series = _format_keyword_series(items)
        if not series:
            return match.group(0)
        return f"{match.group(2)} across {series}"

    text = re.sub(
        r"(\*\*[^*]+\*\*)\s+(\*\*[^*]+\*\*)\s+(\*\*[^*]+\*\*)\s+(coverage|delivery practices)",
        r"\1, \2, and \3 \4",
        text,
    )
    text = re.sub(
        r"(\*\*[^*]+\*\*)\s+(\*\*[^*]+\*\*)\s+(coverage|delivery practices)",
        r"\1 and \2 \3",
        text,
    )
    text = re.sub(
        r"\bacross\s+((?:\*\*[^*]+\*\*|[A-Za-z0-9+#./-]+)(?:\s+and\s+(?:\*\*[^*]+\*\*|[A-Za-z0-9+#./-]+)){2,})\s+application delivery",
        collapse_application_delivery,
        text,
    )
    text = re.sub(
        rf"\b((?:{keyword_token}-aligned\s+){{2,}})(services?|applications?|microservices?|workloads?)\b",
        collapse_aligned_stack,
        text,
    )
    text = re.sub(
        rf"\b({keyword_token})-aligned\s+(service readiness)\b",
        lambda match: f"{match.group(2)} for {_format_keyword_series([match.group(1)])}",
        text,
    )
    text = re.sub(
        rf"\bfor\s+({keyword_token})-aligned\s+infrastructure outcomes\b",
        lambda match: f"across {_format_keyword_series([match.group(1)])} infrastructure outcomes",
        text,
    )
    text = re.sub(r"\*\*Docker\*\*\s+\*\*Kubernetes\*\*", r"**Docker/Kubernetes**", text)
    text = re.sub(r"\*\*Docker\*\*/\*\*Kubernetes\*\*", r"**Docker/Kubernetes**", text)
    text = re.sub(
        r"\b(?:\*\*)?React(?:\*\*)?-aligned\s+(?:\*\*)?TypeScript(?:\*\*)?-aligned\s+services\b",
        r"**React/TypeScript**-aligned services",
        text,
    )
    return text


_CLOUD_PROVIDER_ITEMS = {
    "aws": ("aws",),
    "azure": ("azure",),
    "gcp": ("gcp", "google cloud"),
}


def _provider_matches(keywords: list[str]) -> set[str]:
    lowered_keywords = " ".join(_keyword_list(keywords)).lower()
    if any(term in lowered_keywords for term in ("multi-cloud", "multi cloud", "multicloud")):
        return set(_CLOUD_PROVIDER_ITEMS)
    matched = set()
    for provider, aliases in _CLOUD_PROVIDER_ITEMS.items():
        if any(alias in lowered_keywords for alias in aliases):
            matched.add(provider)
    return matched


def _provider_for_item(item: str) -> str | None:
    lowered = item.lower()
    for provider, aliases in _CLOUD_PROVIDER_ITEMS.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", lowered) for alias in aliases):
            return provider
    return None


_CLOUD_CATEGORY_ALLOWED_HINTS = (
    "api gateway",
    "app service",
    "auto scaling",
    "aws organizations",
    "bigquery",
    "control tower",
    "cost optimization",
    "cloud",
    "cloud migration",
    "cloud run",
    "cloud storage",
    "cloudfront",
    "database",
    "disaster recovery",
    "dns",
    "dynamodb",
    "ec2",
    "ecs",
    "eks",
    "elb",
    "entra id",
    "eventbridge",
    "fargate",
    "finops",
    "functions",
    "gke",
    "guardduty",
    "high availability",
    "iam",
    "identity",
    "kafka",
    "kms",
    "landing zone",
    "lambda",
    "load balancing",
    "mongodb",
    "network",
    "pub/sub",
    "rabbitmq",
    "rds",
    "redis",
    "route 53",
    "s3",
    "secrets manager",
    "security hub",
    "serverless",
    "sns",
    "sqs",
    "ssl/tls",
    "sql server",
    "step functions",
    "storage",
    "vnet",
    "vpc",
    "waf",
)
_MONITORING_SECURITY_CATEGORY_ALLOWED_HINTS = (
    "alert",
    "application insights",
    "audit",
    "azure monitor",
    "cis benchmarks",
    "checkov",
    "cloud security",
    "cloud logging",
    "cloud monitoring",
    "cloudtrail",
    "cloudwatch",
    "compliance",
    "datadog",
    "dast",
    "elastic",
    "error budget",
    "fedramp",
    "fluent bit",
    "grafana",
    "hashicorp vault",
    "hipaa",
    "incident",
    "iso 27001",
    "jaeger",
    "kibana",
    "logging",
    "log analytics",
    "logstash",
    "mfa",
    "monitor",
    "new relic",
    "network security",
    "nist",
    "observability",
    "opa",
    "okta",
    "open policy agent",
    "oauth",
    "oidc",
    "on-call",
    "opsgenie",
    "pagerduty",
    "pci dss",
    "playbook",
    "policy as code",
    "prisma cloud",
    "prometheus",
    "rbac",
    "root cause",
    "runbook",
    "saml",
    "sast",
    "sbom",
    "security",
    "sentinel",
    "siem",
    "sla",
    "sli",
    "slo",
    "sox",
    "soc 2",
    "splunk",
    "sso",
    "tfsec",
    "tracing",
    "trivy",
    "vault",
    "vulnerability",
    "wiz",
)
_LANGUAGE_TOOLS_CATEGORY_ALLOWED_HINTS = (
    "agile",
    "api",
    "api development",
    "angular",
    "bash",
    "c#",
    "cli",
    "confluence",
    "css",
    "cypress",
    "django",
    "dotnet",
    "express",
    "fastapi",
    "flask",
    "go",
    "golang",
    "gradle",
    "graphql",
    "html",
    "itil",
    "java",
    "javascript",
    "jest",
    "jira",
    "json",
    "kanban",
    "linq",
    "linux",
    "maven",
    "node.js",
    "next.js",
    "nunit",
    "npm",
    "powershell",
    "python",
    "react",
    "rest",
    "redux",
    "scrum",
    "sdk",
    "selenium",
    "sdlc",
    "servicenow",
    "shell",
    "sql",
    "sql server",
    "spring",
    "tailwind",
    "typescript",
    "unit testing",
    "vue",
    "windows",
    "xunit",
    "yaml",
)


def _keyword_fits_category(label: str, keyword: str) -> bool:
    lowered_label = label.lower()
    lowered_keyword = keyword.lower()
    if any(hint in lowered_label for hint in _DEVOPS_CATEGORY_HINTS):
        return _keyword_matches_allowed_terms(keyword, _DEVOPS_ALLOWED_HINTS)
    if "cloud" in lowered_label or "infrastructure" in lowered_label:
        return (
            _provider_for_item(keyword) is not None
            or _keyword_matches_allowed_terms(keyword, _CLOUD_CATEGORY_ALLOWED_HINTS)
        ) and not _contains_any(lowered_keyword, _CLOUD_CATEGORY_FORBIDDEN_HINTS)
    if "monitor" in lowered_label or "security" in lowered_label:
        return _keyword_matches_allowed_terms(keyword, _MONITORING_SECURITY_CATEGORY_ALLOWED_HINTS)
    if "language" in lowered_label or "tools" in lowered_label:
        return (
            _keyword_matches_allowed_terms(keyword, _LANGUAGE_TOOLS_CATEGORY_ALLOWED_HINTS)
            and not _contains_any(lowered_keyword, _LANGUAGES_CATEGORY_FORBIDDEN_HINTS)
        )
    return False


def _category_keyword_insertions(label: str, keywords: list[str], existing_items: str) -> list[str]:
    existing_lower = existing_items.replace("**", "").lower()
    insertions = [
        keyword for keyword in _content_keyword_list(keywords)
        if _keyword_fits_category(label, keyword) and not _keyword_present(keyword, existing_lower)
    ]
    return insertions[:10]


def _rewrite_technical_category(entry: dict, raw_candidate: str, keywords: list[str]) -> str:
    candidate = _strip_stray_category_prefix(_clean_field_text(raw_candidate).strip())
    candidate = _enforce_category_contract(entry["label"], candidate or entry["items"], entry["items"])

    parts = _dedupe_tool_items(_split_tool_items(candidate))
    if not parts:
        parts = _dedupe_tool_items(_split_tool_items(entry["items"]))
    if not parts:
        return entry["items"]

    parts = _dedupe_tool_items(
        _category_keyword_insertions(entry["label"], keywords, _join_tool_items(parts)) + parts
    )
    contracted = _enforce_category_contract(entry["label"], _join_tool_items(parts), entry["items"])
    parts = _dedupe_tool_items(_split_tool_items(contracted))

    label_lower = entry["label"].lower()
    matched_providers = _provider_matches(keywords)
    if "cloud" in label_lower and matched_providers:
        filtered = [
            part for part in parts
            if (provider := _provider_for_item(part)) is None or provider in matched_providers
        ]
        if filtered:
            parts = filtered

    ranked_parts = sorted(
        enumerate(parts),
        key=lambda pair: (
            _keyword_rank(pair[1], keywords) is None,
            _keyword_rank(pair[1], keywords) if _keyword_rank(pair[1], keywords) is not None else 10_000,
            pair[0],
        ),
    )
    rewritten_parts = [_bold_keywords(part, keywords, limit=2) for _, part in ranked_parts]
    rewritten = _join_tool_items(rewritten_parts)

    if _comparison_key(rewritten) == _comparison_key(entry["items"]) and len(rewritten_parts) > 1:
        rewritten_parts = rewritten_parts[1:] + rewritten_parts[:1]
        rewritten = _join_tool_items(rewritten_parts)

    return rewritten or entry["items"]


def _keyword_modifier(keyword: str) -> str:
    lowered = keyword.lower()
    if lowered in {"infrastructure as code", "iac"}:
        return "IaC-driven"
    if lowered == "ci/cd":
        return "CI/CD"
    if "/" in keyword or " " in keyword:
        return f"{keyword}-enabled"
    return f"{keyword}-backed"


def _is_database_keyword(keyword: str) -> bool:
    return _keyword_canonical_key(keyword) in {
        "sql",
        "sql server",
        "postgresql",
        "mysql",
        "mongodb",
        "nosql",
        "dynamodb",
        "rds",
        "azure sql",
    }


def _inject_keyword_into_bullet(text: str, keyword: str) -> str:
    keyword = keyword.strip()
    if not keyword or _keyword_match_count(text, [keyword]):
        return text

    lowered_keyword = keyword.lower()
    lowered_text = text.lower()

    def append_phrase(phrase: str) -> str:
        stripped = text.rstrip()
        if phrase.endswith(" coverage"):
            keyword_phrase = phrase.removesuffix(" coverage")
            coverage_match = re.search(r"\s+with\s+(.+?)\s+coverage\.?$", stripped, re.IGNORECASE)
            if coverage_match:
                prefix = stripped[: coverage_match.start()]
                existing = coverage_match.group(1).rstrip(", ")
                return f"{prefix} with {_add_keyword_to_series(existing, keyword_phrase)} coverage."
        if phrase.endswith(" delivery practices"):
            keyword_phrase = phrase.removesuffix(" delivery practices").removeprefix("through ")
            delivery_match = re.search(r"\s+through\s+(.+?)\s+delivery practices\.?$", stripped, re.IGNORECASE)
            if delivery_match:
                prefix = stripped[: delivery_match.start()]
                existing = delivery_match.group(1).rstrip(", ")
                return f"{prefix} through {_add_keyword_to_series(existing, keyword_phrase)} delivery practices."
        if phrase.endswith(" automation"):
            keyword_phrase = phrase.removesuffix(" automation").removeprefix("using ")
            automation_match = re.search(r"\s+using\s+(.+?)\s+automation\.?$", stripped, re.IGNORECASE)
            if automation_match:
                prefix = stripped[: automation_match.start()]
                existing = automation_match.group(1).rstrip(", ")
                return f"{prefix} using {_add_keyword_to_series(existing, keyword_phrase)} automation."
        if phrase.endswith(" application delivery"):
            keyword_phrase = phrase.removesuffix(" application delivery").removeprefix("across ")
            keyword_phrase = keyword_phrase.removeprefix("for ")
            app_match = re.search(r"\s+(across|for)\s+(.+?)\s+application delivery\.?$", stripped, re.IGNORECASE)
            if app_match:
                prefix = stripped[: app_match.start()]
                preposition = app_match.group(1).lower()
                existing = app_match.group(2).rstrip(", ")
                return f"{prefix} {preposition} {_add_keyword_to_series(existing, keyword_phrase)} application delivery."
        if phrase.endswith(" application services"):
            keyword_phrase = phrase.removesuffix(" application services").removeprefix("for ")
            service_match = re.search(r"\s+for\s+(.+?)\s+application services\.?$", stripped, re.IGNORECASE)
            if service_match:
                prefix = stripped[: service_match.start()]
                existing = service_match.group(1).rstrip(", ")
                return f"{prefix} for {_add_keyword_to_series(existing, keyword_phrase)} application services."
        if phrase.endswith(" infrastructure"):
            keyword_phrase = phrase.removesuffix(" infrastructure").removeprefix("across ")
            infrastructure_match = re.search(r"\s+across\s+(.+?)\s+infrastructure\.?$", stripped, re.IGNORECASE)
            if infrastructure_match:
                prefix = stripped[: infrastructure_match.start()]
                existing = infrastructure_match.group(1).rstrip(", ")
                return f"{prefix} across {_add_keyword_to_series(existing, keyword_phrase)} infrastructure."
        if stripped.endswith("."):
            return f"{stripped[:-1]} {phrase}."
        return f"{stripped} {phrase}"

    if lowered_keyword != "ci/cd" and re.search(r"\bci/cd\s+pipelines?\s+with\b", text, re.IGNORECASE):
        if _keyword_matches_allowed_terms(keyword, _DEVOPS_ALLOWED_HINTS):
            candidate = re.sub(
                r"\b(CI/CD\s+pipelines?)\b",
                lambda match: f"{match.group(1)} and {keyword} workflows",
                text,
                count=1,
                flags=re.IGNORECASE,
            )
            if candidate != text and _keyword_match_count(candidate, [keyword]):
                return candidate

    if lowered_keyword != "ci/cd" and re.search(r"\bci/cd\s+pipelines?\b", text, re.IGNORECASE):
        if _keyword_matches_allowed_terms(keyword, _DEVOPS_ALLOWED_HINTS):
            candidate = re.sub(
                r"\b(CI/CD\s+pipelines?)\b",
                lambda match: f"{match.group(1)} with {keyword}",
                text,
                count=1,
                flags=re.IGNORECASE,
            )
            if candidate != text and _keyword_match_count(candidate, [keyword]):
                return candidate

    devops_keyword = _keyword_matches_allowed_terms(keyword, _DEVOPS_ALLOWED_HINTS)
    container_keyword = _keyword_matches_allowed_terms(
        keyword,
        (
            "Docker",
            "Kubernetes",
            "Container Orchestration",
            "Service Mesh",
            "Istio",
            "Envoy",
            "NGINX",
            "Ingress",
            "KEDA",
            "HPA",
            "Cluster Autoscaler",
            "EKS",
            "GKE",
            "AKS",
            "ECS",
            "Fargate",
            "OpenShift",
            "Rancher",
            "Kustomize",
            "Helm",
        ),
    )
    cloud_keyword = _provider_for_item(keyword) is not None or _keyword_matches_allowed_terms(
        keyword,
        _CLOUD_CATEGORY_ALLOWED_HINTS,
    )
    monitoring_keyword = _keyword_matches_allowed_terms(keyword, _MONITORING_SECURITY_CATEGORY_ALLOWED_HINTS)
    language_keyword = _keyword_matches_allowed_terms(keyword, _LANGUAGE_TOOLS_CATEGORY_ALLOWED_HINTS)

    replacements: list[tuple[re.Pattern, object]] = []
    if devops_keyword:
        replacements.extend(
            [
                (
                    re.compile(r"\b(pipelines?)\b", re.IGNORECASE),
                    lambda match: f"{keyword} {match.group(1)}",
                ),
                (
                    re.compile(
                        r"\b(deployment|release)\s+(consistency|efficiency|errors|workflows?)\b",
                        re.IGNORECASE,
                    ),
                    lambda match: f"{keyword} {match.group(1)} {match.group(2)}",
                ),
                (
                    re.compile(r"\b(infrastructure)\b", re.IGNORECASE),
                    lambda match: f"{_keyword_modifier(keyword)} {match.group(1)}",
                ),
                (
                    re.compile(r"\b(automation)\b", re.IGNORECASE),
                    lambda match: f"{_keyword_modifier(keyword)} {match.group(1)}",
                ),
                (
                    re.compile(r"\b(artifact management|registry-backed services|registries)\b", re.IGNORECASE),
                    lambda match: f"{keyword} {match.group(1)}",
                ),
                (
                    re.compile(r"\b(platform|delivery)\b", re.IGNORECASE),
                    lambda match: f"{_keyword_modifier(keyword)} {match.group(1)}",
                ),
            ]
        )
    if container_keyword:
        replacements.append(
            (
                re.compile(r"\b(workloads?)\b", re.IGNORECASE),
                lambda match: f"{keyword} {match.group(1)}",
            )
        )
    if cloud_keyword:
        existing_infrastructure_match = re.search(r"\bacross\s+(.+?)\s+infrastructure\b", text, re.IGNORECASE)
        if existing_infrastructure_match:
            candidate = re.sub(
                r"\bacross\s+(.+?)\s+infrastructure\b",
                lambda match: f"across {_add_keyword_to_series(match.group(1), keyword)} infrastructure",
                text,
                count=1,
                flags=re.IGNORECASE,
            )
            if candidate != text and _keyword_match_count(candidate, [keyword]):
                return candidate
        replacements.extend(
            [
                (
                    re.compile(r"\b(architectures?)\b", re.IGNORECASE),
                    lambda match: f"{_keyword_modifier(keyword)} {match.group(1)}",
                ),
                (
                    re.compile(r"\b(infrastructure)\b", re.IGNORECASE),
                    lambda match: f"{_keyword_modifier(keyword)} {match.group(1)}",
                ),
            ]
        )
    if monitoring_keyword:
        replacements.extend(
            [
                (
                    re.compile(r"\b(reliability|MTTR|incident response)\b", re.IGNORECASE),
                    lambda match: f"{match.group(1)} with {keyword}",
                ),
                (
                    re.compile(r"\b(monitoring|observability|logging|alerting)\b", re.IGNORECASE),
                    lambda match: f"{keyword} {match.group(1)}",
                ),
                (
                    re.compile(r"\b(security|compliance|controls?)\b", re.IGNORECASE),
                    lambda match: f"{keyword} {match.group(1)}",
                ),
            ]
        )
    if language_keyword:
        if _is_database_keyword(keyword) and _contains_any(
            lowered_text,
            ("application", "service", "software", "api", "workload", "readiness", "development"),
        ):
            return append_phrase(f"supporting {keyword} database connectivity")
        existing_automation_match = re.search(r"\busing\s+(.+?)\s+automation\b", text, re.IGNORECASE)
        if existing_automation_match:
            candidate = re.sub(
                r"\busing\s+(.+?)\s+automation\b",
                lambda match: f"using {_add_keyword_to_series(match.group(1), keyword)} automation",
                text,
                count=1,
                flags=re.IGNORECASE,
            )
            if candidate != text and _keyword_match_count(candidate, [keyword]):
                return candidate
        existing_app_match = re.search(r"\b(?:across|for)\s+(.+?)\s+application delivery\b", text, re.IGNORECASE)
        if existing_app_match:
            candidate = re.sub(
                r"\b(across|for)\s+(.+?)\s+application delivery\b",
                lambda match: f"{match.group(1)} {_add_keyword_to_series(match.group(2), keyword)} application delivery",
                text,
                count=1,
                flags=re.IGNORECASE,
            )
            if candidate != text and _keyword_match_count(candidate, [keyword]):
                return candidate
        if _contains_any(
            lowered_text,
            ("application", "deployment", "delivery", "service", "software", "development", "enterprise", "pipeline", "release"),
        ):
            if re.search(r"\b(services?|applications?|microservices?)\s+using\b", text, re.IGNORECASE):
                candidate = re.sub(
                    r"\b(services?|applications?|microservices?)\s+using\b",
                    lambda match: f"{match.group(1)} for {keyword} application delivery using",
                    text,
                    count=1,
                    flags=re.IGNORECASE,
                )
                if candidate != text and _keyword_match_count(candidate, [keyword]):
                    return candidate
            return append_phrase(f"across {keyword} application delivery")
        if _contains_any(lowered_text, ("automated", "automation", "script", "tool", "api", "administration", "operations")):
            return append_phrase(f"using {keyword} automation")
        replacements.extend(
            [
                (
                    re.compile(r"\b(automation)\b", re.IGNORECASE),
                    lambda match: f"{keyword} {match.group(1)}",
                ),
                (
                    re.compile(r"\b(operations|administration)\b", re.IGNORECASE),
                    lambda match: f"{match.group(1)} using {keyword}",
                ),
                (
                    re.compile(r"\b(development|delivery)\b", re.IGNORECASE),
                    lambda match: f"{match.group(1)} for {keyword}",
                ),
            ]
        )

    for pattern, replacement in replacements:
        candidate = pattern.sub(replacement, text, count=1)
        if candidate != text and _keyword_match_count(candidate, [keyword]):
            return candidate

    if lowered_keyword in {"ci/cd", "release automation", "devsecops"} and "pipeline" not in text.lower():
        return re.sub(r"\.$", f" through {keyword} delivery.", text, count=1)

    if _keyword_matches_allowed_terms(keyword, _DEVOPS_ALLOWED_HINTS) and _contains_any(
        lowered_text,
        ("pipeline", "deployment", "release", "rollback", "platform", "artifact", "registry"),
    ):
        return append_phrase(f"through {keyword} delivery practices")
    if cloud_keyword and _contains_any(
        lowered_text,
        ("cloud", "architecture", "infrastructure", "workload", "network", "scalable"),
    ):
        return append_phrase(f"across {keyword} infrastructure")
    if _keyword_matches_allowed_terms(keyword, _MONITORING_SECURITY_CATEGORY_ALLOWED_HINTS) and _contains_any(
        lowered_text,
        ("reliability", "uptime", "mttr", "monitor", "incident", "security", "controls"),
    ):
        return append_phrase(f"with {keyword} coverage")
    if _keyword_matches_allowed_terms(keyword, _LANGUAGE_TOOLS_CATEGORY_ALLOWED_HINTS) and _contains_any(
        lowered_text,
        (
            "automated",
            "automation",
            "script",
            "tool",
            "api",
            "administration",
            "application",
            "deployment",
            "delivery",
            "service",
            "software",
            "development",
            "enterprise",
            "pipeline",
            "release",
        ),
    ):
        if _contains_any(
            lowered_text,
            ("application", "deployment", "delivery", "service", "software", "development", "enterprise", "pipeline", "release"),
        ):
            return append_phrase(f"across {keyword} application delivery")
        return append_phrase(f"using {keyword} automation")
    return text


_BULLET_ACTION_REWRITES = (
    (re.compile(r"^Designed\s+and\s+governed\b", re.IGNORECASE), "Architected and governed"),
    (re.compile(r"^Optimized\b", re.IGNORECASE), "Hardened and optimized"),
    (re.compile(r"^Built\s+automated\b", re.IGNORECASE), "Engineered automated"),
    (re.compile(r"^Built\b", re.IGNORECASE), "Engineered"),
    (re.compile(r"^Automated\b", re.IGNORECASE), "Codified and automated"),
    (re.compile(r"^Managed\s+the\s+lifecycle\s+of\b", re.IGNORECASE), "Governed the lifecycle of"),
    (re.compile(r"^Managed\b", re.IGNORECASE), "Governed"),
    (re.compile(r"^Developed\b", re.IGNORECASE), "Built and optimized"),
    (re.compile(r"^Provisioned\b", re.IGNORECASE), "Codified"),
    (re.compile(r"^Deployed\b", re.IGNORECASE), "Orchestrated"),
    (re.compile(r"^Supported\b", re.IGNORECASE), "Operated and improved"),
)

_DEFAULT_BULLET_HIGHLIGHTS = (
    "AWS",
    "Azure",
    "GCP",
    "Terraform",
    "Ansible",
    "Python",
    "Bash",
    "GitHub Actions",
    "Jenkins",
    "SonarQube",
    "CI/CD",
    "Docker",
    "Kubernetes",
    "ACR/ECR",
    "Nexus",
    "CloudWatch",
    "Prometheus",
    "Grafana",
    "99.9%",
    "60%+",
    "5-10 minutes",
    "10-30s",
    "MTTR",
)


def _rewrite_experience_bullet(
    original: str,
    keywords: list[str],
    *,
    avoid_keywords: list[str] | None = None,
) -> str:
    rewritten = _clean_field_text(original).strip()
    content_keywords = _content_keyword_list(keywords)
    avoid_keys = {_keyword_canonical_key(keyword) for keyword in _content_keyword_list(avoid_keywords or [])}

    def already_used(keyword: str) -> bool:
        return _keyword_canonical_key(keyword) in avoid_keys

    def unused_content_keyword(keyword: str) -> str:
        wanted_key = _keyword_canonical_key(keyword)
        for content_keyword in content_keywords:
            if (
                _keyword_canonical_key(content_keyword) == wanted_key
                and not already_used(content_keyword)
                and not _keyword_match_count(rewritten, [content_keyword])
            ):
                return content_keyword
        return ""

    for pattern, replacement in _BULLET_ACTION_REWRITES:
        rewritten, count = pattern.subn(replacement, rewritten, count=1)
        if count:
            break

    keyword_text = " ".join(content_keywords).lower()
    if (
        "ci/cd" in keyword_text
        and not already_used("CI/CD")
        and "ci/cd" not in rewritten.lower()
        and "pipeline" in rewritten.lower()
    ):
        rewritten = re.sub(r"\b(pipelines?)\b", r"CI/CD \1", rewritten, count=1, flags=re.IGNORECASE)
    if "registry" in keyword_text and not already_used("registry") and "containerized services" in rewritten.lower():
        rewritten = re.sub(
            r"\bcontainerized\s+services\b",
            "container registry-backed services",
            rewritten,
            count=1,
            flags=re.IGNORECASE,
        )
    if (
        "production reliability" in keyword_text
        and not already_used("Production Reliability")
        and "production workloads" in rewritten.lower()
    ):
        rewritten = re.sub(
            r"\bproduction\s+workloads\b",
            "production reliability workloads",
            rewritten,
            count=1,
            flags=re.IGNORECASE,
        )

    unit_testing_keyword = unused_content_keyword("Unit Testing")
    if unit_testing_keyword and re.search(r"\bquality\s+gates?\b", rewritten, re.IGNORECASE):
        rewritten = re.sub(
            r"\bquality\s+gates?\b",
            f"{unit_testing_keyword} quality gates",
            rewritten,
            count=1,
            flags=re.IGNORECASE,
        )

    dotnet_keyword = unused_content_keyword(".NET")
    if dotnet_keyword and _contains_any(
        rewritten.lower(),
        ("application", "deployment", "delivery", "service", "software", "pipeline", "release", "workload"),
    ):
        rewritten = _inject_keyword_into_bullet(rewritten, dotnet_keyword)

    bullet_keywords = _bullet_keyword_candidates(original, content_keywords)
    if avoid_keys:
        fresh_keywords = [
            keyword for keyword in bullet_keywords if _keyword_canonical_key(keyword) not in avoid_keys
        ]
        reused_keywords = [
            keyword for keyword in bullet_keywords if _keyword_canonical_key(keyword) in avoid_keys
        ]
        bullet_keywords = fresh_keywords if len(fresh_keywords) >= 3 else fresh_keywords + reused_keywords

    target_bullet_keyword_count = min(4, len(bullet_keywords))
    for keyword in bullet_keywords:
        if _keyword_match_count(rewritten, bullet_keywords) >= target_bullet_keyword_count:
            break
        rewritten = _inject_keyword_into_bullet(rewritten, keyword)

    highlight_terms = bullet_keywords + content_keywords + list(_DEFAULT_BULLET_HIGHLIGHTS)
    rewritten = _bold_keywords(rewritten, highlight_terms, limit=4)
    rewritten = _normalize_stacked_keyword_phrases(rewritten)

    if _comparison_key(rewritten) == _comparison_key(original):
        rewritten = f"Delivered {rewritten[0].lower()}{rewritten[1:]}" if rewritten else original
    return rewritten


def _repair_tailored_payload(
    payload: _TailoredPayload,
    experience_entries: list[dict] | None,
    skills_entries: list[dict] | None,
    repair_keywords: list[str],
) -> None:
    """Second-attempt repair for usable JSON that is too literal. This keeps
    the model's rewritten summary/keywords, but forces Technical Expertise
    and Professional Experience to materially change when the model leaves them
    unchanged or omits items."""
    payload.keywords = _content_keyword_list(payload.keywords + repair_keywords)
    if skills_entries is not None:
        repaired_items: list[str] = []
        for i, entry in enumerate(skills_entries):
            raw_candidate = payload.technical_expertise[i] if i < len(payload.technical_expertise) else ""
            candidate = _strip_stray_category_prefix(_clean_field_text(raw_candidate).strip())
            candidate = _enforce_category_contract(entry["label"], candidate, entry["items"])
            if (
                not candidate
                or not _is_valid_category_rewording(candidate, entry)
                or _comparison_key(candidate) == _comparison_key(entry["items"])
            ):
                candidate = _rewrite_technical_category(entry, raw_candidate, payload.keywords)
            else:
                candidate = _rewrite_technical_category(entry, candidate, payload.keywords)
            repaired_items.append(candidate)
        payload.technical_expertise = repaired_items

    if experience_entries is not None:
        repaired_groups: list[list[str]] = []
        for i, entry in enumerate(experience_entries):
            repaired_bullets: list[str] = []
            role_used_keywords: list[str] = []
            for original in entry["bullets"]:
                rewritten = _rewrite_experience_bullet(
                    original,
                    payload.keywords,
                    avoid_keywords=role_used_keywords,
                )
                repaired_bullets.append(rewritten)
                role_used_keywords.extend(_matched_keywords(rewritten, payload.keywords))
            repaired_groups.append(repaired_bullets)
        payload.experience_bullets = repaired_groups


def _validate_tailored_payload(
    payload: _TailoredPayload,
    summary_required: bool,
    experience_entries: list[dict] | None,
    skills_entries: list[dict] | None,
    *,
    require_complete: bool = True,
    reject_unchanged_categories: bool = True,
    reject_unchanged_bullets: bool = True,
    target_keywords: list[str] | None = None,
) -> None:
    """Rejects responses that would undermine forced tailoring. Older behavior
    silently fell back to untouched bullets/categories on count mismatch; this
    service now refuses structurally incomplete model output so users do not
    receive a half-tailored resume. After the retry path, callers can disable
    complete-output requirements to salvage valid tailored sections from a
    locally brittle model response instead of falling all the way back to the
    master CV."""
    target_keywords = _content_keyword_list(target_keywords or [])

    if summary_required:
        summary = _prepare_summary_text(payload.summary)
        if not summary or _looks_like_leaked_key(summary):
            if require_complete:
                raise TailoringError("Model did not return a usable tailored Executive Summary.")
            summary = ""
        payload.summary = summary

    if skills_entries is not None:
        if require_complete and len(payload.technical_expertise) != len(skills_entries):
            raise TailoringError(
                f"Model returned {len(payload.technical_expertise)} Technical Expertise categories "
                f"but {len(skills_entries)} categories were sent."
            )
        cleaned_items: list[str] = []
        for i, entry in enumerate(skills_entries, start=1):
            if i > len(payload.technical_expertise):
                cleaned_items.append("")
                continue
            raw_item = payload.technical_expertise[i - 1]
            candidate = _strip_stray_category_prefix(_clean_field_text(raw_item).strip())
            candidate = _enforce_category_contract(entry["label"], candidate, entry["items"])
            if not candidate or not _is_valid_category_rewording(candidate, entry):
                if require_complete:
                    raise TailoringError(f"Model returned an invalid Technical Expertise category at index {i}.")
                candidate = ""
            elif reject_unchanged_categories and _comparison_key(candidate) == _comparison_key(entry["items"]):
                raise TailoringError(f"Model returned an unchanged Technical Expertise category at index {i}.")
            cleaned_items.append(candidate)

        if require_complete and target_keywords:
            for i, (entry, cleaned_item) in enumerate(zip(skills_entries, cleaned_items), start=1):
                category_keywords = [
                    keyword for keyword in target_keywords if _keyword_fits_category(entry["label"], keyword)
                ]
                required_skill_matches = min(4, len(category_keywords))
                if not required_skill_matches:
                    continue
                matched_skill_keywords = _keyword_match_count(cleaned_item, category_keywords)
                if matched_skill_keywords < required_skill_matches:
                    raise TailoringError(
                        f"Model did not tailor Technical Expertise category {i} around enough target JD keywords: "
                        f"matched {matched_skill_keywords}, required {required_skill_matches}."
                    )
        payload.technical_expertise = cleaned_items

    if experience_entries is not None:
        if require_complete and len(payload.experience_bullets) != len(experience_entries):
            raise TailoringError(
                f"Model returned {len(payload.experience_bullets)} employer bullet lists "
                f"but {len(experience_entries)} employers were sent."
            )

        cleaned_groups: list[list[str]] = []
        for employer_idx, entry in enumerate(experience_entries, start=1):
            raw_bullets = (
                payload.experience_bullets[employer_idx - 1]
                if employer_idx <= len(payload.experience_bullets)
                else []
            )
            original_bullets = entry["bullets"]
            cleaned = [
                _clean_field_text(bullet).strip()
                for bullet in raw_bullets
                if bullet and not _looks_like_leaked_key(bullet)
            ]
            if len(cleaned) != len(original_bullets):
                if require_complete:
                    raise TailoringError(
                        f"Model returned {len(cleaned)} usable bullets for employer {employer_idx} "
                        f"but {len(original_bullets)} were sent."
                    )
                cleaned_groups.append(cleaned)
                continue
            for bullet_idx, (original, tailored) in enumerate(zip(original_bullets, cleaned), start=1):
                if not tailored or _looks_like_leaked_key(tailored):
                    if require_complete:
                        raise TailoringError(
                            f"Model returned an invalid bullet for employer {employer_idx}, bullet {bullet_idx}."
                        )
                    cleaned[bullet_idx - 1] = original
                    continue
                if reject_unchanged_bullets and _comparison_key(tailored) == _comparison_key(original):
                    raise TailoringError(
                        f"Model returned an unchanged bullet for employer {employer_idx}, bullet {bullet_idx}."
                    )
                bullet_keywords = _bullet_keyword_candidates(original, target_keywords)
                required_bullet_matches = min(3, len(bullet_keywords))
                if require_complete and required_bullet_matches:
                    matched_bullet_keywords = _keyword_match_count(tailored, bullet_keywords)
                    if matched_bullet_keywords < required_bullet_matches:
                        raise TailoringError(
                            f"Model tailored employer {employer_idx}, bullet {bullet_idx} with only "
                            f"{matched_bullet_keywords} JD keyword matches; required {required_bullet_matches}."
                        )
                    repeated_keywords = _repeated_long_keywords(tailored, bullet_keywords)
                    if repeated_keywords:
                        raise TailoringError(
                            f"Model repeated JD keyword(s) in employer {employer_idx}, bullet {bullet_idx}: "
                            f"{', '.join(repeated_keywords)}."
                        )
            required_keyword_bullets = _required_keyword_bullet_count(original_bullets, target_keywords)
            if require_complete and required_keyword_bullets:
                keyword_bullets = sum(1 for bullet in cleaned if _keyword_match_count(bullet, target_keywords))
                if keyword_bullets < required_keyword_bullets:
                    raise TailoringError(
                        f"Model tailored only {keyword_bullets} JD-keyword-bearing bullets for employer "
                        f"{employer_idx}; required {required_keyword_bullets}."
                    )
            required_role_keywords = _required_role_keyword_count(original_bullets, target_keywords)
            if require_complete and required_role_keywords:
                role_keywords = _keyword_list([
                    keyword
                    for original in original_bullets
                    for keyword in _bullet_keyword_candidates(original, target_keywords)
                ])
                matched_role_keywords = _keyword_match_count(" ".join(cleaned), role_keywords)
                if matched_role_keywords < required_role_keywords:
                    raise TailoringError(
                        f"Model tailored employer {employer_idx} with only {matched_role_keywords} "
                        f"distinct JD keyword matches; required {required_role_keywords}."
                    )
            cleaned_groups.append(cleaned)

        required_experience_keywords = _required_experience_keyword_count(experience_entries, target_keywords)
        if require_complete and required_experience_keywords:
            experience_keywords = _experience_keyword_candidates(experience_entries, target_keywords)
            matched_experience_keywords = _keyword_match_count(
                " ".join(" ".join(group) for group in cleaned_groups),
                experience_keywords,
            )
            if matched_experience_keywords < required_experience_keywords:
                raise TailoringError(
                    f"Model tailored Professional Experience with only {matched_experience_keywords} "
                    f"distinct JD keyword matches; required {required_experience_keywords}."
                )
        payload.experience_bullets = cleaned_groups


def _chat_messages(prompt: str) -> list[dict[str, str]]:
    user_prompt = prompt
    if prompt.startswith(SYSTEM_PROMPT):
        user_prompt = prompt[len(SYSTEM_PROMPT) :].strip()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _deterministic_summary(job_title: str, target_keywords: list[str]) -> str:
    role_label = _sanitize_resume_headline_title(job_title) or "Cloud engineering role"
    top_keywords = _content_keyword_list(target_keywords)[:4]
    if top_keywords:
        keyword_phrase = ", ".join(top_keywords)
        return (
            f"{role_label} focused on cloud infrastructure, automation, and production delivery "
            f"using {keyword_phrase}. Applies hands-on DevOps, platform, monitoring, and scripting "
            "experience to improve reliability, deployment speed, and operational execution."
        )
    return (
        f"{role_label} focused on cloud infrastructure, automation, and production delivery. "
        "Applies hands-on DevOps, platform, monitoring, and scripting experience to improve "
        "reliability, deployment speed, and operational execution."
    )


def _deterministic_tailored_payload(
    job_title: str,
    experience_entries: list[dict] | None,
    skills_entries: list[dict] | None,
    target_keywords: list[str],
) -> _TailoredPayload:
    payload = _TailoredPayload(
        keywords=_keyword_list(target_keywords),
        summary=_deterministic_summary(job_title, target_keywords),
    )
    _repair_tailored_payload(payload, experience_entries, skills_entries, target_keywords)
    return payload


async def _request_tailored_payload(settings, prompt: str, temperature: float) -> _TailoredPayload:
    api_key = (settings.openai_api_key or "").strip()
    if not api_key:
        raise LLMExecutionError("OPENAI_API_KEY is missing; set it in backend/.env.")

    # Preserve medium reasoning for Terra and Astra; omit sampling controls.
    generation_options = (
        {"reasoning_effort": "medium"}
        if settings.openai_model.startswith(("gpt-5.6-terra", "gpt-6-astra"))
        else {"temperature": temperature}
    )
    request_context = f"model='{settings.openai_model}', options={generation_options}"
    try:
        async with AsyncOpenAI(api_key=api_key, timeout=180.0, max_retries=0) as client:
            completion = await client.chat.completions.create(
                model=settings.openai_model,
                messages=_chat_messages(prompt),
                response_format={"type": "json_object"},
                **generation_options,
            )
    except APIError as exc:
        raise LLMExecutionError(
            f"OpenAI API request failed ({request_context}): {exc}"
        ) from exc

    try:
        raw_output = (completion.choices[0].message.content or "").strip()
    except (AttributeError, IndexError) as exc:
        raise LLMExecutionError(
            f"OpenAI response did not include assistant content ({request_context}): {exc}"
        ) from exc

    if not raw_output:
        raise LLMExecutionError(f"OpenAI returned an empty response ({request_context}).")

    try:
        parsed_json = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise TailoringError(
            f"OpenAI did not return valid tailored JSON ({request_context}): {exc}. "
            f"Raw response starts with: {raw_output[:500]!r}"
        ) from exc

    try:
        return _TailoredPayload.model_validate(parsed_json)
    except ValidationError as exc:
        raise TailoringError(
            f"OpenAI JSON response did not match the expected shape ({request_context}): {exc}"
        ) from exc


def _master_cv_fallback_text(sections: list[dict]) -> str:
    fallback_text = _reconstruct_tailored_text(sections, _TailoredPayload())
    if not fallback_text.strip():
        raise TailoringError("LLM failed and fallback Master CV reconstruction produced empty output.")
    return fallback_text


def _result_from_payload(
    sections: list[dict],
    payload: _TailoredPayload,
    *,
    cacheable: bool,
    used_fallback: bool = False,
    target_job_title: str = "",
) -> TailorCVResult:
    tailored_text = _reconstruct_tailored_text(sections, payload, target_job_title=target_job_title)
    if not tailored_text.strip():
        raise TailoringError("Tailoring produced empty output.")

    keywords = [k.strip() for k in payload.keywords if k and k.strip()]
    return TailorCVResult(
        keywords=keywords,
        text=tailored_text,
        cacheable=cacheable,
        used_fallback=used_fallback,
    )


def _fallback_result(sections: list[dict]) -> TailorCVResult:
    return TailorCVResult(
        keywords=[],
        text=_master_cv_fallback_text(sections),
        cacheable=False,
        used_fallback=True,
    )


async def tailor_cv(
    master_cv: MasterCV,
    job_title: str,
    company_name: str,
    job_description_text: str,
    *,
    allow_fallback: bool = False,
) -> TailorCVResult:
    """Returns a tailored result or a non-cacheable fallback result.

    The OpenAI model gets one tailoring attempt, then one retry at the same
    generation settings if the model returns malformed JSON or fails strict
    output validation. OpenAI execution failures fall back to clean Master CV
    text after logging the exact exception.
    """
    settings = get_settings()
    job_title = (job_title or "").strip() or "the target role"
    company_name = (company_name or "").strip() or "the hiring company"
    job_description_text = (job_description_text or "").strip() or job_title

    sections = json.loads(master_cv.sections_json)
    if not sections:
        raise TailoringError("Master CV has no parsed sections to tailor.")

    prompt, experience_entries, skills_entries = _build_prompt(
        sections, job_title, company_name, job_description_text
    )
    summary_required = any(_is_summary_section(s["name"]) for s in sections)
    target_keywords = _content_keyword_list(_keywords_from_text(f"{job_title}\n{job_description_text}"))
    last_payload: _TailoredPayload | None = None

    for attempt, temperature in enumerate((0.3, 0.3), start=1):
        tailored_payload: _TailoredPayload | None = None
        try:
            tailored_payload = await _request_tailored_payload(settings, prompt, temperature)
            tailored_payload.keywords = _content_keyword_list(target_keywords + tailored_payload.keywords)
            last_payload = tailored_payload
            repair_keywords = _content_keyword_list(target_keywords + tailored_payload.keywords)
            _repair_tailored_payload(
                tailored_payload,
                experience_entries,
                skills_entries,
                repair_keywords,
            )
            validation_keywords = _content_keyword_list(target_keywords + tailored_payload.keywords)
            _validate_tailored_payload(
                tailored_payload,
                summary_required,
                experience_entries,
                skills_entries,
                target_keywords=validation_keywords,
            )
            return _result_from_payload(
                sections,
                tailored_payload,
                cacheable=True,
                target_job_title=job_title,
            )
        except TailoringError as exc:
            if attempt == 1:
                logging.warning(
                    "CV tailoring attempt 1 failed; retrying once with the same model settings. Exact error: %s",
                    exc,
                )
                continue

            if isinstance(exc, LLMExecutionError):
                logging.exception(
                    "OpenAI CV tailoring failed after retry; rendering clean Master CV fallback. "
                    "Exact error: %s",
                    exc,
                )
                return _fallback_result(sections)

            repair_keywords = _content_keyword_list(
                target_keywords + (last_payload.keywords if last_payload is not None else [])
            )
            repair_candidates: list[tuple[str, _TailoredPayload]] = []
            if last_payload is not None:
                repair_candidates.append(("last parseable OpenAI payload", last_payload))
            repair_candidates.append(
                (
                    "deterministic tailored payload",
                    _deterministic_tailored_payload(
                        job_title,
                        experience_entries,
                        skills_entries,
                        repair_keywords,
                    ),
                )
            )

            for repair_source, repair_payload in repair_candidates:
                try:
                    _repair_tailored_payload(
                        repair_payload,
                        experience_entries,
                        skills_entries,
                        repair_keywords,
                    )
                    _validate_tailored_payload(
                        repair_payload,
                        summary_required,
                        experience_entries,
                        skills_entries,
                        target_keywords=target_keywords,
                    )
                    logging.warning(
                        "OpenAI output failed strict validation after retry; using %s instead of "
                        "returning 502. Exact error: %s",
                        repair_source,
                        exc,
                    )
                    return _result_from_payload(
                        sections,
                        repair_payload,
                        cacheable=True,
                        target_job_title=job_title,
                    )
                except TailoringError as repair_exc:
                    logging.exception(
                        "Tailoring repair candidate failed. Source: %s. Original error: %s. Repair error: %s",
                        repair_source,
                        exc,
                        repair_exc,
                    )

            # Both the OpenAI response and the deterministic repair candidate
            # failed strict (require_complete=True) validation. Rather than
            # surface that as an unhandled TailoringError (a 502 for the
            # caller), fall back to salvaging whichever fields of the best
            # available parsed payload DO pass a relaxed pass -- a
            # partially-tailored resume beats no resume.
            for repair_source, repair_payload in repair_candidates:
                try:
                    _validate_tailored_payload(
                        repair_payload,
                        summary_required,
                        experience_entries,
                        skills_entries,
                        require_complete=False,
                        reject_unchanged_categories=False,
                        reject_unchanged_bullets=False,
                        target_keywords=target_keywords,
                    )
                    logging.warning(
                        "OpenAI output and deterministic repair both failed strict validation; "
                        "salvaging best available %s instead of returning 502. Exact error: %s",
                        repair_source,
                        exc,
                    )
                    return _result_from_payload(
                        sections,
                        repair_payload,
                        cacheable=True,
                        target_job_title=job_title,
                    )
                except TailoringError as salvage_exc:
                    logging.exception(
                        "Salvage of best available payload failed. Source: %s. Salvage error: %s",
                        repair_source,
                        salvage_exc,
                    )

            if not allow_fallback:
                logging.exception("CV tailoring failed after retry. Exact error: %s", exc)
                raise TailoringError(f"CV tailoring failed after retry: {exc}") from exc

            logging.exception(
                "CV tailoring failed after retry; rendering clean Master CV fallback. Exact error: %s",
                exc,
            )
            return _fallback_result(sections)

    if not allow_fallback:
        raise TailoringError("CV tailoring failed after retry.")
    return _fallback_result(sections)


def _keyword_present(keyword: str, tailored_text_lower: str) -> bool:
    """Whole-word match for plain-text keywords (so "Go" doesn't match inside
    "Google"), falling back to a plain substring check for keywords containing
    punctuation a word-boundary regex can't handle (e.g. "CI/CD", "Node.js")."""
    keyword_lower = keyword.strip().lower()
    if not keyword_lower:
        return False
    if re.fullmatch(r"[\w\s\-]+", keyword_lower):
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
