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
    LLMExecutionError, PayloadFormatError, TailoringError, _request_tailored_payload, _strip_meta_text,
    _split_experience_entries, SYSTEM_PROMPT, TailorCVResult, short_role_title,
)
from app.services.pdf_generator import CVOverflowError, build_ats_pdf
from app.services.text_sections import strip_bullet
from app.services.cv_repair import request_field_repairs
from app.services.cv_schema import FIELD_LIMITS
from app.services.openai_retry import rate_limit_retry_budget

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
    if not aliases:
        return []
    return list(re.finditer(r"(?<![\w])(?:" + "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True)) + r")(?![\w])", text, re.I))


class FinalizedValidationError(TailoringError):
    """A locally generated, safe-to-display validation reason."""


class FieldValidationError(FinalizedValidationError):
    def __init__(self, message, paths):
        self.paths = paths
        super().__init__(message)


class DuplicateTechnologyError(FinalizedValidationError):
    """Strict duplicate rejection with exact field locations for model repair."""

    def __init__(self, repairs):
        self.repairs = repairs
        super().__init__("Repeated technology; maximum 1 mention across editable fields: " +
                         "; ".join(r["technology"] + " (" + ", ".join(r["occurrences"]) + ")" for r in repairs))


def _duplicate_repairs(payload):
    fields = [("role_title", payload.role_title), ("summary", payload.summary)]
    fields += [(f"core_skills.{i}", text) for i, text in enumerate(payload.core_skills)]
    fields += [(f"experience_bullets.{i}.{j}", text)
               for i, group in enumerate(payload.experience_bullets) for j, text in enumerate(group)]
    repairs = []
    for name, aliases in TECHNOLOGIES.items():
        found = [(path, text, len(_mentions(text, aliases))) for path, text in fields
                 if _mentions(text, aliases)]
        if sum(count for _, _, count in found) <= 1:
            continue
        # Retain the concrete project evidence before general descriptions.
        keep = min(found, key=lambda item: (
            0 if item[1].startswith("Selected Project:") else
            1 if item[0].startswith("experience_bullets.") else
            2 if item[0].startswith("core_skills.") else
            3 if item[0] == "summary" else 4,
        ))[0]
        repairs.append({"technology": name, "keep_in": keep, "maximum_mentions": 1,
                        "occurrences": {path: count for path, _, count in found},
                        "rewrite_without_name_or_alias": [path for path, _, _ in found if path != keep]})
    return repairs


# Tool-free stand-ins for repeated mentions that are not list items, so prose
# stays grammatical. None of these may contain a TECHNOLOGIES name or alias.
GENERIC_TERMS = {
    "AWS": "cloud", "Azure": "cloud", "GCP": "cloud", "Kubernetes": "orchestration",
    "Terraform": "infrastructure-as-code", "Docker": "container", "Jenkins": "CI/CD",
    "Ansible": "configuration-management", "Python": "scripting", "Bash": "shell",
    "Helm": "packaging", "GitHub Actions": "CI/CD", "SonarQube": "code-quality",
    "FluxCD": "GitOps", "ArgoCD": "GitOps", "ECR": "registry", "ACR": "registry",
    "GitLab": "source-control", "Prometheus": "metrics", "Grafana": "dashboard",
    "PowerShell": "scripting", "SQL": "database", "Linux": "server",
    "Snowflake": "warehouse", "Databricks": "analytics", "Kafka": "streaming",
    "Airflow": "orchestration", "Splunk": "logging", "Sentinel": "SIEM",
    "SOC 2": "compliance", "ISO 27001": "compliance", "NIST": "compliance",
    "HIPAA": "compliance", "PCI DSS": "compliance",
}
_LEADING_SEP = re.compile(r"(?:,\s*(?:and|or)\s+|\s+(?:and|or|&)\s+|,\s*|\s*/\s*)$", re.I)
_TRAILING_SEP = re.compile(r"^(?:\s*,\s*|\s+(?:and|or|&)\s+|\s*/\s*)", re.I)


def _drop_mention(text, start, end, generic, *, title=False):
    """Remove one mention at [start, end) without leaving broken prose."""
    if text[max(0, start - 2):start] == "**" and text[end:end + 2] == "**":
        start, end = start - 2, end + 2
    head, tail = text[:start], text[end:]
    if (lead := _LEADING_SEP.search(head)) and re.search(r"\w", head[:lead.start()][-1:] or ""):
        head = head[:lead.start()]
        if re.search(r"\b(?:and|or|&)\b", lead.group(0), re.I):
            # "A, B and X" -> "A and B": move the conjunction onto the new last item.
            comma = head.rfind(", ")
            if comma >= 0 and not re.search(r"[.;:()]", head[comma:]) and len(head[comma:].split()) <= 4:
                head = head[:comma] + " " + lead.group(0).strip(" ,") + " " + head[comma + 2:]
    elif (trail := _TRAILING_SEP.match(tail)) and re.match(r"\s*\w", tail[trail.end():]):
        tail = tail[trail.end():]
    elif head.endswith("(") and tail.startswith(")"):
        head, tail = head[:-1].rstrip(), tail[1:]
    elif title:
        pass
    else:
        if generic[0] in "aeiou" and re.search(r"\ba $", head, re.I):
            head = head[:-2] + head[-2] + "n "
        elif generic[0] not in "aeiou" and re.search(r"\ban $", head, re.I):
            head = head[:-3] + head[-3] + " "
        head += generic if head.strip() else generic[:1].upper() + generic[1:]
    text = re.sub(r"\s{2,}", " ", head + tail)
    text = re.sub(r"\s+([,.;:)])", r"\1", text).replace("()", "")
    return re.sub(r"(:|^)\s*[,/]\s*", r"\1 ", text).strip()


def dedupe_technologies(payload):
    """Keep each technology once, in the field validation prefers, and scrub the rest.

    LLMs routinely repeat a tool across fields despite the one-mention rule;
    fixing that locally is cheaper and more reliable than another model call.
    """
    fields = _editable_fields(payload)
    for repair in _duplicate_repairs(payload):
        aliases = TECHNOLOGIES[repair["technology"]]
        generic = GENERIC_TERMS.get(repair["technology"], "tooling")
        for path in repair["occurrences"]:
            text = fields[path]
            matches = _mentions(text, aliases)
            # Right to left so earlier offsets stay valid.
            for match in reversed(matches[1:] if path == repair["keep_in"] else matches):
                text = _drop_mention(text, match.start(), match.end(), generic, title=path == "role_title")
            fields[path] = text
    if not fields["role_title"].strip():
        fields["role_title"] = payload.role_title
    return _with_fields(payload, fields)


def _with_fields(payload, fields):
    groups = [list(group) for group in payload.experience_bullets]
    for path, text in fields.items():
        if path.startswith("experience_bullets."):
            _, i, j = path.split(".")
            groups[int(i)][int(j)] = text
    return payload.model_copy(update={
        "role_title": fields["role_title"], "summary": fields["summary"],
        "core_skills": [fields[f"core_skills.{i}"] for i in range(len(payload.core_skills))],
        "experience_bullets": groups,
    })


def _word_range(path):
    """Guardrail word bounds per editable field (see FIELD_LIMITS)."""
    return FIELD_LIMITS[path.split(".")[0]][:2]


def _char_limit(path):
    """Layout limit per editable field: characters, not words, decide line wraps."""
    return FIELD_LIMITS[path.split(".")[0]][2]


def _char_count(text):
    return len(_clean(text))


def _budget_text(path):
    _, high = _word_range(path)
    limit = _char_limit(path)
    return f"at most {high} words" + (f" and {limit} characters" if limit is not None else "") + " each."


def _over_budget(path, text):
    limit = _char_limit(path)
    return _word_count(text) > _word_range(path)[1] or (limit is not None and _char_count(text) > limit)


# Tool-, metric- and claim-free tails for near-miss fields. Larger gaps than
# MAX_PADDING_WORDS mean thin content, which goes to the model repair loop instead.
PADDING_PHRASES = (
    " for engineering and operations teams", ", supporting dependable business outcomes",
    ", with clear stakeholder communication", " through documented, repeatable practices",
    ", aligned with delivery priorities", " while maintaining operational stability",
    " across production environments", " with accountable ownership", " consistently", " reliably",
)
MAX_PADDING_WORDS = 8
_TAIL_STOPWORDS = {"a", "an", "the", "and", "or", "&", "with", "to", "for", "by", "of", "in", "on", "at",
                   "into", "from", "via", "while", "through", "across", "including", "using", "as", "-"}
_STRONG_CLAUSE_STARTS = {"while", "through", "across", "using", "including", "enabling", "ensuring",
                         "supporting", "reducing", "improving", "making", "helping", "so", "which", "that"}
_WEAK_CLAUSE_STARTS = {"and", "with", "by", "to", "for"}


def _word_count(text):
    return len(_clean(text).split())


def _trim_to_budget(text, high, low=0, max_chars=None):
    """Cut an over-budget field (words or characters) at its cleanest clause boundary."""
    period = 1 if text.rstrip().endswith(".") else 0

    def fits(tokens):
        joined = " ".join(tokens)
        return _word_count(joined) <= high and (max_chars is None or _char_count(joined.rstrip(".")) + period <= max_chars)

    tokens = text.split()
    if fits(tokens):
        return text

    def strip_tail(head):
        head = list(head)
        while head and head[-1].lower().strip(",;*") in _TAIL_STOPWORDS:
            head.pop()
        return head

    def boundary_rank(cut):
        # Lower is cleaner: sentence end, strong clause, comma (may split a list), weak joiner.
        last, following = tokens[cut - 1].rstrip("*"), tokens[cut].lower().strip("*,")
        return (0 if last.endswith((".", ";")) else 1 if following in _STRONG_CLAUSE_STARTS
                else 2 if last.endswith(",") else 3 if following in _WEAK_CLAUSE_STARTS else None)

    kept = None
    candidates = []
    for cut in range(len(tokens) - 1, 0, -1):
        head = tokens[:cut]
        rank = boundary_rank(cut)
        if rank is None or not fits(head):
            continue
        head = strip_tail(head)
        if _word_count(" ".join(head)) >= low:
            candidates.append((rank, -cut, head))
    if candidates:
        kept = min(candidates, key=lambda item: item[:2])[2]
    if kept is None:  # No clean clause boundary: hard cut at the budget.
        kept = tokens
        while kept and not fits(kept):
            kept = kept[:-1]
        kept = strip_tail(kept) or kept
    result = " ".join(kept).rstrip(",;:-. ")
    if result.count("**") % 2:
        cut = result.rfind("**")
        result = result[:cut] + result[cut + 2:]
    return result + "." if text.rstrip().endswith(".") and not result.endswith(".") else result


def _pad_words(text, low, high, used, max_chars=None):
    """Append neutral tails to a field a few words short of its minimum, within its character cap."""
    count = _word_count(text)
    if not text.strip() or count >= low or low - count > MAX_PADDING_WORDS:
        return text
    body = text.rstrip()
    end = "." if body.endswith(".") else ""
    body = body.rstrip(".")
    while count < low:
        options = [phrase for phrase in PADDING_PHRASES if _word_count(phrase.strip(" ,")) <= high - count
                   and phrase.strip(" ,").casefold() not in body.casefold()
                   and (max_chars is None or _char_count(body + phrase + end) <= max_chars)]
        if not options:
            break
        # Prefer phrases no other field used, so tails do not read as boilerplate.
        phrase = max([p for p in options if p not in used] or options, key=lambda p: _word_count(p.strip(" ,")))
        used.add(phrase)
        body += phrase
        count += _word_count(phrase.strip(" ,"))
    return body + end


def _fit_field(path, text, used, *, pad=True):
    low, high = _word_range(path)
    limit = _char_limit(path)
    text = _trim_to_budget(text, high, low, limit)
    return _pad_words(text, low, high, used, limit) if pad else text


def fit_field_budgets(payload, *, pad=True):
    """Trim (and optionally pad) each prose field into its character and word budget before validation.

    Prompts cannot make a model hit exact lengths on one attempt; near misses
    are corrected here instead of costing a repair request.
    """
    fields, used = _editable_fields(payload), set()
    for path, text in fields.items():
        if path != "role_title":  # short_role_title owns the title.
            fields[path] = _fit_field(path, text, used, pad=pad)
    return _with_fields(payload, fields)


def prepare_payload(payload, *, pad=True):
    """Local, deterministic clean-up of model output: meta-text, one mention per tool, then length budgets."""
    fields = {path: _strip_meta_text(text) for path, text in _editable_fields(payload).items()}
    return fit_field_budgets(dedupe_technologies(_with_fields(payload, fields)), pad=pad)


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


def _validate(payload, master_text: str | None = None, job_description: str = ""):
    payload = _normalized_payload(payload)
    if not payload.role_title.strip() or len(payload.role_title.split()) > 4 or len(payload.role_title) > 32 or "\n" in payload.role_title:
        raise FinalizedValidationError("Role title must be 1-4 words and at most 32 characters on one line.")
    if not payload.summary.split() or "\n" in payload.summary or _over_budget("summary", payload.summary):
        raise FieldValidationError("Executive Summary must be one paragraph within its length budget: "
                                   + _budget_text("summary"), ["summary"])
    skills = payload.core_skills
    if len(skills) != 3 or any(":" not in s or not s.split(":", 1)[1].strip() for s in skills):
        raise FinalizedValidationError("Core Skills must contain exactly three labeled, nonempty bullets.")
    if not skills[2].startswith(LEADERSHIP):
        raise FinalizedValidationError("Core Skills bullet 3 must cover Leadership & Cross-Functional Collaboration.")
    if skills[0].split(":", 1)[0].casefold() == skills[1].split(":", 1)[0].casefold():
        raise FinalizedValidationError("Core Skills domain labels must be distinct.")
    long_skills = [f"core_skills.{i}" for i, skill in enumerate(skills) if _over_budget(f"core_skills.{i}", skill)]
    if long_skills:
        raise FieldValidationError("Core Skills length budget exceeded: " + _budget_text("core_skills"), long_skills)
    groups = payload.experience_bullets
    if len(groups) != 2 or [len(g) for g in groups] != [4, 3]:
        raise FinalizedValidationError("Experience must have exactly four Arqon and three Ventera bullets.")
    for employer, (group, prefix) in enumerate(zip(groups, PROJECTS)):
        if not group[-1].startswith(prefix) or not group[-1][len(prefix):].strip():
            raise FieldValidationError("Selected Project prefix or project description is missing: " + prefix,
                                       [f'experience_bullets.{employer}.{len(group) - 1}'])
        misplaced = [f'experience_bullets.{employer}.{index}' for index, bullet in enumerate(group[:-1])
                     if 'selected project:' in bullet.lower()]
        if misplaced:
            raise FieldValidationError("Selected Projects must occur only in the final employer bullet. Rewrite these as standard achievements without a project label.", misplaced)
    long_bullets = [f"experience_bullets.{i}.{j}" for i, group in enumerate(groups)
                    for j, bullet in enumerate(group) if _over_budget(f"experience_bullets.{i}.{j}", bullet)]
    if long_bullets:
        raise FieldValidationError("Experience length budget exceeded: " + _budget_text("experience_bullets"),
                                   long_bullets)
    prose = [payload.summary, *skills, *groups[0], *groups[1]]
    if any(not text.strip() or "\n" in text for text in prose):
        raise FinalizedValidationError("Every bullet must be nonempty, single-paragraph text.")
    # Exact and near-duplicate claim checks remove category/project labels first.
    claims = [re.sub(r"[^a-z0-9 ]", "", _clean(text.split(":", 1)[-1]).lower()) for text in prose]
    for index, claim in enumerate(claims):
        for other in claims[:index]:
            if SequenceMatcher(None, claim, other).ratio() >= 0.82:
                raise FinalizedValidationError("Duplicate or near-duplicate claim across resume sections.")
    repairs = _duplicate_repairs(payload)
    if repairs:
        raise DuplicateTechnologyError(repairs)
    metrics = re.findall(r"\b\d+(?:\.\d+)?\s*%", " ".join(prose))
    if len(metrics) != len(set(metrics)):
        raise FinalizedValidationError("Repeated quantified accomplishment across sections.")


def _context(source, payload, master_text=None, job_description=""):
    payload = _normalized_payload(payload).model_copy(update={"role_title": short_role_title(payload.role_title)})
    _validate(payload, master_text, job_description)
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


def _validate_active_rewrite(payload, master):
    """Reject copied experience locally instead of trusting a prompt alone."""
    sections = json.loads(master.sections_json)
    text = next((s["content"] for s in sections if "experience" in s["name"].lower()), "")
    entries = _split_experience_entries(text)
    def claim(value):
        value = _clean(value).lower()
        if value.startswith("selected project:"):
            value = value.partition(" - ")[2]
        return re.sub(r"[^a-z0-9]+", " ", value).strip()
    copied = []
    copied_paths = []
    for employer, (entry, bullets) in enumerate(zip(entries, payload.experience_bullets), 1):
        originals = [claim(b) for b in entry["bullets"]]
        for index, bullet in enumerate(bullets, 1):
            candidate = claim(bullet)
            if candidate and any(SequenceMatcher(None, candidate, original).ratio() >= .96 for original in originals):
                copied.append(f"employer {employer} bullet {index}")
                copied_paths.append(f"experience_bullets.{employer - 1}.{index - 1}")
    if copied:
        raise FieldValidationError("Experience wording is unchanged or minimally edited: " +
                                       "; ".join(copied) + ". Rewrite the technical action and scope for the JD using role-aligned scope.", copied_paths)
    # Word floors only catch degenerate output; the character caps govern layout.
    underfilled = []
    underfilled_paths = []
    labels = {"summary": "summary",
              **{f"core_skills.{i}": f"Core Skills {i + 1}" for i in range(len(payload.core_skills))},
              **{f"experience_bullets.{i}.{j}": f"employer {i + 1} bullet {j + 1}"
                 for i, group in enumerate(payload.experience_bullets) for j in range(len(group))}}
    fields = _editable_fields(payload)
    for path, label in labels.items():
        low, high = _word_range(path)
        if _word_count(fields[path]) < low:
            underfilled.append(f"{label}: target {low}-{high} words")
            underfilled_paths.append(path)
    if underfilled:
        raise FieldValidationError("Insufficient technical detail: " + "; ".join(underfilled) +
                                       ". Expand with concrete role-aligned action, implementation and quantified impact.", underfilled_paths)


def _editable_fields(payload):
    return {"role_title": payload.role_title, "summary": payload.summary,
            **{f"core_skills.{i}": text for i, text in enumerate(payload.core_skills)},
            **{f"experience_bullets.{i}.{j}": text for i, group in enumerate(payload.experience_bullets)
               for j, text in enumerate(group)}}


def _repair_plan(payload, error, previous=None):
    fields = _editable_fields(payload)
    paths = set(getattr(error, "paths", []))
    caps = {}
    line_targets = {}
    if isinstance(error, DuplicateTechnologyError):
        for repair in error.repairs:
            paths.update(repair["rewrite_without_name_or_alias"])
            if repair["occurrences"][repair["keep_in"]] > 1:
                paths.add(repair["keep_in"])
    elif isinstance(error, CVOverflowError):
        if error.measurements:
            measured = error.measurements
            logging.getLogger(__name__).warning(
                'PDF fit: %.1fpx used / %.1fpx available; field lines=%s',
                measured['content_height'], measured['available_height'],
                {f['path']: f['lines'] for f in measured['fields']})
            remaining = max(measured['content_height'] - measured['available_height'],
                            min((f['line_height'] for f in measured['fields']), default=0))
            # Reclaim only the measured excess. Experience may occupy three lines;
            # do not squeeze every field to two lines after any page overflow.
            ordered = sorted(measured['fields'], key=lambda f: (
                -max(0, f['lines'] - (3 if f['path'].startswith('experience') else 2)),
                -f['lines'], -f['line_height']))
            for field in ordered:
                path = field["path"].replace("technical_expertise.", "core_skills.")
                minimum = 2
                target_lines = field['lines']
                while remaining > 0 and target_lines > minimum:
                    target_lines -= 1
                    remaining -= field['line_height']
                if target_lines < field['lines']:
                    paths.add(path)
                    line_targets[path] = dict(rendered_lines=field['lines'], target_lines=target_lines)
                    # A width-based estimate guides rewriting; actual glyph widths
                    # and PDF pagination, not string length, decide whether it fits.
                    prefix = fields.get(path, "").partition(":")[0] + ": " if path.startswith("core_skills.") else ""
                    caps[path] = int(.9 * (field["width"] * target_lines - field["prefix_width"]) /
                                     max(field["average_char_width"], 1)) + len(prefix)
                    prior = (previous or {}).get(path, {})
                    if prior.get('target_lines') == target_lines and prior.get('target_characters'):
                        # The previous estimate did not produce fewer physical lines.
                        # Tighten the guidance instead of asking for the same failed fit.
                        caps[path] = min(caps[path], int(prior['target_characters'] * .9))
        if not paths:
            paths = set(fields) - {"role_title"}
    elif "Executive Summary" in str(error):
        paths.add("summary")
    elif not paths:
        paths.update(path for path, text in fields.items() if path != "role_title" and _over_budget(path, text))
    # Assign each technology to its existing field, with concrete projects taking priority.
    owners = {r["technology"]: r["keep_in"] for r in _duplicate_repairs(payload)}
    for name, aliases in TECHNOLOGIES.items():
        owners.setdefault(name, next((path for path, text in fields.items() if _mentions(text, aliases)), None))
    plan = {}
    for path in sorted(paths):
        if path not in fields:
            continue
        text = fields[path]
        low, high = _word_range(path)
        project_paths = {'experience_bullets.0.3': PROJECTS[0], 'experience_bullets.1.2': PROJECTS[1]}
        prefix = text.partition(":")[0] + ":" if path.startswith("core_skills.") else project_paths.get(path, '')
        spec = dict(text=text, min_words=low, max_words=high, required_prefix=prefix,
                    forbidden_terms=[alias for name, aliases in TECHNOLOGIES.items()
                                     if owners[name] is not None and owners[name] != path for alias in aliases])
        if _char_limit(path) is not None:
            spec["max_characters"] = _char_limit(path)
        previous_spec = (previous or {}).get(path, {})
        if "last_rejection" in previous_spec:
            spec["last_rejection"] = previous_spec["last_rejection"]
        old_cap = previous_spec.get("target_characters")
        cap = caps.get(path, old_cap)
        if cap is not None:
            spec["target_characters"] = cap
        # Overflowing fields shrink by characters (target_characters), not by a forced word count.
        if path in line_targets:
            spec.update(line_targets[path])
        elif previous_spec.get('target_lines'):
            for key in ('rendered_lines', 'target_lines'):
                spec[key] = previous_spec[key]
        plan[path] = spec
    return plan


def _apply_field_repairs(payload, replacements, plan):
    data = payload.model_dump()
    errors = []
    for path, spec in plan.items():
        value = replacements.get(path)
        if isinstance(value, str) and value.strip() and "\n" not in value:
            value = _strip_meta_text(value)
            limit = spec.get("max_characters")
            value = _pad_words(_trim_to_budget(value, spec["max_words"], spec["min_words"], limit),
                               spec["min_words"], spec["max_words"], set(), limit)
        reasons = []
        if not isinstance(value, str):
            reasons.append("Return a nonempty text string.")
        else:
            words = _word_count(value)
            characters = _char_count(value)
            if "\n" in value:
                reasons.append("Remove newlines; keep one paragraph.")
            if not spec["min_words"] <= words <= spec["max_words"]:
                reasons.append(f"Returned {words} words; required {spec['min_words']}-{spec['max_words']} words.")
            if spec["required_prefix"] and not value.startswith(spec["required_prefix"]):
                reasons.append("Preserve the required_prefix exactly.")
            if "max_characters" in spec and characters > spec["max_characters"]:
                reasons.append(f"Returned {characters} characters; maximum {spec['max_characters']}.")
            forbidden = sorted({match[0] for match in _mentions(value, spec["forbidden_terms"])})
            if forbidden:
                reasons.append("Remove forbidden technology names/aliases: " + ", ".join(forbidden))
            repeated = [name for name, aliases in TECHNOLOGIES.items() if len(_mentions(value, aliases)) > 1]
            if repeated:
                reasons.append("Use each technology only once: " + ", ".join(repeated))
        if reasons:
            if spec.get("last_rejection", {}).get("text") == value:
                reasons.append("This is identical to the previous rejected replacement; produce a different correction.")
            spec["last_rejection"] = {"text": value if isinstance(value, str) else None, "reasons": reasons}
            errors.append(path)
            continue
        spec.pop("last_rejection", None)
        target = data
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[int(part)] if isinstance(target, list) else target[part]
        target[int(parts[-1]) if isinstance(target, list) else parts[-1]] = value.strip()
    return type(payload).model_validate(data), errors


async def generate_tailored_result(raw_email: str, master: MasterCV, job_title: str = "Email referral"):
    settings = get_settings()
    try:
        with rate_limit_retry_budget(3):
            async with asyncio.timeout(settings.tailoring_timeout_seconds):
                return await _generate_with_repairs(raw_email, master, job_title, settings)
    except TimeoutError as exc:
        raise FinalizedValidationError(
            f"Tailoring stopped after {settings.tailoring_timeout_seconds} seconds. No invalid resume was accepted.") from exc


async def _generate_with_repairs(raw_email, master, job_title, settings):
    source = _source_context(master)
    from app.services.cv_tailor import jd_keywords
    target_terms = jd_keywords(raw_email)
    request_context = {"target_role": job_title, "job_description": raw_email,
                       "master_cv_baseline": master.raw_text,
                       "target_keywords": target_terms,
                       "writing_goal": "Expand project scope, technical workstreams, tooling and metrics to align directly with the target role. Use senior ownership verbs and clear action, implementation and quantified impact. Keep employer headers and credentials unchanged; avoid keyword stuffing."}
    base_prompt = json.dumps(request_context, ensure_ascii=False)
    payload, last_error, plan = None, None, {}
    feedback = []
    for attempt in range(settings.tailoring_max_attempts):
        try:
            if payload is not None and plan:
                replacements = await request_field_repairs(
                    settings, plan, dict(request_context, current_candidate=payload.model_dump()), feedback[-4:])
                payload, rejected = _apply_field_repairs(payload, replacements, plan)
                if rejected:
                    reasons = {path: plan[path]["last_rejection"]["reasons"] for path in rejected}
                    raise FieldValidationError("Field repairs rejected: " + json.dumps(reasons), rejected)
            else:
                prompt = base_prompt
                if last_error is not None:
                    prompt += "\nCorrect all recorded validation failures: " + json.dumps(feedback[-4:])
                if payload is not None:
                    prompt += "\nPrevious candidate: " + payload.model_dump_json()
                payload = await _request_tailored_payload(settings, prompt, system_prompt=RULEBOOK)
            payload = prepare_payload(payload)
            context = _context(source, payload, master.raw_text, raw_email)
            _validate_active_rewrite(payload, master)
            pdf = await asyncio.to_thread(build_ats_pdf, context)
            from app.services.cv_tailor import jd_keywords
            return TailorCVResult(jd_keywords(raw_email, payload.keywords), _serialize(context), True, False), pdf
        except LLMExecutionError:
            raise
        except (TailoringError, CVOverflowError) as exc:
            last_error = exc
            feedback.append(str(exc))
            logging.getLogger(__name__).warning("Finalized CV attempt %s/%s rejected (%s): %s",
                                               attempt + 1, settings.tailoring_max_attempts, type(exc).__name__, exc)
            if payload is not None:
                # Malformed patch responses retry the same fields without losing valid text.
                if not isinstance(exc, PayloadFormatError):
                    plan = _repair_plan(payload, exc, plan)
    raise TailoringError(f"Finalized CV did not pass validation after {settings.tailoring_max_attempts} attempts. "
                        "No invalid resume was accepted.") from last_error


async def generate_tailored_resume(raw_email: str, master: MasterCV, job_title: str = "Email referral") -> bytes:
    _, pdf = await generate_tailored_result(raw_email, master, job_title)
    return pdf
