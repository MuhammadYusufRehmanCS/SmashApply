"""Generation constraints; final semantic and PDF validation still apply."""

# Single source of truth for editable-field budgets: (min_words, max_words, max_characters).
# Characters are the layout limit. Measured in the finalized template (Chromium),
# the longest text that still renders in two lines with dense technical wording is:
# summary 246, Core Skills 234, bullets 222 (Selected Project 233). Caps sit a few
# characters below that. Word bounds are only guardrails against empty or runaway
# output. Characters are counted without Markdown markers.
FIELD_LIMITS = {
    "role_title": (1, 4, None),
    "summary": (14, 25, 240),
    "core_skills": (16, 30, 230),
    "experience_bullets": (16, 35, 215),
}


def prose_schema(min_words, max_words, max_characters=None):
    # Single spaces and no Markdown markers keep this count identical to _clean().split().
    # The character cap is only described, never sent as maxLength: OpenAI strict mode
    # support for string-length keywords varies by model, and Python enforces the cap.
    schema = {"type": "string", "pattern": rf"^[^\s*]+( [^\s*]+){{{min_words - 1},{max_words - 1}}}$",
              "description": f"{min_words}-{max_words} space-separated words, including labels/prefixes. Plain prose; no Markdown markers."}
    if max_characters is not None:
        schema["description"] += f" At most {max_characters} characters."
    return schema


def field_schema(field):
    return prose_schema(*FIELD_LIMITS[field])


NO_TOOLS = " Name no specific tools or technologies; describe concepts and outcomes."
ONE_MENTION = (" May name tools, but each tool (or alias) appears at most once in the whole document:"
               " in either core_skills or one experience bullet, never both.")


def described(schema, note):
    return {**schema, "description": schema["description"] + note}


def finalized_schema():
    return {"type": "object", "additionalProperties": False,
            "required": ["role_title", "keywords", "summary", "core_skills", "experience_bullets"],
            "properties": {
                "role_title": described(field_schema("role_title"), NO_TOOLS),
                "keywords": {"type": "array", "items": {"type": "string"}},
                "summary": described(field_schema("summary"), NO_TOOLS),
                "core_skills": {"type": "array", "minItems": 3, "maxItems": 3,
                                "items": described(field_schema("core_skills"), ONE_MENTION)},
                "experience_bullets": {"type": "object", "additionalProperties": False,
                    "required": ["arqon", "ventera"], "properties": {
                        employer: {"type": "array", "minItems": count, "maxItems": count,
                                   "items": described(field_schema("experience_bullets"), ONE_MENTION)}
                        for employer, count in (("arqon", 4), ("ventera", 3))}},
            }}
