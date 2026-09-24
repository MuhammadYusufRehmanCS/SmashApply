"""Generation constraints; final semantic and PDF validation still apply."""


def prose_schema(min_words, max_words):
    # Single spaces and no Markdown markers keep this count identical to _clean().split().
    return {"type": "string", "pattern": rf"^[^\s*]+( [^\s*]+){{{min_words - 1},{max_words - 1}}}$",
            "description": f"{min_words}-{max_words} space-separated words, including labels/prefixes. Plain prose; no Markdown markers."}


NO_TOOLS = " Name no specific tools or technologies; describe concepts and outcomes."
ONE_MENTION = (" May name tools, but each tool (or alias) appears at most once in the whole document:"
               " in either core_skills or one experience bullet, never both.")


def described(schema, note):
    return {**schema, "description": schema["description"] + note}


def finalized_schema():
    return {"type": "object", "additionalProperties": False,
            "required": ["role_title", "keywords", "summary", "core_skills", "experience_bullets"],
            "properties": {
                "role_title": described(prose_schema(1, 4), NO_TOOLS),
                "keywords": {"type": "array", "items": {"type": "string"}},
                "summary": described(prose_schema(20, 25), NO_TOOLS),
                "core_skills": {"type": "array", "minItems": 3, "maxItems": 3,
                                "items": described(prose_schema(25, 30), ONE_MENTION)},
                "experience_bullets": {"type": "object", "additionalProperties": False,
                    "required": ["arqon", "ventera"], "properties": {
                        employer: {"type": "array", "minItems": count, "maxItems": count,
                                   "items": described(prose_schema(28, 35), ONE_MENTION)}
                        for employer, count in (("arqon", 4), ("ventera", 3))}},
            }}
