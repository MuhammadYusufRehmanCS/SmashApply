"""Generation constraints; final semantic and PDF validation still apply."""


def prose_schema(min_words, max_words):
    # Single spaces and no Markdown markers keep this count identical to _clean().split().
    return {"type": "string", "pattern": rf"^[^\s*]+( [^\s*]+){{{min_words - 1},{max_words - 1}}}$",
            "description": f"{min_words}-{max_words} space-separated words, including labels/prefixes. Plain prose; no Markdown markers."}


def finalized_schema():
    return {"type": "object", "additionalProperties": False,
            "required": ["role_title", "keywords", "summary", "core_skills", "experience_bullets"],
            "properties": {
                "role_title": prose_schema(1, 4),
                "keywords": {"type": "array", "items": {"type": "string"}},
                "summary": prose_schema(20, 25),
                "core_skills": {"type": "array", "minItems": 3, "maxItems": 3,
                                "items": prose_schema(25, 30)},
                "experience_bullets": {"type": "object", "additionalProperties": False,
                    "required": ["arqon", "ventera"], "properties": {
                        employer: {"type": "array", "minItems": count, "maxItems": count,
                                   "items": prose_schema(28, 35)}
                        for employer, count in (("arqon", 4), ("ventera", 3))}},
            }}
