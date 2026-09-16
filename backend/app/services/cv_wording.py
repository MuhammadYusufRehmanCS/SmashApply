"""Shorten individual model-written fields without regenerating CV structure."""
import json
import logging

from openai import APIError, AsyncOpenAI

from app.services.cv_tailor import LLMExecutionError, TailoringError


async def request_wording_replacements(settings, fields: dict, title: str) -> dict[str, str]:
    """Return checked replacements keyed by exact field path; never truncate prose."""
    if not (settings.openai_api_key or '').strip():
        raise LLMExecutionError('OPENAI_API_KEY is missing; set it in backend/.env.')
    pending = dict(fields)
    accepted = {}
    feedback = {}
    options = ({'reasoning_effort': 'medium'}
               if settings.openai_model.startswith(('gpt-5.6-terra', 'gpt-6-astra'))
               else {'temperature': 0.2})
    for attempt in range(3):
        # No arrays: the model cannot add, remove, reorder, or relabel categories
        # or employer bullets. It can only replace the explicitly requested text.
        schema = {'type': 'object', 'additionalProperties': False,
                  'required': list(pending),
                  'properties': {path: {'type': 'string'} for path in pending}}
        messages = [
            {'role': 'system', 'content': (
                'You are a precise copy editor shortening already-tailored resume text. '
                'Return only the JSON object required by the response schema. Each key identifies ONE '
                'existing field; its value must be a shorter replacement string, never a list or object. '
                'The visible-character limits are mandatory and INCLUDE spaces and punctuation '
                '(exclude paired ** emphasis markers). Aim 15% below each limit. '
                'Keep each bullet a complete accomplishment: action, implementation, outcome. '
                'Keep its target-domain meaning; do not invent a new or longer initiative. '
                'Use direct verbs and short clauses. Summary: two brief sentences. Skills: a short '
                'comma-separated tool list, WITHOUT a category label. Preserve useful bold emphasis. '
                'Do not output headings, explanations, or extra fields. Never cut off a sentence.')},
            {'role': 'user', 'content': json.dumps({
                'target_role': title, 'fields_to_shorten': pending,
                'previous_errors': feedback,
            })},
        ]
        try:
            async with AsyncOpenAI(api_key=settings.openai_api_key, timeout=180, max_retries=0) as client:
                completion = await client.chat.completions.create(
                    model=settings.openai_model, messages=messages,
                    response_format={'type': 'json_schema', 'json_schema': {
                        'name': 'cv_wording_replacements', 'strict': True, 'schema': schema}},
                    **options,
                )
        except APIError as exc:
            raise LLMExecutionError('OpenAI wording-repair request failed.') from exc
        try:
            replacements = json.loads(completion.choices[0].message.content or '')
        except (ValueError, AttributeError, IndexError) as exc:
            feedback = {'response': 'Return a JSON object of field keys to replacement strings.'}
            logging.warning('Wording repair response %s was not usable JSON: %s', attempt + 1, type(exc).__name__)
            continue
        if not isinstance(replacements, dict):
            feedback = {'response': 'Return a JSON object, not an array.'}
            continue
        feedback = {}
        for path, spec in list(pending.items()):
            value = replacements.get(path)
            if not isinstance(value, str) or not value.strip():
                feedback[path] = 'Missing or empty replacement string.'
                continue
            value = value.strip()
            count = len(value.replace('**', ''))
            cap = spec['max_characters']
            if count > cap or '\n' in value:
                feedback[path] = f'Returned {count} characters; maximum {cap}. Rewrite more concisely as one line of text.'
                continue
            accepted[path] = value
            del pending[path]
        if not pending:
            return accepted
        logging.warning('Wording repair %s needs another edit: %s', attempt + 1, feedback)
    raise TailoringError('Wording replacements failed length validation: ' + json.dumps(feedback))
