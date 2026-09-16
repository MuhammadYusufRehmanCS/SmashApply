import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.cv_wording import request_wording_replacements
from app.services.cv_tailor import TailoringError


def response(value):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))])


class WordingTests(unittest.IsolatedAsyncioTestCase):
    async def test_bad_lengths_and_missing_categories_retry_only_unresolved_fields(self):
        fields = {'technical_expertise.0': {'text': 'Long list', 'max_characters': 20},
                  'technical_expertise.3': {'text': 'Another long list', 'max_characters': 20},
                  'experience_bullets.0.0': {'text': 'Long accomplishment', 'max_characters': 60}}
        responses = [
            {'technical_expertise.0': '**AWS**, Azure',
             'experience_bullets.0.0': 'Long unchanged wording ' * 10},
            {'technical_expertise.3': 'Python, Bash',
             'experience_bullets.0.0': 'Built Kafka schema checks to prevent corrupt records.'},
        ]
        client = AsyncMock()
        client.chat.completions.create.side_effect = [response(value) for value in responses]
        with patch('app.services.cv_wording.AsyncOpenAI') as factory:
            factory.return_value.__aenter__.return_value = client
            result = await request_wording_replacements(
                SimpleNamespace(openai_api_key='test', openai_model='gpt-4o'), fields, 'Engineer')
        self.assertEqual(set(result), set(fields))
        self.assertEqual(result['technical_expertise.0'], '**AWS**, Azure')
        calls = client.chat.completions.create.await_args_list
        second_schema = calls[1].kwargs['response_format']['json_schema']['schema']
        self.assertEqual(set(second_schema['required']), {'technical_expertise.3', 'experience_bullets.0.0'})
        self.assertFalse(second_schema['additionalProperties'])
        self.assertTrue(all(value == {'type': 'string'} for value in second_schema['properties'].values()))
        self.assertIn('maximum 60', calls[1].kwargs['messages'][1]['content'])

    async def test_overlong_text_never_reaches_renderer_or_gets_truncated(self):
        client = AsyncMock()
        client.chat.completions.create.return_value = response({'summary': 'Too long ' * 30})
        with patch('app.services.cv_wording.AsyncOpenAI') as factory:
            factory.return_value.__aenter__.return_value = client
            with self.assertRaisesRegex(TailoringError, 'length validation'):
                await request_wording_replacements(
                    SimpleNamespace(openai_api_key='test', openai_model='gpt-4o'),
                    {'summary': {'text': 'Original candidate', 'max_characters': 40}}, 'Engineer')
        self.assertEqual(client.chat.completions.create.await_count, 3)
