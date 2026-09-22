import json
import re
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services.cv_repair import request_field_repairs
from app.services.cv_tailor import PayloadFormatError


class FieldRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_fit_request_uses_shortest_allowed_word_count(self):
        fields = {'core_skills.0': {'min_words': 25, 'max_words': 30, 'generation_word_count': 25}}
        client = AsyncMock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=json.dumps({'core_skills.0': ' '.join(['word'] * 25)})))])
        with patch('app.services.cv_repair.AsyncOpenAI') as factory:
            factory.return_value.__aenter__ = AsyncMock(return_value=client)
            factory.return_value.__aexit__ = AsyncMock(return_value=False)
            await request_field_repairs(SimpleNamespace(openai_api_key='test', openai_model='gpt-4o'), fields, {}, [])
        schema = client.chat.completions.create.call_args.kwargs['response_format']['json_schema']['schema']
        pattern = schema['properties']['core_skills.0']['pattern']
        self.assertIsNotNone(re.fullmatch(pattern, ' '.join(['word'] * 25)))
        for count in (24, 26, 30):
            self.assertIsNone(re.fullmatch(pattern, ' '.join(['word'] * count)))

    async def test_word_range_is_sent_as_generation_constraint(self):
        fields = {'core_skills.0': {'min_words': 25, 'max_words': 30}}
        client = AsyncMock()
        text = ' '.join(['word'] * 25)
        client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=json.dumps({'core_skills.0': text})))])
        with patch('app.services.cv_repair.AsyncOpenAI') as factory:
            factory.return_value.__aenter__ = AsyncMock(return_value=client)
            factory.return_value.__aexit__ = AsyncMock(return_value=False)
            await request_field_repairs(SimpleNamespace(openai_api_key='test', openai_model='gpt-4o'), fields, {}, [])
        schema = client.chat.completions.create.call_args.kwargs['response_format']['json_schema']['schema']
        pattern = schema['properties']['core_skills.0']['pattern']
        for count in (23, 24, 31):
            self.assertIsNone(re.fullmatch(pattern, ' '.join(['word'] * count)))
        for count in (25, 26, 30):
            self.assertIsNotNone(re.fullmatch(pattern, ' '.join(['word'] * count)))

    async def test_patch_schema_cannot_change_other_fields(self):
        fields = {"core_skills.0": {"text": "Cloud: AWS", "forbidden_terms": ["AWS"]}}
        client = AsyncMock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=json.dumps({"core_skills.0": "Cloud: Reliable services"})))])
        with patch('app.services.cv_repair.AsyncOpenAI') as factory:
            factory.return_value.__aenter__ = AsyncMock(return_value=client)
            factory.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await request_field_repairs(
                SimpleNamespace(openai_api_key='test', openai_model='gpt-4o'), fields, {}, [])
        self.assertEqual(set(result), set(fields))
        schema = client.chat.completions.create.call_args.kwargs['response_format']['json_schema']['schema']
        self.assertEqual(schema['required'], ['core_skills.0'])
        self.assertFalse(schema['additionalProperties'])
        client.chat.completions.create.assert_awaited_once()

    async def test_extra_keys_are_rejected_before_merge(self):
        client = AsyncMock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content='{"summary":"Valid", "role_title":"Changed"}'))])
        with patch('app.services.cv_repair.AsyncOpenAI') as factory:
            factory.return_value.__aenter__ = AsyncMock(return_value=client)
            factory.return_value.__aexit__ = AsyncMock(return_value=False)
            with self.assertRaises(PayloadFormatError):
                await request_field_repairs(SimpleNamespace(openai_api_key='test', openai_model='gpt-4o'),
                                            {'summary': {}}, {}, [])
