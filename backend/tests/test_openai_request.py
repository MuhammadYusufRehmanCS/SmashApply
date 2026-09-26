import unittest
import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.config import Settings
from app.services.cv_tailor import _request_tailored_payload


class OpenAIRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_finalized_schema_enforces_counts_and_preserves_internal_payload(self):
        from tests.test_finalized_tailor import rewritten_candidate
        original = rewritten_candidate()
        data = original.model_dump()
        data['experience_bullets'] = dict(arqon=original.experience_bullets[0], ventera=original.experience_bullets[1])
        client = AsyncMock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])
        with patch('app.services.cv_tailor.AsyncOpenAI') as factory:
            factory.return_value.__aenter__ = AsyncMock(return_value=client)
            factory.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _request_tailored_payload(Settings(_env_file=None, openai_api_key='test'), 'Test', system_prompt='Rules')
        self.assertEqual(result.model_dump(), original.model_dump())
        request = client.chat.completions.create.call_args.kwargs
        self.assertTrue(request['response_format']['json_schema']['strict'])
        schema = request['response_format']['json_schema']['schema']['properties']
        for name, count in (('arqon', 4), ('ventera', 3)):
            group = schema['experience_bullets']['properties'][name]
            self.assertEqual((group['minItems'], group['maxItems']), (count, count))
        from app.services.cv_schema import FIELD_LIMITS
        for prop, field in ((schema['summary'], 'summary'), (schema['core_skills']['items'], 'core_skills'),
                            (group['items'], 'experience_bullets')):
            low, high, max_characters = FIELD_LIMITS[field]
            self.assertNotIn('maxLength', prop)
            self.assertIn(f'At most {max_characters} characters', prop['description'])
            for count in (low - 1, high + 1):
                self.assertIsNone(re.fullmatch(prop['pattern'], ' '.join(['word'] * count)))
            for count in (low, high):
                self.assertIsNotNone(re.fullmatch(prop['pattern'], ' '.join(['word'] * count)))

    async def test_empty_response_reports_refusal_or_finish_reason(self):
        from app.services.cv_tailor import PayloadFormatError
        for message, finish_reason, expected in (
                (SimpleNamespace(content=None, refusal="I can't help with that."), 'stop', "refusal: I can't help with that."),
                (SimpleNamespace(content='', refusal=None), 'length', 'finish_reason: length')):
            with self.subTest(expected=expected):
                client = AsyncMock()
                client.chat.completions.create.return_value = SimpleNamespace(choices=[
                    SimpleNamespace(message=message, finish_reason=finish_reason)])
                with patch('app.services.cv_tailor.AsyncOpenAI') as factory:
                    factory.return_value.__aenter__ = AsyncMock(return_value=client)
                    factory.return_value.__aexit__ = AsyncMock(return_value=False)
                    with self.assertRaisesRegex(PayloadFormatError, 'empty response') as caught:
                        await _request_tailored_payload(Settings(_env_file=None, openai_api_key='test'), 'Test',
                                                        system_prompt='Rules')
                self.assertIn(expected, str(caught.exception))

    async def test_flat_seven_bullets_are_reshaped_and_meta_text_stripped(self):
        from tests.test_finalized_tailor import rewritten_candidate
        original = rewritten_candidate()
        data = original.model_dump()
        data['experience_bullets'] = [*original.experience_bullets[0], *original.experience_bullets[1]]
        data['summary'] = original.summary + ' (24 words total)'
        data['core_skills'][1] = original.core_skills[1] + ' 29 words total. 29 words total.'
        client = AsyncMock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])
        with patch('app.services.cv_tailor.AsyncOpenAI') as factory:
            factory.return_value.__aenter__ = AsyncMock(return_value=client)
            factory.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _request_tailored_payload(Settings(_env_file=None, openai_api_key='test'), 'Test', system_prompt='Rules')
        self.assertEqual(result.experience_bullets, original.experience_bullets)
        self.assertEqual(result.summary, original.summary)
        self.assertEqual(result.core_skills, original.core_skills)

    async def test_reasoning_and_legacy_temperature_requests(self):
        for model in ("gpt-5.6-terra", "gpt-6-astra", "gpt-4o"):
            with self.subTest(model=model):
                client = AsyncMock()
                client.chat.completions.create.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"summary":"Tailored summary","technical_expertise":["Cloud infrastructure","Release engineering","Leadership and collaboration"]}'))]
                )
                with patch("app.services.cv_tailor.AsyncOpenAI") as factory:
                    factory.return_value.__aenter__ = AsyncMock(return_value=client)
                    factory.return_value.__aexit__ = AsyncMock(return_value=False)
                    result = await _request_tailored_payload(
                        Settings(_env_file=None, openai_api_key="test-key", openai_model=model),
                        "Return tailored JSON",
                    )
                self.assertEqual(result.summary, "Tailored summary")
                request = client.chat.completions.create.call_args.kwargs
                self.assertEqual(request["model"], model)
                self.assertEqual(request["response_format"]["type"], "json_schema")
                self.assertTrue(request["response_format"]["json_schema"]["strict"])
                schema = request["response_format"]["json_schema"]["schema"]
                self.assertEqual(schema["required"], ["role_title", "keywords", "summary",
                                                      "core_skills", "experience_bullets"])
                self.assertEqual(schema["properties"]["core_skills"]["minItems"], 3)
                self.assertEqual(schema["properties"]["core_skills"]["maxItems"], 3)
                if model in ("gpt-5.6-terra", "gpt-6-astra"):
                    self.assertEqual(request["reasoning_effort"], "medium")
                    self.assertNotIn("temperature", request)
                else:
                    self.assertEqual(request["temperature"], 0.1)
                    self.assertNotIn("reasoning_effort", request)

    async def test_invalid_payload_reports_field_without_private_input(self):
        from app.services.cv_tailor import PayloadFormatError, TailoringError
        from app.routers.jobs import _tailoring_failure_detail
        for content, expected in (
            ('{"core_skills":["PRIVATE_CV_TEXT"]}', 'core_skills: too_short'),
            ('PRIVATE_CV_TEXT invalid JSON', 'not valid JSON'),
        ):
            client = AsyncMock()
            client.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
            with patch('app.services.cv_tailor.AsyncOpenAI') as factory:
                factory.return_value.__aenter__ = AsyncMock(return_value=client)
                factory.return_value.__aexit__ = AsyncMock(return_value=False)
                with self.assertRaises(PayloadFormatError) as caught:
                    await _request_tailored_payload(
                        Settings(_env_file=None, openai_api_key='test-key'), 'Test')
            exhausted = TailoringError('Finalized CV did not pass after three attempts.')
            exhausted.__cause__ = caught.exception
            detail = _tailoring_failure_detail(exhausted)
            self.assertIn(expected, detail)
            self.assertNotIn('PRIVATE_CV_TEXT', detail)

    def test_connection_and_timeout_errors_have_distinct_safe_messages(self):
        from openai import APIConnectionError, APITimeoutError
        from app.services.cv_tailor import LLMExecutionError
        from app.routers.jobs import _tailoring_failure_detail
        from httpx import Request
        for cause, expected in (
            (APIConnectionError(request=Request('POST', 'https://api.openai.com/v1/chat/completions')), 'could not connect'),
            (APITimeoutError(request=Request('POST', 'https://api.openai.com/v1/chat/completions')), 'timed out'),
        ):
            error = LLMExecutionError('private provider details')
            error.__cause__ = cause
            detail = _tailoring_failure_detail(error)
            self.assertIn(expected, detail)
            self.assertNotIn('private provider details', detail)
