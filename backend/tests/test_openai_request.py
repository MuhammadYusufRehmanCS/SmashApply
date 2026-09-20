import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.config import Settings
from app.services.cv_tailor import _request_tailored_payload


class OpenAIRequestTests(unittest.IsolatedAsyncioTestCase):
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
                self.assertIn("role_title", schema["required"])
                self.assertNotIn("header", schema["properties"])
                self.assertNotIn("contact", schema["properties"])
                self.assertEqual(schema["properties"]["core_skills"]["minItems"], 3)
                self.assertEqual(schema["properties"]["core_skills"]["maxItems"], 3)
                if model in ("gpt-5.6-terra", "gpt-6-astra"):
                    self.assertEqual(request["reasoning_effort"], "medium")
                    self.assertNotIn("temperature", request)
                else:
                    self.assertEqual(request["temperature"], 0.7)
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
