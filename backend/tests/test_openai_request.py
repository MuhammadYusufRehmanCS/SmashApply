import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.config import Settings
from app.services.cv_tailor import _request_tailored_payload


class OpenAIRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_reasoning_and_legacy_temperature_requests(self):
        for model in ("gpt-5.6-terra", "gpt-6-astra", "gpt-4o-mini"):
            with self.subTest(model=model):
                client = AsyncMock()
                client.chat.completions.create.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"summary":"Tailored summary"}'))]
                )
                with patch("app.services.cv_tailor.AsyncOpenAI") as factory:
                    factory.return_value.__aenter__ = AsyncMock(return_value=client)
                    factory.return_value.__aexit__ = AsyncMock(return_value=False)
                    result = await _request_tailored_payload(
                        Settings(_env_file=None, openai_api_key="test-key", openai_model=model),
                        "Return tailored JSON", 0.3,
                    )
                self.assertEqual(result.summary, "Tailored summary")
                request = client.chat.completions.create.call_args.kwargs
                self.assertEqual(request["model"], model)
                self.assertEqual(request["response_format"], {"type": "json_object"})
                if model in ("gpt-5.6-terra", "gpt-6-astra"):
                    self.assertEqual(request["reasoning_effort"], "medium")
                    self.assertNotIn("temperature", request)
                else:
                    self.assertEqual(request["temperature"], 0.3)
                    self.assertNotIn("reasoning_effort", request)
