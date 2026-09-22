import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from openai import RateLimitError

from app.services.openai_retry import request_with_backoff, rate_limit_retry_budget, retry_delay
from app.services.cv_tailor import LLMExecutionError
from app.routers.jobs import _tailoring_failure_detail


def limited(code='rate_limit_exceeded', headers=None, message='Rate limit reached'):
    response = httpx.Response(429, headers=headers or {}, request=httpx.Request('POST', 'https://api.openai.com/v1/chat/completions'))
    return RateLimitError(message, response=response, body={'code': code, 'type': code})


class RateRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_waits_for_server_then_repeats_identical_request(self):
        call = AsyncMock(side_effect=[limited(headers={'retry-after': '3.592'}), 'ok'])
        with patch('app.services.openai_retry.asyncio.sleep', new=AsyncMock()) as sleep, \
                patch('app.services.openai_retry.uniform', return_value=.1):
            self.assertEqual(await request_with_backoff(call, model='gpt-4o', messages=['same']), 'ok')
        self.assertEqual(call.call_args_list[0], call.call_args_list[1])
        self.assertAlmostEqual(sleep.call_args.args[0], 3.692)

    async def test_quota_is_not_retried_or_mislabeled(self):
        error = limited('insufficient_quota')
        call = AsyncMock(side_effect=error)
        with patch('app.services.openai_retry.asyncio.sleep', new=AsyncMock()) as sleep:
            with self.assertRaises(RateLimitError):
                await request_with_backoff(call)
        call.assert_awaited_once()
        sleep.assert_not_awaited()
        wrapped = LLMExecutionError('provider failure')
        wrapped.__cause__ = error
        self.assertIn('insufficient API quota', _tailoring_failure_detail(wrapped))

    async def test_budget_is_shared_across_generation_and_repairs(self):
        first = AsyncMock(side_effect=[limited(), 'first'])
        second = AsyncMock(side_effect=limited())
        with patch('app.services.openai_retry.asyncio.sleep', new=AsyncMock()) as sleep:
            with rate_limit_retry_budget(2):
                self.assertEqual(await request_with_backoff(first), 'first')
                with self.assertRaises(RateLimitError):
                    await request_with_backoff(second)
        self.assertEqual(sleep.await_count, 2)
        self.assertEqual(second.await_count, 2)

    async def test_persistent_limit_is_bounded_and_not_reported_as_billing(self):
        error = limited()
        call = AsyncMock(side_effect=error)
        with patch('app.services.openai_retry.asyncio.sleep', new=AsyncMock()) as sleep:
            with self.assertRaises(RateLimitError):
                await request_with_backoff(call)
        self.assertEqual(call.await_count, 4)
        self.assertEqual(sleep.await_count, 3)
        wrapped = LLMExecutionError('provider failure')
        wrapped.__cause__ = error
        self.assertIn('temporary request/token rate limit', _tailoring_failure_detail(wrapped))
        self.assertNotIn('Check API billing', _tailoring_failure_detail(wrapped))

    async def test_cancellation_during_backoff_does_not_send_again(self):
        call = AsyncMock(side_effect=limited())
        with patch('app.services.openai_retry.asyncio.sleep', new=AsyncMock(side_effect=asyncio.CancelledError)):
            with self.assertRaises(asyncio.CancelledError):
                await request_with_backoff(call)
        call.assert_awaited_once()

    async def test_long_server_delay_is_not_ignored(self):
        call = AsyncMock(side_effect=limited(headers={'retry-after': '500'}))
        with patch('app.services.openai_retry.asyncio.sleep', new=AsyncMock()) as sleep:
            with self.assertRaises(RateLimitError):
                await request_with_backoff(call)
        sleep.assert_not_awaited()
        call.assert_awaited_once()

    def test_logged_error_message_delay_and_millisecond_header(self):
        self.assertEqual(retry_delay(limited(message='Please try again in 3.592s.'), 0), 3.592)
        self.assertEqual(retry_delay(limited(headers={'retry-after-ms': '4500'}), 0), 4.5)
        self.assertEqual(retry_delay(limited(message='Please try again in 500ms.'), 0), .5)
