"""Bounded retry of temporary 429s, without retrying billing/quota failures."""
import asyncio
import logging
import math
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from email.utils import parsedate_to_datetime
from random import uniform

from openai import RateLimitError

_budget = ContextVar("openai_rate_retry_budget", default=None)


@contextmanager
def rate_limit_retry_budget(retries=3):
    token = _budget.set({"remaining": retries})
    try:
        yield
    finally:
        _budget.reset(token)


def is_quota_error(error):
    body = getattr(error, "body", None)
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        body = body["error"]
    body = body if isinstance(body, dict) else {}
    quota_codes = {"insufficient_quota", "billing_hard_limit_reached", "billing_not_active"}
    return any(value in quota_codes for value in
               (getattr(error, "code", None), body.get("code"), body.get("type"))
               if isinstance(value, str))


def retry_delay(error, attempt):
    headers = error.response.headers
    for key, divisor in (("retry-after-ms", 1000), ("retry-after", 1)):
        try:
            delay = float(headers[key]) / divisor
            if math.isfinite(delay) and delay >= 0:
                return delay
        except (KeyError, TypeError, ValueError):
            pass
    try:
        return max(0, parsedate_to_datetime(headers["retry-after"]).timestamp() - time.time())
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    # The observed token-per-minute response supplies its delay in the message.
    match = re.search(r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|s|seconds?)\b", str(error), re.I)
    if match:
        return float(match[1]) / (1000 if match[2].lower() == "ms" else 1)
    return min(2 ** (attempt + 1), 30)


async def request_with_backoff(operation, **kwargs):
    for attempt in range(4):
        try:
            return await operation(**kwargs)
        except RateLimitError as exc:
            budget = _budget.get()
            if (is_quota_error(exc) or attempt == 3 or
                    (budget is not None and budget["remaining"] <= 0) or
                    exc.response.headers.get("x-should-retry") == "false"):
                raise
            delay = retry_delay(exc, attempt)
            # Do not retry earlier than the provider asked or wait indefinitely.
            if delay > 120:
                raise
            if budget is not None:
                budget["remaining"] -= 1
            delay += uniform(.1, .3)
            logging.getLogger(__name__).warning(
                "Temporary OpenAI rate limit; waiting %.2fs before retrying the same request", delay)
            while delay > 0:
                chunk = min(delay, 60)
                await asyncio.sleep(chunk)
                delay -= chunk
