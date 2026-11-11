"""Shared HTTP timeout, pacing, and retry behavior for providers."""

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from paper_search.exceptions import APIClientError
from paper_search.metadata import ScientificPaper

logger = logging.getLogger(__name__)


class HTTPProvider:
    """Common transport for caller-owned HTTP clients.

    Reuse provider instances for pacing. Limits apply per instance, not across
    processes or multiple instances sharing an API key.
    """

    endpoint: str
    provider_name: str
    min_interval: float = 0

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 30,
        max_retries: int = 2,
        max_retry_delay: float = 60,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Configure timeouts and backoff; retries exclude the first try."""
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        if type(max_retries) is not int or not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be between 0 and 5")
        if not math.isfinite(max_retry_delay) or max_retry_delay < 0:
            raise ValueError("max_retry_delay must be finite and nonnegative")
        self._client = client
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._max_delay = max_retry_delay
        self._sleep = sleep
        self._clock = clock
        self._lock = asyncio.Lock()
        self._last_request: float | None = None

    def auth_headers(self) -> dict[str, str]:
        """Return provider-specific authentication headers."""
        return {}

    async def _send(
        self, params: dict[str, str], headers: dict[str, str]
    ) -> httpx.Response:
        """Allow one in-flight request per provider, with paced starts."""
        async with self._lock:
            if self._last_request is not None:
                remaining = self.min_interval - (
                    self._clock() - self._last_request
                )
                if remaining > 0:
                    await self._sleep(remaining)
            self._last_request = self._clock()
            return await self._client.get(
                self.endpoint, params=params, headers=headers,
                timeout=self._timeout, follow_redirects=False,
            )

    async def _get(self, params: dict[str, str]) -> httpx.Response:
        """Request without exposing query text or credentials in error logs."""
        headers = self.auth_headers()
        for attempt in range(self._max_retries + 1):
            delay = min(2.0**attempt, self._max_delay)
            try:
                response = await self._send(params, headers)
            except httpx.RequestError as exc:
                if attempt == self._max_retries:
                    raise APIClientError(
                        f"{self.provider_name} transport request failed"
                    ) from exc
            else:
                status = response.status_code
                if 200 <= status < 300:
                    return response
                if status != 429 and not 500 <= status < 600:
                    raise APIClientError(
                        f"{self.provider_name} returned HTTP {status}"
                    )
                if attempt == self._max_retries:
                    raise APIClientError(
                        f"{self.provider_name} returned HTTP {status}; "
                        "retries exhausted"
                    )
                retry_after = _retry_after(response.headers.get("Retry-After"))
                if retry_after is not None:
                    if retry_after > self._max_delay:
                        raise APIClientError(
                            f"{self.provider_name} retry delay exceeds "
                            "configured limit"
                        )
                    delay = max(delay, retry_after)
            logger.warning(
                "Retrying transient %s request failure", self.provider_name
            )
            await self._sleep(delay)
        raise AssertionError("Retry loop must return or raise")

    def _papers(
        self,
        response: httpx.Response,
        key: str,
        parser: Callable[[dict[str, Any]], ScientificPaper],
        limit: int,
    ) -> list[ScientificPaper]:
        """Validate JSON envelopes and skip individual malformed records."""
        try:
            payload = response.json()
        except ValueError as exc:
            raise APIClientError(
                f"{self.provider_name} returned invalid JSON"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(
            payload.get(key), list
        ):
            raise APIClientError(
                f"{self.provider_name} response has no {key} list"
            )
        papers = []
        for record in payload[key][:limit]:
            try:
                if not isinstance(record, dict):
                    raise ValueError("Paper is not an object")
                papers.append(parser(record))
            except ValueError:
                logger.warning(
                    "Skipping malformed %s paper", self.provider_name
                )
        return papers


def _retry_after(value: str | None) -> float | None:
    """Parse Retry-After seconds or HTTP date, ignoring invalid values."""
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            timestamp = parsedate_to_datetime(value)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            seconds = (timestamp - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None
