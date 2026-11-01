"""Explicit Gemini Developer API transport with mandatory billing checks."""

import asyncio
import json
import math
import re
from collections.abc import Awaitable, Callable
from email.utils import parsedate_to_datetime

import httpx

from paper_search.relevance import cache as clock
from paper_search.relevance.cache import Cache
from paper_search.relevance.dataset import (
    MODELS,
    Judgment,
    body,
    cache_key,
)

ROOT = "https://generativelanguage.googleapis.com/v1beta/models/"


class EvaluationStopped(RuntimeError):
    """A fail-closed stop that preserves previous valid judgments."""


async def safe_get(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    params: dict | None = None,
) -> dict:
    """Perform a read-only control-plane check without exposing key URLs."""
    try:
        response = await client.get(
            url, headers=headers, params=params
        )
        if response.status_code != 200:
            raise EvaluationStopped(
                f"Read-only verification failed: HTTP {response.status_code}"
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError
        return payload
    except (httpx.RequestError, ValueError):
        raise EvaluationStopped(
            "Read-only verification unavailable"
        ) from None


async def key_project(
    client: httpx.AsyncClient, api_key: str, oauth_token: str
) -> str:
    """Resolve the actual key owner; never trust an unrelated project ID."""
    result = await safe_get(
        client,
        "https://apikeys.googleapis.com/v2/keys:lookupKey",
        {"Authorization": f"Bearer {oauth_token}"},
        {"keyString": api_key},
    )
    parent = result.get("parent", "")
    match = re.fullmatch(r"projects/(\d+)/locations/global", parent)
    if not match or not result.get("name"):
        raise EvaluationStopped("Cannot verify API key project")
    return match.group(1)


async def verify_unbilled(
    client: httpx.AsyncClient, project: str, oauth_token: str
) -> None:
    """Refuse active, suspended, linked or unknown billing configurations."""
    info = await safe_get(
        client,
        f"https://cloudbilling.googleapis.com/v1/projects/{project}/billingInfo",
        {"Authorization": f"Bearer {oauth_token}"},
    )
    # No linked account, even if presently suspended or on trial credits.
    if info.get("billingEnabled") is not False or info.get(
        "billingAccountName"
    ):
        raise EvaluationStopped(
            "Billing is linked, enabled, or unverifiable"
        )


def retry_delay(response: httpx.Response, current: float) -> float:
    """Honor Retry-After and RetryInfo; never retry automatically."""
    delay = 60.0
    value = response.headers.get("Retry-After", "")
    try:
        parsed = float(value)
        if 0 <= parsed < float("inf"):
            delay = max(delay, parsed)
    except ValueError:
        try:
            delay = max(
                delay,
                parsedate_to_datetime(value).timestamp() - current,
            )
        except (ValueError, TypeError, OverflowError):
            pass
    try:
        for detail in (
            response.json().get("error", {}).get("details", [])
        ):
            value = detail.get("retryDelay", "")
            if re.fullmatch(r"\d+(?:\.\d+)?s", value):
                delay = max(delay, float(value[:-1]))
            for violation in detail.get("violations", []):
                if "perday" in json.dumps(
                    violation
                ).lower().replace("_", ""):
                    delay = max(delay, 86400)
    except (ValueError, AttributeError, TypeError):
        pass
    return delay


def parse_judgment(payload: dict, pair: dict) -> Judgment:
    """Reject truncation, safety blocks, invalid labels and invented quotes."""
    candidates = payload.get("candidates", [])
    if (
        len(candidates) != 1
        or candidates[0].get("finishReason") != "STOP"
    ):
        raise ValueError("Missing, blocked or incomplete candidate")
    text = "".join(
        part.get("text", "")
        for part in candidates[0]
        .get("content", {})
        .get("parts", [])
        if not part.get("thought")
    )
    judgment = Judgment.model_validate_json(text)
    if judgment.quote and judgment.quote not in (
        pair["title"] + "\n" + pair["abstract"]
    ):
        raise ValueError("Quote not in supplied metadata")
    if judgment.label == 2 and not judgment.quote.strip():
        raise ValueError(
            "Potential evidence requires a supplied-text quote"
        )
    return judgment


async def evaluate(
    manifest: dict,
    cache: Cache,
    client: httpx.AsyncClient,
    *,
    api_key: str,
    oauth_token: str,
    max_calls: int = 20,
    daily_limit: int = 20,
    interval: float = 15,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    time: Callable[[], float] = clock.now,
) -> int:
    """Process uncached pairs serially; stop rather than retry uncertain work.

    A reservation written before POST prevents duplicate generation after a
    crash or timeout. Such ambiguous items require manual inspection. A 429
    was rejected and may be attempted on a later run after its cooldown.
    """
    if not api_key or not oauth_token:
        raise EvaluationStopped(
            "Set GEMINI_API_KEY and GOOGLE_OAUTH_ACCESS_TOKEN"
        )
    if (
        max_calls < 1
        or daily_limit < 1
        or interval < 15
        or not math.isfinite(interval)
    ):
        raise ValueError(
            "Positive caps and interval >=15 seconds required"
        )
    model = manifest["model"]
    if cache.setting("blocked_reason"):
        raise EvaluationStopped(
            "Evaluation blocked: server model changed"
        )
    if model not in MODELS:
        raise EvaluationStopped(
            "Model is not on the free-tier allowlist"
        )
    todo = sorted(
        (
            p
            for p in manifest["pairs"]
            if cache.get(cache_key(p, model)) is None
        ),
        key=lambda p: p["id"],
    )
    if not todo:
        return 0
    if float(cache.setting("not_before") or 0) > time():
        raise EvaluationStopped(
            "Saved cooldown active; wait before resuming"
        )
    project = await key_project(client, api_key, oauth_token)
    await verify_unbilled(client, project, oauth_token)
    model_info = await safe_get(
        client, ROOT + model, {"x-goog-api-key": api_key}
    )
    if "generateContent" not in model_info.get(
        "supportedGenerationMethods", []
    ):
        raise EvaluationStopped(
            "Fixed model is unavailable for this key"
        )
    sent = 0
    for pair in todo:
        if sent >= max_calls:
            break
        delay = cache.wait_seconds(interval, time())
        if delay:
            await sleep(delay)
        # Check immediately before each request; no billing mutation endpoints.
        await verify_unbilled(client, project, oauth_token)
        key = cache_key(pair, model)
        if not cache.reserve(key, daily_limit, time()):
            break
        sent += 1
        try:
            response = await client.post(
                ROOT + model + ":generateContent",
                headers={"x-goog-api-key": api_key},
                json=body(pair, model),
            )
        except httpx.RequestError:
            cache.finish(
                key, "uncertain", {"error": "transport_uncertain"}
            )
            raise EvaluationStopped(
                "Transport outcome uncertain; not retried automatically"
            ) from None
        if response.status_code == 429:
            cache.rejected(
                key, time() + retry_delay(response, time())
            )
            raise EvaluationStopped(
                "Quota/rate limit; cooldown saved, no retry"
            )
        if response.status_code != 200:
            cache.finish(
                key, "error", {"http_status": response.status_code}
            )
            raise EvaluationStopped(
                f"Generation HTTP {response.status_code}; stopped, no retry"
            )
        try:
            payload = response.json()
            version = payload.get("modelVersion")
            if not isinstance(version, str) or not version:
                raise ValueError("Missing model version")
            previous = cache.setting("model_version")
            if previous and previous != version:
                cache.set_setting(
                    "blocked_reason", "model_version_changed"
                )
                raise ValueError("Server model version changed")
            judgment = parse_judgment(payload, pair)
            cache.set_setting("model_version", version)
        except (ValueError, KeyError, TypeError, AttributeError):
            cache.finish(
                key,
                "error",
                {"error": "invalid_or_changed_response"},
            )
            raise EvaluationStopped(
                "Invalid/blocked response or model version changed; no retry"
            ) from None
        cache.finish(
            key,
            "done",
            {
                "id": pair["id"],
                **judgment.model_dump(),
                "model_version": version,
                "usage": payload.get("usageMetadata"),
                "response_id": payload.get("responseId"),
            },
        )
    return sent
