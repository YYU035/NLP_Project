"""All relevance API tests use MockTransport; no Google calls are made."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from paper_search.relevance.cache import Cache
from paper_search.relevance.dataset import (
    Judgment,
    blinded_input,
    body,
    cache_key,
    csv_write,
    prepare,
    read_pairs,
)
from paper_search.relevance.gemini import (
    EvaluationStopped,
    evaluate,
    parse_judgment,
    retry_delay,
)
from paper_search.relevance.report import generate_report


def pair(identifier: str = "opaque") -> dict:
    return {
        "id": identifier,
        "claim_id": "c01",
        "identity": identifier,
        "claim": "Exercise improves memory.",
        "title": "Exercise and memory",
        "abstract": "Memory improved.",
        "missing_abstract": False,
    }


def response(
    label: int = 2, version: str = "gemini-2.5-flash"
) -> dict:
    return {
        "modelVersion": version,
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "label": label,
                                    "justification": (
                                        "Relevant outcome reported."
                                    ),
                                    "quote": "Memory improved.",
                                }
                            )
                        }
                    ]
                },
            }
        ],
    }


def transport(handler):
    """Wrap a test-specific generate handler with billing/model endpoints."""

    def wrapped(request: httpx.Request) -> httpx.Response:
        if request.url.host == "apikeys.googleapis.com":
            assert request.url.params["keyString"] == "secret"
            return httpx.Response(
                200,
                json={
                    "parent": "projects/123/locations/global",
                    "name": "projects/123/locations/global/keys/key-id",
                },
            )
        if request.url.host == "cloudbilling.googleapis.com":
            assert (
                request.url.path == "/v1/projects/123/billingInfo"
            )
            return httpx.Response(
                200,
                json={
                    "billingEnabled": False,
                    "billingAccountName": "",
                },
            )
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "supportedGenerationMethods": [
                        "generateContent"
                    ]
                },
            )
        assert (
            request.url.host == "generativelanguage.googleapis.com"
        )
        return handler(request)

    return httpx.MockTransport(wrapped)


def run_evaluation(
    cache: Cache, pairs: list[dict], adapter, **options
) -> int:
    async def run() -> int:
        async with httpx.AsyncClient(transport=adapter) as client:
            return await evaluate(
                {"model": "gemini-2.5-flash", "pairs": pairs},
                cache,
                client,
                api_key="secret",
                oauth_token="oauth-secret",
                time=lambda: 100000.0,
                **options,
            )

    return asyncio.run(run())


def test_blinding_and_cache_key() -> None:
    item = {
        **pair(),
        "provider": "arxiv",
        "rank": 1,
        "query": "secret query",
        "arm": "alternative",
    }
    payload = body(item, "gemini-2.5-flash")
    sent = json.loads(payload["contents"][0]["parts"][0]["text"])
    assert sent == blinded_input(item)
    assert set(sent) == {"claim", "title", "abstract"}
    assert "tools" not in payload
    changed = {**item, "provider": "openalex", "rank": 10}
    assert cache_key(item, "gemini-2.5-flash") == cache_key(
        changed, "gemini-2.5-flash"
    )
    assert cache_key(item, "gemini-2.5-flash") != cache_key(
        {**item, "claim": "Other claim"}, "gemini-2.5-flash"
    )


def test_valid_response_cached_without_repeat_calls(
    tmp_path,
) -> None:
    cache = Cache(tmp_path / "cache.db")
    calls = []

    def handler(request):
        calls.append(request)
        assert request.headers["x-goog-api-key"] == "secret"
        return httpx.Response(200, json=response())

    adapter = transport(handler)
    assert run_evaluation(cache, [pair()], adapter) == 1
    assert run_evaluation(cache, [pair()], adapter) == 0
    assert len(calls) == 1
    assert (
        cache.get(cache_key(pair(), "gemini-2.5-flash"))[0]
        == "done"
    )
    cache.close()


@pytest.mark.parametrize(
    "billing",
    [
        {"billingEnabled": True},
        {},
        {
            "billingEnabled": False,
            "billingAccountName": "billingAccounts/linked",
        },
    ],
)
def test_billing_guard_refuses_before_generation(
    tmp_path, billing
) -> None:
    cache = Cache(tmp_path / "cache.db")
    seen = []

    def handler(request):
        seen.append(request.method)
        if request.url.host == "apikeys.googleapis.com":
            return httpx.Response(
                200,
                json={
                    "parent": "projects/123/locations/global",
                    "name": "key",
                },
            )
        return httpx.Response(200, json=billing)

    with pytest.raises(EvaluationStopped, match="Billing"):
        run_evaluation(
            cache, [pair()], httpx.MockTransport(handler)
        )
    assert "POST" not in seen
    cache.close()


def test_429_stops_and_persists_cooldown(tmp_path) -> None:
    cache = Cache(tmp_path / "cache.db")
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "120"})

    with pytest.raises(EvaluationStopped, match="Quota"):
        run_evaluation(cache, [pair()], transport(handler))
    assert cache.setting("not_before") == "100120.0"
    with pytest.raises(EvaluationStopped, match="cooldown"):
        run_evaluation(cache, [pair()], transport(handler))
    assert len(calls) == 1
    assert cache.get(cache_key(pair(), "gemini-2.5-flash")) is None
    cache.close()


def test_timeout_is_not_automatically_resent_after_restart(
    tmp_path,
) -> None:
    path = tmp_path / "cache.db"
    cache = Cache(path)

    def handler(request):
        raise httpx.ReadTimeout("secret should never be logged")

    with pytest.raises(
        EvaluationStopped, match="uncertain"
    ) as error:
        run_evaluation(cache, [pair()], transport(handler))
    assert "secret" not in str(error.value)
    cache.close()
    resumed = Cache(path)
    assert (
        run_evaluation(resumed, [pair()], transport(handler)) == 0
    )
    assert (
        resumed.get(cache_key(pair(), "gemini-2.5-flash"))[0]
        == "uncertain"
    )
    resumed.close()


def test_quota_counts_survive_restart(tmp_path) -> None:
    path = tmp_path / "cache.db"
    cache = Cache(path)
    assert cache.reserve("first", 1, 100000)
    cache.close()
    resumed = Cache(path)
    assert not resumed.reserve("second", 1, 100001)
    assert resumed.wait_seconds(15, 100001) == 14
    assert resumed.reserve("second", 1, 200000)
    resumed.close()


def test_model_version_change_blocks_future_calls(tmp_path) -> None:
    cache = Cache(tmp_path / "cache.db")
    cache.set_setting("model_version", "previous-version")
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=response())

    with pytest.raises(EvaluationStopped, match="version changed"):
        run_evaluation(cache, [pair()], transport(handler))
    with pytest.raises(EvaluationStopped, match="blocked"):
        run_evaluation(cache, [pair()], transport(handler))
    assert len(calls) == 1
    cache.close()


def test_billing_is_rechecked_immediately_before_generation(tmp_path) -> None:
    cache = Cache(tmp_path / "cache.db")
    billing_checks = []
    adapter = transport(lambda request: pytest.fail("Must not generate"))

    def handler(request):
        if request.url.host == "cloudbilling.googleapis.com":
            billing_checks.append(request)
            if len(billing_checks) == 2:
                return httpx.Response(200, json={"billingEnabled": True})
        return adapter.handle_request(request)

    with pytest.raises(EvaluationStopped, match="Billing"):
        run_evaluation(cache, [pair()], httpx.MockTransport(handler))
    assert len(billing_checks) == 2
    cache.close()


def test_cli_requires_live_opt_in(monkeypatch) -> None:
    from scripts.evaluate_relevance import main

    monkeypatch.setattr("sys.argv", ["evaluate_relevance.py", "run"])
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 2


@pytest.mark.parametrize("label", [-1, 3, True, "2"])
def test_strict_label_schema(label) -> None:
    with pytest.raises(ValidationError):
        Judgment(label=label, justification="Reason", quote="")


def test_invented_quote_and_truncation_rejected() -> None:
    with pytest.raises(ValueError, match="Quote"):
        parse_judgment(
            response(), {**pair(), "abstract": "Other text"}
        )
    payload = response()
    payload["candidates"][0]["finishReason"] = "MAX_TOKENS"
    with pytest.raises(ValueError):
        parse_judgment(payload, pair())


def test_retry_info_daily_quota_and_http_date() -> None:
    assert (
        retry_delay(
            httpx.Response(
                429,
                json={
                    "error": {
                        "details": [
                            {
                                "violations": [
                                    {
                                        "quotaId": "GenerateRequestsPerDay"
                                    }
                                ],
                                "retryDelay": "200s",
                            }
                        ]
                    }
                },
            ),
            0,
        )
        == 86400
    )
    assert (
        retry_delay(
            httpx.Response(
                429,
                headers={
                    "Retry-After": "Thu, 01 Jan 1970 00:02:00 GMT"
                },
            ),
            0,
        )
        == 120
    )


def test_dedup_metadata_selection_ignores_provider_and_order(
    tmp_path,
) -> None:
    source = tmp_path / "input.csv"
    base = {
        **pair(),
        "provider": "arxiv",
        "arm": "original",
        "rank": 1,
        "pdf_urls": "[]",
    }
    rows = [base, {**base, "provider": "openalex", "abstract": ""}]
    csv_write(source, rows)
    _, pairs = read_pairs(source)
    csv_write(source, list(reversed(rows)))
    assert read_pairs(source)[1] == pairs
    assert len(pairs) == 1
    assert not pairs[0]["missing_abstract"]


def metric_manifest() -> dict:
    pairs = [pair("shared"), pair("lost"), pair("added")]
    rows = [
        {
            "claim_id": "c01",
            "identity": identity,
            "provider": "arxiv",
            "arm": arm,
            "pdf_urls": json.dumps(["https://example.org/a.pdf"])
            if identity == "added"
            else "[]",
        }
        for arm, ids in [
            ("original", ["shared", "lost"]),
            ("alternative", ["shared", "added"]),
        ]
        for identity in ids
    ]
    return {
        "model": "gemini-2.5-flash",
        "pairs": pairs,
        "rows": rows,
        "experiment": {
            "calls": [
                {
                    "provider": "arxiv",
                    "claim_id": "c01",
                    "kind": arm,
                    "status": "success",
                }
                for arm in ("original", "alternative")
            ]
        },
    }


def test_metrics_thresholds_displacement_and_partial_labels(
    tmp_path,
) -> None:
    manifest = metric_manifest()
    # Cache keys are content-based, so ensure distinct paper metadata.
    for item in manifest["pairs"]:
        item["title"] = item["identity"]
    cache = Cache(tmp_path / "cache.db")
    partial = generate_report(manifest, cache, tmp_path)
    assert partial["metrics"][0]["precision_at_10"] is None
    for item, label in zip(
        manifest["pairs"], [1, 2, 2], strict=True
    ):
        key = cache_key(item, manifest["model"])
        cache.reserve(key, 100, 100000)
        cache.finish(key, "done", {"label": label})
    result = generate_report(manifest, cache, tmp_path)
    original = result["metrics"][0]
    assert original["precision_at_10"] == 0.2
    evidence = next(
        m
        for m in result["metrics"]
        if m["provider"] == "arxiv"
        and m["arm"] == "alternative"
        and m["label_threshold"] == 2
    )
    assert evidence["precision_at_10"] == 0.1
    assert evidence["relevant_pairs_with_pdf_url"] == 1
    assert (
        result["displacement"][0]["relevant_concept_additions"] == 1
    )
    assert (
        result["displacement"][0]["relevant_original_displaced"]
        == 1
    )
    cache.close()


def test_actual_dataset_prepares_234_pairs_and_30_blinded_examples(
    tmp_path,
):
    root = Path("artifacts/arxiv-concepts-20261031")
    if not root.exists():
        pytest.skip("Local saved experiment not distributed")
    manifest = prepare(
        root / "manual_assessment.csv",
        root / "results.json",
        tmp_path / "evaluation",
    )
    assert manifest["row_count"] == 250
    assert manifest["pair_count"] == 234
    assert (
        sum(p["missing_abstract"] for p in manifest["pairs"]) == 29
    )
    sample = tmp_path / "evaluation" / "manual_validation_30.csv"
    import csv

    with sample.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 30
    assert (
        not {"provider", "rank", "arm", "query", "label"}
        & rows[0].keys()
    )
    with pytest.raises(ValueError, match="Manifest changed"):
        prepare(
            root / "manual_assessment.csv",
            root / "results.json",
            tmp_path / "evaluation",
            "gemini-3.5-flash-lite",
        )
