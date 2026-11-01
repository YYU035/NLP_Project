"""Small, read-only checks against real APIs; no PDFs are fetched."""

import asyncio
import json
import os
import time
from typing import Any
from xml.etree import ElementTree as ET

import httpx
import pytest

from paper_search.clients.arxiv import ArxivClient
from paper_search.clients.openalex import OpenAlexClient
from paper_search.clients.semantic_scholar import SemanticScholarClient
from paper_search.exceptions import APIClientError
from paper_search.metadata import ScientificPaper
from paper_search.query_processing import SearchRequest

pytestmark = pytest.mark.live
PROVIDERS = {
    "openalex": (OpenAlexClient, "OPENALEX_API_KEY", "Authorization"),
    "arxiv": (ArxivClient, None, None),
    "semantic_scholar": (
        SemanticScholarClient, "SEMANTIC_SCHOLAR_API_KEY", "x-api-key"
    ),
}


def _raw_count(response: httpx.Response, provider: str) -> int | None:
    """Count returned entries so dropped records cannot masquerade as zero."""
    try:
        if provider == "arxiv":
            root = ET.fromstring(response.content)
            return len(root.findall("{http://www.w3.org/2005/Atom}entry"))
        payload = response.json()
        if provider == "semantic_scholar" and payload == {
            "total": 0, "offset": 0
        }:
            return 0
        key = "results" if provider == "openalex" else "data"
        entries = payload.get(key) if isinstance(payload, dict) else None
        return len(entries) if isinstance(entries, list) else None
    except (ValueError, ET.ParseError):
        return None


def _completeness(papers: list[ScientificPaper]) -> dict[str, Any]:
    """Report observed availability without requiring optional metadata."""
    fields = (
        "title", "authors", "doi", "publication_date", "year",
        "landing_url", "abstract", "pdf_urls",
    )
    counts = {
        field: sum(bool(getattr(paper, field)) for paper in papers)
        for field in fields
    }
    return {
        "present_counts": counts,
        "present_percent": {
            field: round(100 * count / len(papers), 1) if papers else None
            for field, count in counts.items()
        },
        "core_complete_count": sum(
            all(getattr(paper, field) for field in (
                "title", "authors", "publication_date", "doi", "landing_url"
            )) for paper in papers
        ),
    }


@pytest.mark.parametrize("provider", list(PROVIDERS))
def test_live_provider_search(
    provider: str, pytestconfig: pytest.Config, live_report: dict[str, Any]
) -> None:
    """Check real auth, retry observations, response parsing, and metadata."""
    request = SearchRequest(
        query=pytestconfig.getoption("--live-claim"),
        max_results=pytestconfig.getoption("--live-limit"),
    )
    client_type, key_variable, auth_header = PROVIDERS[provider]
    key = os.getenv(key_variable) or None if key_variable else None
    observations: dict[str, Any] = {
        "status": "running",
        "authentication": "key_supplied" if key else (
            "not_required" if provider == "arxiv" else "anonymous"
        ),
        "auth_header_check": None,
        "attempts": 0, "http_statuses": [], "rate_limit_headers": [],
        "sleep_seconds": [], "raw_result_count": None,
        "result_count": None, "metadata": None,
        "rate_limit_handling": "not_exercised_no_429_observed",
    }
    live_report["providers"][provider] = observations
    started = time.monotonic()

    async def on_request(http_request: httpx.Request) -> None:
        # A strict allowlist guarantees this test never follows paper/PDF URLs.
        allowed = httpx.URL(client_type.endpoint)
        assert (http_request.url.scheme, http_request.url.host,
                http_request.url.path) == (
                    allowed.scheme, allowed.host, allowed.path
                ), "Only the provider search endpoint may be requested"
        expected = f"Bearer {key}" if provider == "openalex" and key else key
        actual = http_request.headers.get(auth_header) if auth_header else None
        auth_ok = actual == expected
        if provider == "arxiv":
            auth_ok = not any(name in http_request.headers for name in (
                "Authorization", "x-api-key"
            ))
        observations["auth_header_check"] = auth_ok
        assert auth_ok, "Authentication header presence/value is incorrect"
        observations["attempts"] += 1
        assert observations["attempts"] <= 3, "Retry budget was exceeded"

    async def on_response(response: httpx.Response) -> None:
        observations["http_statuses"].append(response.status_code)
        observations["rate_limit_headers"].append({
            name: response.headers[name]
            for name in (
                "retry-after", "x-ratelimit-limit", "x-ratelimit-remaining",
                "x-ratelimit-reset",
            ) if name in response.headers
        })
        await response.aread()
        if response.is_success:
            observations["raw_result_count"] = _raw_count(response, provider)

    async def sleep(delay: float) -> None:
        observations["sleep_seconds"].append(delay)
        await asyncio.sleep(delay)

    async def run() -> list[ScientificPaper]:
        async with httpx.AsyncClient(event_hooks={
            "request": [on_request], "response": [on_response],
        }) as http:
            client = client_type(
                http, api_key=key, timeout_seconds=15,
                max_retries=2, max_retry_delay=10, sleep=sleep,
            )
            return await client.search(request.query,
                                       max_results=request.max_results)

    try:
        papers = asyncio.run(run())
        observations["result_count"] = len(papers)
        observations["metadata"] = _completeness(papers)
        assert len(papers) <= request.max_results
        assert all(isinstance(paper, ScientificPaper) for paper in papers)
        assert all(paper.source == provider and paper.source_id
                   for paper in papers)
        raw_count = observations["raw_result_count"]
        assert raw_count is not None, "No valid response envelope was observed"
        assert not (raw_count and not papers), (
            "Provider returned records, but parsing dropped all records"
        )
        observations["status"] = "success" if papers else "zero_results"
        observations["dropped_record_count"] = max(
            0, min(raw_count, request.max_results) - len(papers)
        )
    except APIClientError as exc:
        statuses = observations["http_statuses"]
        observations["status"] = (
            "authentication_failure" if any(s in (401, 403) for s in statuses)
            else "rate_limited" if 429 in statuses
            else "provider_failure"
        )
        # Domain messages omit credentials and URLs. Never print the chained
        # httpx exception, whose request URL could contain sensitive input.
        observations["error"] = str(exc)
        observations["cause_type"] = (
            type(exc.__cause__).__name__ if exc.__cause__ else None
        )
        pytest.fail(
            f"{provider}: {observations['status']}: {exc}", pytrace=False
        )
    except Exception:
        observations["status"] = "test_or_parsing_failure"
        raise
    finally:
        observations["elapsed_seconds"] = round(time.monotonic() - started, 2)
        if 429 in observations["http_statuses"]:
            observations["rate_limit_handling"] = (
                "recovered_after_429" if observations["status"] in (
                    "success", "zero_results"
                ) else "bounded_failure_after_429"
            )
        print(f"\n{provider}: {json.dumps(observations, sort_keys=True)}")
