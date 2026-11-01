"""Mocked OpenAlex contract tests; no live network access is required."""

import asyncio
from datetime import UTC, date, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest
from pydantic import ValidationError

from paper_search.clients.openalex import (
    OpenAlexClient,
    parse_work,
    reconstruct_abstract,
)
from paper_search.exceptions import APIClientError
from paper_search.metadata import PaperMetadata


@pytest.fixture(autouse=True)
def disable_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test backoff separately from pacing, which has shared clock tests."""
    monkeypatch.setattr(OpenAlexClient, "min_interval", 0)


def test_search_parameters_and_metadata() -> None:
    """Claims, date filters, authentication, and metadata map correctly."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/works"
        assert request.url.params["search"] == "Exercise improves memory"
        assert request.url.params["per_page"] == "2"
        assert request.url.params["filter"] == (
            "from_publication_date:2020-01-01,"
            "to_publication_date:2025-12-31"
        )
        assert "api_key" not in request.url.params
        assert request.headers["Authorization"] == "Bearer secret"
        assert request.extensions["timeout"]["read"] == 12
        return httpx.Response(200, json={"results": [{
            "id": "https://openalex.org/W1",
            "title": "Memory and exercise",
            "doi": "https://doi.org/10.1234/example",
            "publication_date": "2023-04-02",
            "publication_year": 2023,
            "abstract_inverted_index": {
                "memory": [2], "Exercise": [0], "improves": [1],
            },
            "authorships": [
                {"author": {"display_name": "Alice"}},
                {"author": None, "raw_author_name": "Bob"},
                {},
            ],
            "primary_location": {
                "landing_page_url": "https://example.org/paper",
                "pdf_url": "https://example.org/paper.pdf",
            },
            "best_oa_location": {
                "pdf_url": "https://repo.org/paper.pdf",
            },
            "locations": [
                {"pdf_url": "https://repo.org/paper.pdf"},
                {"pdf_url": "https://other.org/paper.pdf"},
            ],
        }]})

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http:
            papers = await OpenAlexClient(
                http, api_key="secret", timeout_seconds=12
            ).search(
                "  Exercise improves memory  ", max_results=2,
                from_publication_date=date(2020, 1, 1),
                to_publication_date=date(2025, 12, 31),
            )
            assert not http.is_closed
        assert len(papers) == 1
        paper = papers[0]
        assert isinstance(paper, PaperMetadata)
        assert paper.title == "Memory and exercise"
        assert paper.abstract == "Exercise improves memory"
        assert paper.authors == ["Alice", "Bob"]
        assert paper.doi == "https://doi.org/10.1234/example"
        assert paper.publication_date == date(2023, 4, 2)
        assert paper.year == 2023
        assert paper.source == "openalex"
        assert paper.source_id == "https://openalex.org/W1"
        assert str(paper.landing_url) == "https://example.org/paper"
        assert str(paper.pdf_url) == "https://repo.org/paper.pdf"
        assert len(paper.pdf_urls) == 3

    asyncio.run(run())


@pytest.mark.parametrize("index,expected", [
    ({"the": [0, 2], "cat": [1]}, "the cat the"),
    (None, None), ({}, None), ([], None),
    ({"word": [-1]}, None), ({"word": [True]}, None),
    ({"word": [999999999]}, None),
    ({"a": [0], "b": [0]}, None), ({"a": "0"}, None),
])
def test_abstract_reconstruction(index: object, expected: str | None) -> None:
    """Reconstruction handles repeated words and malformed input safely."""
    assert reconstruct_abstract(index) == expected


def test_missing_and_malformed_optional_fields() -> None:
    """Absent information stays unknown; invalid URLs are not emitted."""
    paper = parse_work({
        "id": "https://openalex.org/W1", "title": None,
        "authorships": None, "primary_location": None,
        "best_oa_location": [], "publication_date": "bad date",
        "publication_year": "bad year",
        "locations": [None, {"pdf_url": "javascript:alert(1)"}],
    })
    assert paper.title is None
    assert paper.abstract is None
    assert paper.authors == []
    assert paper.doi is None
    assert paper.publication_date is None
    assert paper.year is None
    assert paper.pdf_urls == []
    assert paper.pdf_url is None
    assert str(paper.landing_url) == "https://openalex.org/W1"


@pytest.mark.parametrize("payload,expected_count", [
    ({"results": []}, 0),
    ({"results": [None, {}, {"id": "W1"}]}, 1),
    ({"results": [{"id": "W1"}, {"id": "W2"}]}, 1),
])
def test_empty_malformed_and_limited_results(
    payload: dict, expected_count: int
) -> None:
    """Skip unidentifiable records and enforce the caller's result cap."""
    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=payload)
        )) as http:
            # Use a larger cap for the malformed-record fixture.
            limit = 3 if len(payload["results"]) == 3 else 1
            assert len(await OpenAlexClient(http).search(
                "claim", max_results=limit
            )) == expected_count
    asyncio.run(run())


@pytest.mark.parametrize("options", [
    {"query": " "}, {"query": "claim", "max_results": 0},
    {"query": "claim", "max_results": 101},
    {"query": "claim", "max_results": True},
    {"query": "claim", "from_publication_date": "invalid"},
    {"query": "claim", "from_publication_date": date(2025, 1, 1),
     "to_publication_date": date(2024, 1, 1)},
])
def test_invalid_input_never_sends_request(options: dict) -> None:
    """Invalid caller input fails before network I/O."""
    def handler(_: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid input must not reach HTTP transport")

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http:
            with pytest.raises((ValueError, ValidationError)):
                await OpenAlexClient(http).search(**options)
    asyncio.run(run())


@pytest.mark.parametrize("failure", [429, 500, "timeout", "connection"])
def test_transient_errors_retry_then_succeed(failure: int | str) -> None:
    """Retry transient HTTP and transport failures with bounded delays."""
    calls = []
    delays = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("timed out", request=request)
            if failure == "connection":
                raise httpx.ConnectError("offline", request=request)
            return httpx.Response(failure, headers={"Retry-After": "3"})
        return httpx.Response(200, json={"results": []})

    async def sleep(delay: float) -> None:
        delays.append(delay)

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http:
            papers = await OpenAlexClient(http, sleep=sleep).search("claim")
            assert papers == []
    asyncio.run(run())
    assert len(calls) == 2
    assert delays == ([3] if isinstance(failure, int) else [1])


@pytest.mark.parametrize("status,retries,expected_calls", [
    (400, 2, 1), (401, 2, 1), (403, 2, 1),
    (429, 2, 3), (503, 2, 3), (429, 0, 1),
])
def test_http_errors_are_bounded(
    status: int, retries: int, expected_calls: int
) -> None:
    """Permanent errors fail immediately; transient retries are finite."""
    calls = []
    delays = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status)

    async def sleep(delay: float) -> None:
        delays.append(delay)

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http:
            with pytest.raises(APIClientError, match=f"HTTP {status}"):
                await OpenAlexClient(
                    http, max_retries=retries, sleep=sleep
                ).search("claim")
    asyncio.run(run())
    assert len(calls) == expected_calls
    assert delays == ([1, 2] if expected_calls == 3 else [])


@pytest.mark.parametrize("body", [b"not json", b"{}", b'{"results":null}'])
def test_invalid_response_envelopes(body: bytes) -> None:
    """Bad API payloads become domain errors rather than empty successes."""
    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=body)
        )) as http:
            with pytest.raises(APIClientError):
                await OpenAlexClient(http).search("claim")
    asyncio.run(run())


@pytest.mark.parametrize("retry_after", [
    "120", format_datetime(datetime.now(UTC) + timedelta(hours=1)),
])
def test_long_retry_after_fails_without_sleep(retry_after: str) -> None:
    """Never retry earlier than a server-requested delay above the cap."""
    async def sleep(_: float) -> None:
        pytest.fail("Long server delays must not sleep or retry")

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _: httpx.Response(
                429, headers={"Retry-After": retry_after}
            )
        )) as http:
            with pytest.raises(APIClientError, match="delay"):
                await OpenAlexClient(http, sleep=sleep).search("claim")
    asyncio.run(run())


def test_timeout_exhaustion_preserves_cause() -> None:
    """Transport errors are wrapped after the configured attempts."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http:
            with pytest.raises(APIClientError) as error:
                await OpenAlexClient(http, max_retries=0).search("claim")
            assert isinstance(error.value.__cause__, httpx.ReadTimeout)
    asyncio.run(run())
