"""Offline contract tests for additional providers and shared transport."""

import asyncio
from datetime import date

import httpx
import pytest

from paper_search.clients.arxiv import ArxivClient
from paper_search.clients.openalex import OpenAlexClient
from paper_search.clients.semantic_scholar import SemanticScholarClient
from paper_search.exceptions import APIClientError
from paper_search.metadata import PaperMetadata, ScientificPaper

FEED = '''<feed xmlns="http://www.w3.org/2005/Atom"
 xmlns:arxiv="http://arxiv.org/schemas/atom">
 <entry>
 <id>http://arxiv.org/abs/2401.12345v1</id>
 <title>Memory and\n exercise</title><summary>An abstract.</summary>
 <published>2024-01-23T12:00:00Z</published>
 <author><name>Alice</name></author><author><name>Bob</name></author>
 <arxiv:doi>10.1234/example</arxiv:doi>
 <link rel="alternate" href="https://arxiv.org/abs/2401.12345v1"/>
 <link title="pdf" href="https://arxiv.org/pdf/2401.12345v1"/>
 </entry></feed>'''


class VirtualClock:
    """Advance simulated time instead of sleeping during rate-limit tests."""

    def __init__(self) -> None:
        self.now = 0.0
        self.delays: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        """Record requested sleep and advance the virtual clock."""
        self.delays.append(delay)
        self.now += delay


def test_arxiv_search_and_normalization() -> None:
    """Atom metadata, claim tokens, date bounds, and limits are mapped."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "export.arxiv.org"
        assert request.url.params["search_query"] == (
            '(all:Exercise AND all:memory) AND '
            'submittedDate:[202001010000 TO 202512312359]'
        )
        assert request.url.params["max_results"] == "2"
        assert request.url.params["start"] == "0"
        assert "x-api-key" not in request.headers
        assert "Authorization" not in request.headers
        return httpx.Response(200, text=FEED)

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http:
            papers = await ArxivClient(http).search(
                "Exercise improves memory", max_results=2,
                from_publication_date=date(2020, 1, 1),
                to_publication_date=date(2025, 12, 31),
            )
            assert not http.is_closed
        paper = papers[0]
        assert isinstance(paper, ScientificPaper)
        assert isinstance(paper, PaperMetadata)
        assert paper.source == "arxiv"
        assert paper.source_id == "http://arxiv.org/abs/2401.12345v1"
        assert paper.title == "Memory and exercise"
        assert paper.abstract == "An abstract."
        assert paper.authors == ["Alice", "Bob"]
        assert paper.doi == "https://doi.org/10.1234/example"
        assert paper.publication_date == date(2024, 1, 23)
        assert paper.year == 2024
        assert str(paper.landing_url).startswith("https://arxiv.org/abs/")
        assert str(paper.pdf_url).startswith("https://arxiv.org/pdf/")
    asyncio.run(run())


@pytest.mark.parametrize("start,end,expected", [
    (date(2020, 1, 1), None, "202001010000 TO 999912312359"),
    (None, date(2024, 1, 1), "000101010000 TO 202401012359"),
])
def test_arxiv_open_date_bounds(start, end, expected: str) -> None:
    """One-sided ranges use the documented submittedDate range syntax."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert expected in request.url.params["search_query"]
        return httpx.Response(200, text=FEED)

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http:
            await ArxivClient(http).search(
                "memory", from_publication_date=start,
                to_publication_date=end,
            )
    asyncio.run(run())


def test_semantic_scholar_search_and_normalization() -> None:
    """Graph parameters, keyed authentication, DOI and PDF map correctly."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/graph/v1/paper/search"
        assert request.url.params["query"] == "memory based learning"
        assert request.url.params["limit"] == "4"
        assert request.url.params["publicationDateOrYear"] == (
            "2020-01-01:2025-12-31"
        )
        assert "abstract" in request.url.params["fields"]
        assert "externalIds" in request.url.params["fields"]
        assert request.headers["x-api-key"] == "s2-secret"
        assert "api_key" not in request.url.params
        return httpx.Response(200, json={"data": [{
            "paperId": "abc123", "title": "A paper", "abstract": "Text",
            "authors": [{"name": "Alice"}, None, {}],
            "publicationDate": "2023-02-05", "year": 2023,
            "externalIds": {"DOI": "10.1234/example"},
            "url": "https://www.semanticscholar.org/paper/abc123",
            "openAccessPdf": {"url": "https://example.org/paper.pdf"},
        }]})

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http:
            papers = await SemanticScholarClient(
                http, api_key="s2-secret"
            ).search(
                "memory-based learning", max_results=4,
                from_publication_date=date(2020, 1, 1),
                to_publication_date=date(2025, 12, 31),
            )
        paper = papers[0]
        assert isinstance(paper, ScientificPaper)
        assert paper.source == "semantic_scholar"
        assert paper.source_id == "abc123"
        assert paper.title == "A paper"
        assert paper.abstract == "Text"
        assert paper.authors == ["Alice"]
        assert paper.doi == "https://doi.org/10.1234/example"
        assert paper.publication_date == date(2023, 2, 5)
        assert paper.year == 2023
        assert len(paper.pdf_urls) == 1
    asyncio.run(run())


@pytest.mark.parametrize("provider,body", [
    (ArxivClient, '<feed xmlns="http://www.w3.org/2005/Atom">'
     '<entry/><entry><id>http://arxiv.org/abs/1</id></entry></feed>'),
    (SemanticScholarClient, '{"data":[null,{}, {"paperId":"1"}]}'),
])
def test_missing_fields(provider, body: str) -> None:
    """Malformed records are skipped and absent optional data stays unknown."""
    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, text=body)
        )) as http:
            papers = await provider(http).search("claim")
        assert len(papers) == 1
        assert papers[0].title is None
        assert papers[0].authors == []
        assert papers[0].abstract is None
        assert papers[0].doi is None
        assert papers[0].publication_date is None
        assert papers[0].pdf_urls == []
    asyncio.run(run())


@pytest.mark.parametrize("provider,body", [
    (ArxivClient, "not XML"), (ArxivClient, "<html/>"),
    (ArxivClient, '<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
     '<id>http://arxiv.org/api/errors#bad_query</id></entry></feed>'),
    (SemanticScholarClient, "not JSON"),
    (SemanticScholarClient, '{"data":null}'),
])
def test_provider_response_errors(provider, body: str) -> None:
    """Bad payloads and arXiv error feeds raise domain errors."""
    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, text=body)
        )) as http:
            with pytest.raises(APIClientError):
                await provider(http).search("claim")
    asyncio.run(run())


@pytest.mark.parametrize("provider,body,interval", [
    (ArxivClient, FEED, 3),
    (SemanticScholarClient, '{"data":[]}', 1),
    (OpenAlexClient, '{"results":[]}', 0.1),
])
@pytest.mark.parametrize("failure", [429, 503, "timeout"])
def test_shared_retries_and_provider_pacing(
    provider, body: str, interval: float, failure
) -> None:
    """All adapters share retries while retaining provider request spacing."""
    clock = VirtualClock()
    starts = []

    def handler(request: httpx.Request) -> httpx.Response:
        starts.append(clock.now)
        if len(starts) == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("timeout", request=request)
            return httpx.Response(failure, headers={"Retry-After": "2"})
        return httpx.Response(200, text=body)

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http:
            client = provider(http, sleep=clock.sleep, clock=clock)
            await client.search("claim")
            await client.search("claim")
    asyncio.run(run())
    assert len(starts) == 3
    assert starts[1] >= max(interval, 1 if failure == "timeout" else 2)
    assert starts[2] - starts[1] >= interval - 1e-9


@pytest.mark.parametrize("provider", [ArxivClient, SemanticScholarClient])
@pytest.mark.parametrize("status,calls_expected", [(401, 1), (429, 3)])
def test_provider_failure_is_bounded(provider, status, calls_expected) -> None:
    """Authentication failures stop immediately; rate-limit retries stop."""
    calls = []
    clock = VirtualClock()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status)

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http:
            with pytest.raises(APIClientError):
                await provider(
                    http, sleep=clock.sleep, clock=clock
                ).search("claim")
    asyncio.run(run())
    assert len(calls) == calls_expected


@pytest.mark.parametrize("provider", [ArxivClient, SemanticScholarClient])
@pytest.mark.parametrize("options", [
    {"query": " "}, {"query": "claim", "max_results": 101},
    {"query": "claim", "from_publication_date": date(2025, 1, 1),
     "to_publication_date": date(2024, 1, 1)},
])
def test_shared_validation_precedes_io(provider, options: dict) -> None:
    """Both adapters reject invalid caller input before HTTP access."""
    def handler(_: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid requests must not reach the transport")

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http:
            with pytest.raises(ValueError):
                await provider(http).search(**options)
    asyncio.run(run())


@pytest.mark.parametrize("start,end,expected", [
    (date(2020, 1, 1), None, "2020-01-01:"),
    (None, date(2024, 1, 1), ":2024-01-01"),
    (None, None, None),
])
def test_semantic_scholar_keyless_and_open_bounds(
    start: date | None, end: date | None, expected: str | None
) -> None:
    """Anonymous calls omit credentials and preserve optional filter bounds."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert "x-api-key" not in request.headers
        assert request.url.params.get("publicationDateOrYear") == expected
        return httpx.Response(200, json={"data": []})

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http:
            assert await SemanticScholarClient(http).search(
                "claim", from_publication_date=start, to_publication_date=end
            ) == []
    asyncio.run(run())


def test_arxiv_serializes_concurrent_requests() -> None:
    """Even slow responses use only one concurrent arXiv connection."""
    active = 0
    maximum = 0
    clock = VirtualClock()

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0)
        active -= 1
        return httpx.Response(200, text=FEED)

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http:
            client = ArxivClient(http, sleep=clock.sleep, clock=clock)
            await asyncio.gather(client.search("one"), client.search("two"))
    asyncio.run(run())
    assert maximum == 1
    assert clock.delays == [3]
