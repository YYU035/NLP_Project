"""Mocked OpenAIRE normalization, quota, factory and novelty regressions."""

import asyncio
from datetime import date

import httpx
import pytest

from paper_search.clients.openaire import (
    OpenAIREClient,
    parse_work,
    render_openaire_query,
)
from paper_search.config import Settings
from paper_search.exceptions import APIClientError
from paper_search.metadata import ScientificPaper
from paper_search.pipeline import SearchPipeline, build_clients


def test_parse_nested_metadata_and_explicit_pdf() -> None:
    paper = parse_work(
        {
            "id": "work1",
            "mainTitle": "CO<sub>2</sub> &amp; climate",
            "descriptions": [
                "",
                "<jats:p>TNF-α does not increase.</jats:p>",
            ],
            "publicationDate": "2024-05-06",
            "authors": [
                {"fullName": "Ada A"},
                {"name": "Ben", "surname": "B"},
                None,
            ],
            "pids": [{"scheme": "doi", "value": "10.123/a"}],
            "instances": [
                {
                    "urls": [
                        "https://example.org/a.pdf",
                        "https://doi.org/10.123/a",
                        "bad",
                    ]
                }
            ],
        }
    )
    assert paper.title == "CO2 & climate"
    assert paper.abstract == "TNF-α does not increase."
    assert paper.authors == ["Ada A", "Ben B"]
    assert paper.year == 2024
    assert paper.doi == "https://doi.org/10.123/a"
    assert len(paper.pdf_urls) == 1
    assert parse_work({"id": "x"}).pdf_urls == []
    assert parse_work({"id": "x"}).abstract is None
    with pytest.raises(ValueError):
        parse_work({"mainTitle": "Missing identifier"})


def test_arxiv_pid_preserved_for_shared_deduplication() -> None:
    from paper_search.deduplication import arxiv_ids

    paper = parse_work(
        {
            "id": "graph-id",
            "pids": [{"scheme": "arxiv", "value": "1706.03762v2"}],
        }
    )
    assert arxiv_ids(paper) == {"1706.03762"}
    assert (
        not paper.pdf_urls
    )  # Do not invent PDF access from an identifier.


def test_claim_negation_does_not_become_boolean_exclusion() -> None:
    assert render_openaire_query(
        "TNF-α does not increase risk"
    ) == ("TNF-α does not increase risk")
    assert render_openaire_query("DNA is NOT increased") == (
        '"DNA is NOT increased"'
    )


def test_search_filters_and_free_optional_authentication() -> None:
    def handler(request):
        assert (
            request.url.params["search"] == "graph neural networks"
        )
        assert request.url.params["type"] == "publication"
        assert (
            request.url.params["fromPublicationDate"]
            == "2020-01-01"
        )
        assert (
            request.url.params["toPublicationDate"] == "2026-10-30"
        )
        assert request.url.params["pageSize"] == "10"
        assert request.headers["Authorization"] == "Bearer token"
        return httpx.Response(
            200, json={"header": {"numFound": 0}, "results": []}
        )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as h:
            assert (
                await OpenAIREClient(h, api_key="token").search(
                    "graph neural networks",
                    max_results=10,
                    from_publication_date=date(2020, 1, 1),
                    to_publication_date=date(2026, 10, 30),
                )
                == []
            )

    asyncio.run(run())


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"header": {"numFound": 1}, "results": []},
        {
            "header": {"numFound": 1},
            "results": [{"title": "No ID"}],
        },
    ],
)
def test_invalid_envelopes_are_failures_not_empty_results(
    payload,
) -> None:
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=payload)
            )
        ) as h:
            with pytest.raises(APIClientError):
                await OpenAIREClient(h).search("claim")

    asyncio.run(run())


def test_hourly_budget_and_retry_after_use_shared_transport() -> (
    None
):
    current = [0.0]
    starts = []

    async def sleep(seconds):
        current[0] += seconds

    def handler(request):
        starts.append(current[0])
        if len(starts) == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(
            200, json={"header": {"numFound": 0}, "results": []}
        )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as h:
            client = OpenAIREClient(
                h, clock=lambda: current[0], sleep=sleep
            )
            for _ in range(60):
                await client.search("claim")

    asyncio.run(run())
    assert len(starts) == 61
    assert starts[1] >= 7
    assert starts[60] >= 3600


def test_opt_in_pipeline_does_not_change_old_defaults() -> None:
    assert not Settings().openaire_enabled

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"header": {"numFound": 0}, "results": []},
                )
            )
        ) as http:
            settings = Settings(
                openaire_enabled=True,
                openalex_enabled=False,
                arxiv_enabled=False,
                semantic_scholar_enabled=False,
            )
            assert list(build_clients(http, settings)) == [
                "openaire"
            ]
            assert (
                await SearchPipeline(
                    settings, http_client=http
                ).run("claim")
                == []
            )

    asyncio.run(run())


