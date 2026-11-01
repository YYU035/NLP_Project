"""Focused offline checks for the additive Person 2 handoff."""

import asyncio
from datetime import date
from unittest.mock import AsyncMock

import httpx
import pytest

from paper_search.arxiv_fallback import quote_phrases, relaxation_plan
from paper_search.config import Settings
from paper_search.exceptions import APIClientError
from paper_search.metadata import ScientificPaper
from paper_search.query_rendering import render_query
from paper_search.retrieval import RetrievalService, RetrievalUnavailable
from paper_search.retrieval_models import ProviderPlan


def plan(provider, claim):
    """Freeze scientific concepts independently of installed local models."""
    return ProviderPlan(
        provider=provider,
        strategy="test",
        query=quote_phrases(["solar cells", "power conversion"]),
        required_phrases=["solar cells", "power conversion"],
    )


def paper(provider, identity="one"):
    return ScientificPaper(
        source=provider,
        source_id=identity,
        title="Solar cell conversion",
        doi="10.1234/test",
    )


def test_optional_first_and_required_atoms_preserved():
    queries = relaxation_plan(["TNF-α", "COVID-19 outcomes"], ["older adults"])
    assert queries[0].omitted_phrases == ("older adults",)
    assert queries[0].query == '"TNF-α" "COVID-19 outcomes"'
    wire = render_query(
        queries[1].query, "arxiv", arxiv_strategy="literal"
    ).text
    assert wire == 'all:"TNF-α" AND all:"COVID-19" AND all:outcomes'
    assert relaxation_plan(["SiC"], []) == []
    assert relaxation_plan(["solar cells"], [], max_attempts=0) == []
    with pytest.raises(ValueError):
        relaxation_plan(["solar cells"], ["solar cells"])


def test_union_provenance_exact_claim_and_failure_isolation():
    async def run():
        clients = {p: AsyncMock() for p in ("openalex", "openaire", "arxiv")}
        for p in clients:
            clients[p].search.return_value = [paper(p)]
        clients["arxiv"].search.side_effect = APIClientError("offline")
        async with RetrievalService(
            planner=plan, factory=lambda *_: clients
        ) as service:
            result = await service.retrieve("  Original claim.  ", top_k=30)
        assert len(result.papers) == 1
        assert result.original_claim == "  Original claim.  "
        assert result.papers[0].identity == "doi:10.1234/test"
        assert len(result.papers[0].provenance) == 2
        assert result.providers["arxiv"].status == "provider_failure"
        assert result.limit_scope == "per_provider"
        assert (
            result.http_attempts == 0
        )  # injected clients made no HTTP requests

    asyncio.run(run())


@pytest.mark.parametrize("enabled,expected", [(False, 1), (True, 2)])
def test_fallback_opt_in_and_dates_unchanged(enabled, expected):
    async def run():
        client = AsyncMock()
        client.search.side_effect = [[], [paper("arxiv")]]
        async with RetrievalService(
            providers=("arxiv",),
            planner=plan,
            factory=lambda *_: {"arxiv": client},
            arxiv_fallback=enabled,
        ) as service:
            result = await service.retrieve(
                "claim", from_publication_date=date(2020, 1, 1)
            )
        assert client.search.await_count == expected
        assert len(result.providers["arxiv"].queries) == expected
        for call in client.search.await_args_list:
            assert call.kwargs["from_publication_date"] == date(2020, 1, 1)
        if enabled:
            assert (
                client.search.await_args_list[1].args[0]
                == "solar cells power conversion"
            )

    asyncio.run(run())


def test_enough_candidates_stop_and_fallback_failure_retains_primary():
    async def run():
        for sufficient in (True, False):
            client = AsyncMock()
            primary = [
                ScientificPaper(source="arxiv", source_id=str(i))
                for i in range(5 if sufficient else 1)
            ]
            client.search.side_effect = [
                primary,
                APIClientError("failed fallback"),
            ]
            async with RetrievalService(
                providers=("arxiv",),
                planner=plan,
                factory=lambda *_, client=client: {"arxiv": client},
                arxiv_fallback=True,
            ) as service:
                result = await service.retrieve("claim")
            assert len(result.papers) == len(primary)
            assert client.search.await_count == (1 if sufficient else 2)
            assert result.providers["arxiv"].status == (
                "success" if sufficient else "partial_failure"
            )

    asyncio.run(run())


def test_valid_zero_distinct_from_all_failed_and_no_fallback_on_failure():
    async def run():
        client = AsyncMock()
        client.search.return_value = []
        async with RetrievalService(
            providers=("arxiv",),
            planner=plan,
            factory=lambda *_: {"arxiv": client},
            arxiv_fallback=True,
        ) as service:
            assert (await service.retrieve("claim")).papers == []
            client.search.reset_mock()
            client.search.side_effect = APIClientError("HTTP 429")
            with pytest.raises(RetrievalUnavailable) as caught:
                await service.retrieve("claim")
            assert (
                caught.value.result.providers["arxiv"].status
                == "provider_failure"
            )
            client.search.assert_awaited_once()

    asyncio.run(run())


def test_real_client_mocked_http_accounting_archive_and_redaction(tmp_path):
    async def run():
        def handler(request):
            assert request.url.params["per_page"] == "30"
            return httpx.Response(200, json={"results": []})

        async with RetrievalService(
            Settings(openalex_api_key="test-secret"),
            providers=("openalex",),
            planner=plan,
            archive=tmp_path,
            transport=httpx.MockTransport(handler),
        ) as service:
            result = await service.retrieve("claim")
        assert result.http_attempts == 1
        trace = result.providers["openalex"].queries[0]
        assert trace.status == "zero_results"
        assert trace.attempts[0].status == 200
        assert (tmp_path / trace.attempts[0].response_file).exists()
        assert "test-secret" not in result.model_dump_json()

    asyncio.run(run())


def test_arxiv_exact_wire_queries_and_http_costs():
    from paper_search.clients.arxiv import ArxivClient

    async def run():
        requests = []

        def handler(request):
            requests.append(request.url.params["search_query"])
            return httpx.Response(
                200, text='<feed xmlns="http://www.w3.org/2005/Atom"/>'
            )

        def factory(http, settings):
            client = ArxivClient(http, max_retries=0)
            client.min_interval = 0  # Mocked transport only; no live calls.
            return {"arxiv": client}

        async with RetrievalService(
            providers=("arxiv",),
            planner=plan,
            factory=factory,
            transport=httpx.MockTransport(handler),
            arxiv_fallback=True,
        ) as service:
            result = await service.retrieve(
                "claim",
                from_publication_date=date(2020, 1, 1),
                to_publication_date=date(2026, 10, 30),
            )
        dates = " AND submittedDate:[202001010000 TO 202610302359]"
        assert requests == [
            '(all:"solar cells" AND all:"power conversion")' + dates,
            "(all:solar AND all:cells AND all:power AND all:conversion)"
            + dates,
        ]
        assert result.http_attempts == 2
        assert [q.returned for q in result.providers["arxiv"].queries] == [
            0,
            0,
        ]

    asyncio.run(run())


def test_bad_fallback_annotations_fail_only_their_provider():
    async def run():
        client = AsyncMock()
        client.search.return_value = []

        def invalid(provider, claim):
            if provider == "arxiv":
                return ProviderPlan(
                    provider=provider,
                    strategy="test",
                    query='"outcomes" "outcomes"',
                    required_phrases=["outcomes"],
                    optional_phrases=["outcomes"],
                )
            return plan(provider, claim)

        async with RetrievalService(
            providers=("openalex", "arxiv"),
            planner=invalid,
            factory=lambda *_: {"openalex": client, "arxiv": AsyncMock()},
            arxiv_fallback=True,
        ) as service:
            result = await service.retrieve("claim")
        assert result.providers["arxiv"].status == "query_generation_failure"
        assert result.providers["openalex"].status == "zero_results"

    asyncio.run(run())


def test_actual_retries_count_as_http_not_extra_queries():
    from paper_search.clients.openalex import OpenAlexClient

    async def run():
        count = 0

        def handler(request):
            nonlocal count
            count += 1
            return (
                httpx.Response(429, headers={"Retry-After": "0"})
                if count == 1
                else httpx.Response(200, json={"results": []})
            )

        def factory(http, settings):
            client = OpenAlexClient(http, max_retries=1)
            client.min_interval = 0
            return {"openalex": client}

        async with RetrievalService(
            providers=("openalex",),
            planner=plan,
            factory=factory,
            transport=httpx.MockTransport(handler),
        ) as service:
            result = await service.retrieve("claim")
        assert result.http_attempts == 2
        queries = result.providers["openalex"].queries
        assert len(queries) == 1
        assert [a.status for a in queries[0].attempts] == [429, 200]

    asyncio.run(run())
