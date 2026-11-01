"""Mocked multi-provider orchestration and configuration tests."""

import asyncio
from datetime import date
from unittest.mock import AsyncMock

import httpx
import pytest

from paper_search.config import Settings
from paper_search.exceptions import APIClientError
from paper_search.metadata import ScientificPaper
from paper_search.pipeline import SearchPipeline, build_clients


def test_all_providers_and_failure_isolation(caplog) -> None:
    """A failed provider does not discard successful provider results."""
    clients = {}
    for source in ("openalex", "arxiv", "semantic_scholar"):
        clients[source] = AsyncMock()
        clients[source].search.return_value = [
            ScientificPaper(source=source, source_id="same-doi-paper")
        ]
    clients["arxiv"].search.side_effect = APIClientError("offline")
    papers = asyncio.run(SearchPipeline(clients=clients).run(
        "a claim", max_results=3, from_publication_date=date(2020, 1, 1)
    ))
    assert [paper.source for paper in papers] == [
        "openalex", "semantic_scholar"
    ]
    for client in clients.values():
        client.search.assert_awaited_once_with(
            query="a claim", max_results=3,
            from_publication_date=date(2020, 1, 1), to_publication_date=None,
        )
    assert "arxiv" in caplog.text


def test_disabled_providers_are_not_called() -> None:
    """Configuration switches govern even injected provider mappings."""
    client = AsyncMock()
    client.search.return_value = []
    disabled = AsyncMock()
    pipeline = SearchPipeline(
        Settings(arxiv_enabled=False, semantic_scholar_enabled=False),
        clients={"openalex": client, "arxiv": disabled},
    )
    assert asyncio.run(pipeline.run("claim")) == []
    disabled.search.assert_not_called()


def test_all_failed_is_distinct_from_no_matches() -> None:
    """Complete provider outage raises; a successful empty search is valid."""
    clients = {name: AsyncMock() for name in (
        "openalex", "arxiv", "semantic_scholar"
    )}
    for client in clients.values():
        client.search.side_effect = APIClientError("failed")
    pipeline = SearchPipeline(clients=clients)
    with pytest.raises(APIClientError, match="All enabled"):
        asyncio.run(pipeline.run("claim"))
    clients["openalex"].search.side_effect = None
    clients["openalex"].search.return_value = []
    assert asyncio.run(pipeline.run("claim")) == []


def test_configuration_environment(monkeypatch) -> None:
    """Provider flags and keys load explicitly without disclosing secrets."""
    monkeypatch.setenv("PAPER_SEARCH_ARXIV_ENABLED", "false")
    monkeypatch.setenv("PAPER_SEARCH_OPENALEX_ENABLED", "true")
    monkeypatch.setenv("PAPER_SEARCH_SEMANTIC_SCHOLAR_ENABLED", "0")
    monkeypatch.setenv("OPENALEX_API_KEY", "openalex-secret")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "s2-secret")
    settings = Settings.from_environment()
    assert settings.openalex_enabled
    assert not settings.arxiv_enabled
    assert not settings.semantic_scholar_enabled
    assert settings.openalex_api_key.get_secret_value() == "openalex-secret"
    assert settings.semantic_scholar_api_key.get_secret_value() == "s2-secret"
    assert "openalex-secret" not in repr(settings)
    assert "s2-secret" not in repr(settings)


def test_factory_authentication_and_enabled_providers() -> None:
    """Factory passes each key only to its intended provider."""
    async def run() -> None:
        async with httpx.AsyncClient() as http:
            settings = Settings(
                openalex_api_key="oa", semantic_scholar_api_key="s2"
            )
            clients = build_clients(http, settings)
            assert clients["openalex"].auth_headers() == {
                "Authorization": "Bearer oa"
            }
            assert clients["semantic_scholar"].auth_headers() == {
                "x-api-key": "s2"
            }
            assert clients["arxiv"].auth_headers() == {}
            assert build_clients(http, Settings(
                arxiv_enabled=False, semantic_scholar_enabled=False
            )).keys() == {"openalex"}
    asyncio.run(run())


def test_no_providers_and_invalid_claim_fail_without_io() -> None:
    """Invalid requests and disabled configurations never contact providers."""
    pipeline = SearchPipeline(Settings(
        openalex_enabled=False, arxiv_enabled=False,
        semantic_scholar_enabled=False,
    ), clients={})
    with pytest.raises(ValueError, match="At least one"):
        asyncio.run(pipeline.run("claim"))
    with pytest.raises(ValueError):
        asyncio.run(pipeline.run(" "))


def test_full_search_with_mock_http() -> None:
    """Factory-built clients share transport but decode their own formats."""
    hosts = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        if request.url.host == "api.openalex.org":
            return httpx.Response(200, json={"results": [{"id": "W1"}]})
        if request.url.host == "api.semanticscholar.org":
            return httpx.Response(200, json={"data": [{"paperId": "S1"}]})
        return httpx.Response(200, text=(
            '<feed xmlns="http://www.w3.org/2005/Atom">'
            '<entry><id>http://arxiv.org/abs/1</id></entry></feed>'
        ))

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http:
            papers = await SearchPipeline(http_client=http).run("claim")
            assert not http.is_closed
        assert [paper.source for paper in papers] == [
            "openalex", "arxiv", "semantic_scholar"
        ]
    asyncio.run(run())
    assert len(hosts) == 3
