"""Provider rendering contracts, including explicit unsupported features."""

import asyncio

import httpx
import pytest

from paper_search.clients.arxiv import ArxivClient
from paper_search.clients.openalex import OpenAlexClient
from paper_search.clients.semantic_scholar import SemanticScholarClient
from paper_search.query_rendering import render_query


def test_arxiv_keeps_scientific_atoms_and_phrase_boundaries() -> None:
    query = '"TNF-α expression" does not change at -5 °C or 1e-3 mg/kg.'
    result = render_query(query, "arxiv", arxiv_strategy="literal")
    assert result.original == query
    assert result.text == (
        'all:"TNF-α expression" AND all:does AND all:not AND all:change '
        'AND all:at AND all:"-5" AND all:"°C" AND all:or '
        'AND all:"1e-3" AND all:"mg/kg"'
    )
    assert "ANDNOT" not in result.text


def test_openalex_phrases_and_literal_negation() -> None:
    result = render_query('“DNA repair” is NOT increased in HER2+ cells.',
                          "openalex")
    assert result.text == '"DNA repair" is "NOT" increased in HER2+ cells.'
    assert render_query("'DNA repair' in T cells", "openalex").text == (
        '"DNA repair" in T cells'
    )


def test_semantic_scholar_reports_loss_and_retains_numbers() -> None:
    query = '"TNF-α expression" at -5 °C and 1e-3 mg/kg in COVID-19.'
    result = render_query(query, "semantic_scholar")
    assert result.original == query
    assert result.text == (
        "TNF α expression at -5 °C and 1e-3 mg/kg in COVID 19."
    )
    assert "phrase_boundaries_not_supported" in result.warnings
    assert "hyphenated_terms_rendered_as_words" in result.warnings
    assert render_query("1e-3.", "semantic_scholar").text == "1e-3."
    assert render_query("10-20 µg", "semantic_scholar").text == "10-20 µg"


@pytest.mark.parametrize("client_type,provider,param,response", [
    (ArxivClient, "arxiv", "search_query",
     '<feed xmlns="http://www.w3.org/2005/Atom"/>'),
    (OpenAlexClient, "openalex", "search", '{"results":[]}'),
    (SemanticScholarClient, "semantic_scholar", "query", '{"data":[]}'),
])
def test_clients_use_renderer_without_signature_changes(
    client_type, provider: str, param: str, response: str
) -> None:
    """Verify exact outgoing text with mocked HTTP transport."""
    query = '"TNF-α expression" does not change at -5 °C.'

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params[param] == render_query(query, provider).text
        return httpx.Response(200, text=response)

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http:
            assert await client_type(http).search(query) == []
    asyncio.run(run())
