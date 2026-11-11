"""Regression tests for auditable backend selection, not relevance claims."""

import asyncio

import httpx
import pytest

from paper_search.arxiv_rendering import select_discovery_concepts
from paper_search.clients.arxiv import ArxivClient
from paper_search.linguistic import Concept, ExtractionResult
from paper_search.query_processing import DeterministicQueryGenerator
from paper_search.query_rendering import render_query


class FixedExtractor:
    """Inject source spans without depending on installed model weights."""

    def __init__(self, phrases: list[str], status: str = "ok") -> None:
        self.phrases = phrases
        self.status = status

    def extract(self, claim: str) -> ExtractionResult:
        """Build nonoverlapping exact source spans used by construction."""
        return ExtractionResult(
            status=self.status,
            concepts=[Concept(text=p, start=claim.index(p),
                              end=claim.index(p) + len(p), kind="noun_phrase")
                      for p in self.phrases],
        )


@pytest.mark.parametrize("claim,phrases,expected,omitted", [
    (
        "TNF-α does not increase COVID-19 risk in older adults.",
        ["TNF-α", "COVID-19 risk", "older adults"],
        ["TNF-α", "COVID-19 risk"], "does not increase",
    ),
    (
        "Lithium-ion batteries retain 90% capacity after 500 cycles at -5 °C.",
        ["Lithium-ion batteries", "90% capacity", "500 cycles", "-5 °C"],
        ["Lithium-ion batteries", "capacity"], "500 cycles at -5 °C",
    ),
    (
        "Graph neural networks improve molecular property prediction "
        "compared with random forests.",
        ["Graph neural networks", "molecular property prediction",
         "random forests"],
        ["Graph neural networks", "molecular property prediction"],
        "compared with random forests",
    ),
])
def test_selection_keeps_surface_terms_and_records_conditions(
    claim: str, phrases: list[str], expected: list[str], omitted: str
) -> None:
    selection = select_discovery_concepts(
        claim, extractor=FixedExtractor(phrases)
    )
    assert selection.phrases == expected
    assert omitted in " ".join(selection.omitted)
    assert all(phrase in claim for phrase in selection.phrases)


def test_missing_model_falls_back_to_original_phrase(monkeypatch) -> None:
    """An extraction failure cannot trigger unbounded or token-AND search."""
    monkeypatch.setattr(
        "paper_search.arxiv_rendering.SpacyConceptExtractor",
        lambda: FixedExtractor([], "unavailable"),
    )
    original = "HER2+ cells do not respond at 10 mg/kg."
    rendered = render_query(original, "arxiv")
    assert rendered.original == original
    assert rendered.text == f'all:"{original}"'
    assert rendered.extraction_status == "unavailable"
    assert "no_concepts_original_phrase_fallback" in rendered.warnings


def test_explicit_phrase_is_not_double_quoted(monkeypatch) -> None:
    monkeypatch.setattr(
        "paper_search.arxiv_rendering.SpacyConceptExtractor",
        lambda: FixedExtractor(['"HER2+ cells"', "DNA repair"]),
    )
    rendered = render_query('"HER2+ cells" affect DNA repair.', "arxiv")
    assert rendered.text == 'all:"HER2+ cells" AND all:"DNA repair"'


def test_numeric_scientific_names_are_not_mistaken_for_conditions() -> None:
    claim = "2D materials improve 5G antennas at 100 kHz."
    selected = select_discovery_concepts(
        claim, extractor=FixedExtractor(["2D materials", "5G antennas",
                                        "100 kHz"])
    )
    assert selected.phrases == ["2D materials", "5G antennas"]
    assert "100 kHz" in " ".join(selected.omitted)


def test_modes_preserve_original_and_do_not_gain_backend_syntax() -> None:
    claim = "  Exercise improves memory.  "
    for mode in ("original_only", "original_plus_concepts"):
        generator = DeterministicQueryGenerator(mode=mode)
        assert generator.generate(claim)[0] == claim
        assert all("all:" not in query for query in generator.generate(claim))


def test_arxiv_client_applies_concepts_before_identical_date_filter() -> None:
    """Intercept a real client call: no changes to its search interface."""
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.params["search_query"] == (
            '(all:"graph neural networks" AND '
            'all:"molecular property prediction") AND '
            'submittedDate:[202001010000 TO 202610302359]'
        )
        assert request.url.params["max_results"] == "10"
        return httpx.Response(200, text=(
            '<feed xmlns="http://www.w3.org/2005/Atom"/>'
        ))

    async def run() -> None:
        from datetime import date

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle)
        ) as http:
            assert await ArxivClient(http).search(
                "Our results show that graph neural networks improve "
                "molecular property prediction compared with random forests.",
                max_results=10, from_publication_date=date(2020, 1, 1),
                to_publication_date=date(2026, 10, 30),
            ) == []
    asyncio.run(run())




