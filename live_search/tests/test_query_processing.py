"""Offline strategy, validation, budget, and ablation tests."""

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from paper_search.config import Settings
from paper_search.linguistic import Concept, ExtractionResult
from paper_search.query_processing import (
    DeterministicQueryGenerator,
    QueryGenerator,
)


@pytest.fixture(autouse=True)
def default_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use known spans so strategy tests need no downloaded model."""
    def extract(self, claim: str) -> ExtractionResult:
        phrases = ["technical materials", "thermal performance"]
        if not all(phrase in claim for phrase in phrases):
            return ExtractionResult(status="failed", warnings=["no_concepts"])
        return ExtractionResult(status="ok", concepts=[Concept(
            text=phrase, start=claim.index(phrase),
            end=claim.index(phrase) + len(phrase), kind="noun_phrase"
        ) for phrase in phrases])

    monkeypatch.setattr(
        "paper_search.query_processing.SpacyConceptExtractor.extract", extract
    )


def test_limits_preserve_original_and_allow_per_call_override() -> None:
    """Limit truncates query lists, never technical content within a query."""
    claim = (
        "Our results show that technical materials improve "
        "thermal performance."
    )
    generator = DeterministicQueryGenerator(max_queries=1)
    assert generator.generate(claim) == [claim]
    assert len(generator.generate(claim, max_queries=2)) == 2
    assert generator.generate(claim) == [claim]
    assert len(generator.generate(claim, max_queries=10)) == 2


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2"])
def test_invalid_limits_are_rejected(limit: object) -> None:
    """Budgets must be positive integers, excluding booleans and coercion."""
    with pytest.raises(ValidationError):
        DeterministicQueryGenerator(max_queries=limit)
    with pytest.raises(ValidationError):
        DeterministicQueryGenerator().generate("A claim", max_queries=limit)


@pytest.mark.parametrize("claim", ["", " \n\t", "...?!", None, 42])
def test_invalid_claims_are_rejected(claim: object) -> None:
    """Reject blank/non-text input and punctuation-only claims."""
    with pytest.raises(ValueError):
        DeterministicQueryGenerator().generate(claim)


def test_failed_extraction_does_not_remove_original() -> None:
    """Failed extraction retains the original and never emits blank text."""
    assert DeterministicQueryGenerator().generate("the") == ["the"]


def test_configured_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read the query budget using the existing explicit settings loader."""
    monkeypatch.setenv("PAPER_SEARCH_MAX_QUERIES", "1")
    settings = Settings.from_environment()
    generator = DeterministicQueryGenerator(max_queries=settings.max_queries)
    assert generator.generate("The treatment may help") == [
        "The treatment may help"
    ]
    monkeypatch.setenv("PAPER_SEARCH_MAX_QUERIES", "0")
    with pytest.raises(ValidationError):
        Settings.from_environment()


def test_strategy_can_be_replaced() -> None:
    """Consumers can depend on the protocol instead of an implementation."""
    class AlternativeGenerator:
        def generate(
            self, claim: str, *, max_queries: int | None = None
        ) -> list[str]:
            """Stand in for a future externally backed strategy."""
            return [claim]

    strategy: QueryGenerator = AlternativeGenerator()
    assert strategy.generate("Exercise improves memory", max_queries=1) == [
        "Exercise improves memory"
    ]


def test_original_only_ablation_never_calls_parser() -> None:
    """The control arm cannot depend on extraction or model availability."""
    extractor = Mock()
    extractor.extract.side_effect = AssertionError("Must not parse control")
    generator = DeterministicQueryGenerator(
        mode="original_only", extractor=extractor
    )
    claim = "  TNF-α\n does not change. "
    details = generator.generate_with_details(claim)
    assert details.queries == [claim]
    assert details.extraction.status == "not_requested"
    extractor.extract.assert_not_called()


def test_query_limit_one_bypasses_extraction() -> None:
    """An explicit one-query budget behaves like the original-only arm."""
    extractor = Mock()
    generator = DeterministicQueryGenerator(extractor=extractor)
    assert generator.generate("A claim", max_queries=1) == ["A claim"]
    extractor.extract.assert_not_called()


def test_diagnostics_and_duplicate_candidates() -> None:
    """Expose extractor errors and deduplicate an already quoted query."""
    extractor = Mock()
    extractor.extract.return_value = ExtractionResult(
        status="unavailable", warnings=["model_unavailable"]
    )
    generator = DeterministicQueryGenerator(extractor=extractor)
    details = generator.generate_with_details("A technical claim")
    assert details.queries == ["A technical claim"]
    assert details.extraction.warnings == ["model_unavailable"]
    extractor.extract.return_value = ExtractionResult(status="ok", concepts=[
        Concept(text='"DNA repair"', start=0, end=12, kind="quoted_phrase")
    ])
    assert generator.generate('"DNA repair"') == ['"DNA repair"']


def test_ablation_environment_and_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Project configuration selects the requested ablation explicitly."""
    monkeypatch.setenv("PAPER_SEARCH_QUERY_MODE", "original_only")
    monkeypatch.setenv("PAPER_SEARCH_QUERY_MODEL", "missing-model")
    generator = DeterministicQueryGenerator.from_settings(
        Settings.from_environment()
    )
    assert generator.generate("The technical claim") == ["The technical claim"]
    monkeypatch.setenv("PAPER_SEARCH_QUERY_MODE", "unknown")
    with pytest.raises(ValidationError):
        Settings.from_environment()
    with pytest.raises(ValidationError):
        DeterministicQueryGenerator(mode="unknown")
