"""Regression tests separating semantic extraction from query construction."""

from copy import deepcopy

import pytest

from paper_search.linguistic import Concept, ExtractionResult
from paper_search.query_construction import build_focused_query
from paper_search.query_processing import DeterministicQueryGenerator
from paper_search.query_rendering import render_query


def _extraction(claim: str, phrases: list[str]) -> ExtractionResult:
    """Supply source spans independently of statistical model weights."""
    return ExtractionResult(status="ok", concepts=[
        Concept(text=text, start=claim.index(text),
                end=claim.index(text) + len(text), kind="noun_phrase")
        for text in phrases
    ])


def test_meaningful_composed_phrases_without_individual_word_quotes() -> None:
    """Prefer scientific referents over generic expression/risk nouns."""
    claim = (
        "Our results show that the expression of TNF-α does not increase "
        "the risk of COVID-19 in older adults."
    )
    extraction = _extraction(claim, [
        "expression", "TNF-α", "risk", "COVID-19", "older adults"
    ])
    before = deepcopy(extraction.model_dump())
    result = build_focused_query(claim, extraction)
    assert result.query == (
        "the expression of TNF-α does not increase the risk of COVID-19 "
        "in older adults."
    )
    assert [c.text for c in result.selected_concepts] == [
        "expression of TNF-α", "risk of COVID-19", "older adults"
    ]
    assert '"' not in result.query
    assert extraction.model_dump() == before


@pytest.mark.parametrize("frame", [
    "Our results show that ", "This study demonstrates that ",
    "In this study, we demonstrate that ", "Experimental results show that ",
    "It has been demonstrated that ",
])
def test_retains_whole_assertion_and_all_conditions(frame: str) -> None:
    """Negation, comparisons, quantities, units, and populations survive."""
    assertion = (
        "SiC MOSFETs are not less efficient than silicon devices at -5 °C "
        "and 100 kHz in high-temperature applications."
    )
    claim = frame + assertion
    result = build_focused_query(claim, _extraction(claim, [
        "SiC MOSFETs", "silicon devices", "-5 °C", "100 kHz",
        "high-temperature applications"
    ]))
    if result.query is None:
        assert result.reason == "alternative_too_similar"
    else:
        assert result.query == assertion
        assert result.recall_risks


@pytest.mark.parametrize("claim", [
    "Our results do not show that physical activity improves working memory.",
    "Our results only show that physical activity improves working memory.",
    "Preliminary findings suggest that physical activity may improve memory.",
    "Every physical activity intervention improves working memory.",
])
def test_never_removes_negated_or_qualified_frames(claim: str) -> None:
    """Do not erase evidential uncertainty, negation, only, or every."""
    phrases = ["physical activity", "working memory"]
    if "working memory" not in claim:
        phrases = ["physical activity", "memory"]
    decision = build_focused_query(claim, _extraction(claim, phrases))
    assert decision.status == "abstained"
    assert decision.query is None


def test_short_claims_and_generic_nouns_have_no_fake_alternatives() -> None:
    """Article removal and query quotas are not reasons to emit a variant."""
    claim = "The physical activity intervention improves working memory."
    assert build_focused_query(claim, _extraction(claim, [
        "physical activity intervention", "working memory"
    ])).query is None
    generic = "Our results show that the system improves the method."
    decision = build_focused_query(generic, _extraction(generic, [
        "system", "method"
    ]))
    assert decision.reason == "insufficient_meaningful_concepts"


def test_long_assertion_with_small_frame_is_not_a_useful_rewrite() -> None:
    """A tiny reduction in a long claim is rejected as a near duplicate."""
    claim = (
        "Our results show that physical activity improves working memory "
        "in older adults after six months of regular supervised exercise "
        "compared with participants receiving standard care under the same "
        "environmental conditions and assessment protocol."
    )
    decision = build_focused_query(claim, _extraction(claim, [
        "physical activity", "working memory", "older adults"
    ]))
    assert decision.reason == "alternative_too_similar"


def test_primary_claim_preserved_byte_for_byte() -> None:
    """Normalization is internal, never a replacement for the primary query."""
    original = " \t TNF-α\n does not change at -5 °C.  "
    generator = DeterministicQueryGenerator(mode="original_only")
    details = generator.generate_with_details(original)
    assert details.queries == [original]
    assert details.normalized_claim == "TNF-α does not change at -5 °C."
    assert render_query(original, "openalex").original == original


@pytest.mark.parametrize("provider", ["openalex", "arxiv", "semantic_scholar"])
def test_no_automatic_exact_match_for_single_words(provider: str) -> None:
    """Provider syntax does not recreate the old individual-word quotes."""
    rendered = render_query(
        "physical activity may improve working memory", provider
    )
    if provider == "arxiv":
        assert rendered.text == (
            'all:"physical activity" AND all:"working memory"'
        )
    else:
        assert '"' not in rendered.text


def test_provider_quotes_are_explicit_and_supported() -> None:
    """Honor requested phrases on supporting APIs, never on Graph search."""
    query = '"working memory" in older adults'
    assert '"working memory"' in render_query(query, "openalex").text
    assert 'all:"working memory"' in render_query(query, "arxiv").text
    semantic = render_query(query, "semantic_scholar")
    assert '"' not in semantic.text
    assert "phrase_boundaries_not_supported" in semantic.warnings
