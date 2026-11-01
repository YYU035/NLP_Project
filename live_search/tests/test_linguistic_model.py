"""Local model checks; no network, LLM, or paper-search requests."""

import json
from pathlib import Path

import pytest
import spacy

from paper_search.linguistic import SpacyConceptExtractor
from paper_search.query_processing import DeterministicQueryGenerator

_AUDIT_CASES = json.loads(
    (Path(__file__).resolve().parents[1] / "docs/query_audit_before.json")
    .read_text(encoding="utf-8")
)["examples"]
_EXPECTED = {
    "biomedical": "the expression of TNF-α does not increase the risk of "
    "COVID-19 in older adults.",
    "battery_engineering": "lithium-ion batteries retain 90% capacity after "
    "500 cycles at -5 °C.",
    "power_electronics": "the efficiency of SiC MOSFETs is not lower than "
    "that of silicon devices at 100 kHz.",
    "climate": "elevated atmospheric CO2 concentrations increase ocean "
    "acidification in coastal waters.",
    "agriculture": "deficit irrigation reduces water use without decreasing "
    "wheat yield under drought conditions.",
    "computer_science": "graph neural networks improve molecular property "
    "prediction compared with random forests.",
    "catalysis": "platinum nanoparticles accelerate hydrogen evolution at "
    "25 °C in acidic solutions.",
    "biomedical_population": "vitamin A supplementation does not reduce "
    "mortality in children under five years of age.",
}


@pytest.fixture(scope="module")
def generator() -> DeterministicQueryGenerator:
    """Use the installed English model; report a skip if weights are absent."""
    try:
        nlp = spacy.load("en_core_web_sm", exclude=["lemmatizer"])
    except OSError:
        pytest.skip("Install .[nlp-model] to exercise real model extraction")
    return DeterministicQueryGenerator(
        extractor=SpacyConceptExtractor(nlp=nlp)
    )


@pytest.mark.parametrize("claim", [
    _EXPECTED[key] for key in (
        "biomedical", "battery_engineering", "climate", "power_electronics"
    )
])
def test_direct_domain_queries_need_no_alternative(
    generator, claim: str,
) -> None:
    """Direct assertions do not need quotation-only alternatives."""
    details = generator.generate_with_details(claim)
    assert details.queries == [claim]
    assert details.construction.status == "abstained"
    assert '"' not in details.queries[0]
    assert details.extraction.status == "ok"
    assert details.extraction.model_version == "3.8.0"
    assert details.extraction.pos_tags
    assert generator.generate(claim) == details.queries


@pytest.mark.parametrize("phrase", [
    "does not cause", "has no effect on", "never increases", "may not reduce",
    "is not more than", "is neither necessary nor sufficient", "might affect",
    "doesn't increase", "doesn’t increase", "increases rather than decreases",
    "affects only", "affects at most", "reduces compared with",
])
def test_all_nonarticle_words_preserved(generator, phrase: str) -> None:
    """Relations/qualifiers and their order survive even imperfect parses."""
    claim = f"The treatment {phrase} the response."
    queries = generator.generate(claim)
    assert queries[0] == claim
    # Phrase quotation must not delete/reorder/rewrite any nonarticle token.
    original_tokens = [w for w in claim.split() if w not in {"The", "the"}]
    focused_tokens = [w for w in queries[-1].replace('"', '').split()
                      if w not in {"The", "the"}]
    assert focused_tokens == original_tokens


@pytest.mark.parametrize("claim,protected", [
    ("The p53/TP53 pathway does not alter m6A at p < 0.05.",
     ["p53/TP53", "m6A", "<", "0.05"]),
    ("The β-catenin protein affects CD4+ T cells at 5 mg/kg.",
     ["β-catenin", "CD4+", "5", "mg/kg"]),
    ("A priori evidence does not establish causation.", ["A priori"]),
    ("a posteriori evidence may be insufficient.", ["a posteriori"]),
    ('The "non-small cell lung cancer" response differs from chemotherapy.',
     ["non-small cell lung cancer"]),
])
def test_technical_forms_and_phrases(generator, claim: str, protected) -> None:
    """Keep scientific forms rather than stemming or replacing synonyms."""
    for query in generator.generate(claim):
        unquoted = query.replace('"', '')
        assert all(term in unquoted for term in protected)


def test_observed_extraction_failure(generator) -> None:
    """The real parser's no-concept result is visible to callers."""
    result = generator.generate_with_details("Run quickly.")
    assert result.queries == ["Run quickly."]
    assert result.extraction.status == "failed"
    assert result.extraction.warnings == ["no_concepts_extracted"]


@pytest.mark.parametrize("case", _AUDIT_CASES, ids=[
    case["domain"] for case in _AUDIT_CASES
])
def test_fourteen_diverse_audit_regressions(generator, case: dict) -> None:
    """Test exact reviewed outputs and abstentions for all audit claims."""
    result = generator.generate_with_details(case["claim"])
    assert result.queries[0] == case["claim"]
    assert result.extraction.status == "ok"
    expected = _EXPECTED.get(case["domain"])
    if expected:
        assert result.queries == [case["claim"], expected]
        assert '"' not in expected
        assert len(result.construction.selected_concepts) >= 2
        assert result.construction.recall_risks
    else:
        assert result.queries == [case["claim"]]
        assert result.construction.status == "abstained"
    assert generator.generate(case["claim"]) == result.queries
