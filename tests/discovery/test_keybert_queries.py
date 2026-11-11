"""Offline KeyBERT strategy contracts; model weights are never downloaded."""

import pytest

from paper_search.keybert_queries import (
    KeyBERTQueryGenerator,
    candidate_phrases,
)


class FixedKeywords:
    """Inject scored source phrases instead of neural inference."""

    def __init__(self, ranked: list[tuple[str, float]]) -> None:
        self.ranked = ranked

    def extract_keywords(self, claim: str, **options):
        assert options["use_mmr"] is True
        assert options["diversity"] == 0.3
        return self.ranked


def test_candidates_preserve_scientific_atoms_and_source_case() -> (
    None
):
    claim = "TNF-α does not increase COVID-19 risk at -5 °C in HER2+ cells."
    phrases = candidate_phrases(claim)
    assert "TNF-α" in phrases
    assert "COVID-19 risk" in phrases
    assert "-5 °C" in phrases
    assert "HER2+ cells" in phrases
    assert all("does not" not in phrase for phrase in phrases)
    assert all(phrase in claim for phrase in phrases)


def test_generator_compatible_original_first_and_scientific_phrases(
    tmp_path,
):
    pytest.importorskip("sklearn")
    claim = (
        "  TNF-α does not increase COVID-19 risk in older adults.  "
    )
    generator = KeyBERTQueryGenerator(
        tmp_path,
        extractor=FixedKeywords(
            [
                ("COVID-19 risk", 0.8),
                ("COVID-19", 0.7),
                ("TNF-α", 0.6),
                ("older adults", 0.5),
            ]
        ),
    )
    details = generator.generate_with_details(claim)
    assert details.queries == [
        claim,
        "TNF-α COVID-19 risk older adults",
    ]
    assert details.selected_phrases == [
        "TNF-α",
        "COVID-19 risk",
        "older adults",
    ]
    assert "does not increase" in " ".join(details.omitted_spans)
    assert "all:" not in details.queries[1]
    assert generator.generate(claim, max_queries=1) == [claim]


def test_original_only_does_not_load_optional_dependencies(
    tmp_path,
):
    generator = KeyBERTQueryGenerator(tmp_path, max_queries=1)
    assert generator.generate(" Exercise improves memory. ") == [
        " Exercise improves memory. "
    ]


def test_missing_local_model_fails_without_remote_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match="pinned local model"):
        KeyBERTQueryGenerator(tmp_path / "missing")._load()


@pytest.mark.parametrize("claim", ["", "   ", "!!!"])
def test_invalid_claim_rejected_without_model(tmp_path, claim):
    with pytest.raises(ValueError):
        KeyBERTQueryGenerator(tmp_path).generate(claim)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_query_limit(tmp_path, limit):
    with pytest.raises(ValueError):
        KeyBERTQueryGenerator(tmp_path).generate(
            "Scientific claim", max_queries=limit
        )


def test_insufficient_nonoverlapping_candidates_abstains(tmp_path):
    pytest.importorskip("sklearn")
    generator = KeyBERTQueryGenerator(
        tmp_path,
        extractor=FixedKeywords(
            [
                ("graph neural networks", 0.9),
                ("neural networks", 0.8),
            ]
        ),
    )
    claim = "Graph neural networks improve predictions."
    # Case is exact; no unsupported casing/synonym inference is performed.
    details = generator.generate_with_details(claim)
    assert details.queries == [claim]
    assert details.status == "abstained"


def test_candidate_ner_labels_and_outcomes_do_not_use_spacy():
    phrases = candidate_phrases(
        "vitamin D supplementation may reduce respiratory infections."
    )
    assert "vitamin D supplementation" in phrases
    assert "respiratory infections" in phrases
    assert not any("may" in phrase.split() for phrase in phrases)


def test_measurement_units_remain_complete_without_empty_punctuation_atoms():
    claim = "SiC MOSFETs operate at 100 kHz and -5 °C."
    candidates = candidate_phrases(claim)
    assert "100 kHz" in candidates
    assert "-5 °C" in candidates
    assert all(c == c.strip() and c in claim for c in candidates)
    assert "100" not in candidates
