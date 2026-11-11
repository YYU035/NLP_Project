"""Extraction mechanics using annotated docs, without trained model weights."""

from unittest.mock import Mock

import pytest
import spacy
from spacy.tokens import Doc

from paper_search.linguistic import (
    Concept,
    ExtractionResult,
    SpacyConceptExtractor,
)
from paper_search.query_construction import build_focused_query
from paper_search.query_processing import DeterministicQueryGenerator


def test_pos_dependency_chunks_and_qualifiers() -> None:
    """Consume actual POS/dependency annotations and retain source context."""
    nlp = spacy.blank("en")
    claim = "The treatment does not reduce mortality in older adults."
    doc = Doc(
        nlp.vocab,
        words=["The", "treatment", "does", "not", "reduce", "mortality",
               "in", "older", "adults", "."],
        spaces=[True] * 8 + [False, False],
        heads=[1, 4, 4, 4, 4, 4, 4, 8, 6, 4],
        deps=["det", "nsubj", "aux", "neg", "ROOT", "dobj", "prep",
              "amod", "pobj", "punct"],
        pos=["DET", "NOUN", "AUX", "PART", "VERB", "NOUN", "ADP",
             "ADJ", "NOUN", "PUNCT"],
    )
    extractor = SpacyConceptExtractor(nlp=lambda _: doc)
    result = extractor.extract(claim)
    assert result.status == "ok"
    assert [c.text for c in result.concepts] == [
        "treatment", "mortality", "older adults"
    ]
    assert [c.dependency for c in result.concepts] == ["nsubj", "dobj", "pobj"]
    assert ("not", "PART") in result.pos_tags
    assert build_focused_query(claim, result).query is None
    assert not any('"' in c.text for c in result.concepts)


def test_scientific_atom_recovery() -> None:
    """Repair a noun chunk covering only part of a technical identifier."""
    nlp = spacy.blank("en")
    claim = "TNF-α binds DNA."
    doc = Doc(
        nlp.vocab, words=["TNF", "-", "α", "binds", "DNA", "."],
        spaces=[False, False, True, True, False, False],
        heads=[3, 0, 3, 3, 3, 3],
        deps=["nsubj", "punct", "nsubj", "ROOT", "dobj", "punct"],
        pos=["PROPN", "PUNCT", "PROPN", "VERB", "PROPN", "PUNCT"],
    )
    result = SpacyConceptExtractor(nlp=lambda _: doc).extract(claim)
    assert result.status == "ok"
    assert [c.text for c in result.concepts] == ["TNF-α", "DNA"]
    assert build_focused_query(claim, result).query is None


def test_missing_model_falls_back_explicitly(monkeypatch) -> None:
    """Never download models or silently call rules POS tagging."""
    monkeypatch.setattr(
        "paper_search.linguistic._load_model", Mock(side_effect=OSError)
    )
    details = DeterministicQueryGenerator().generate_with_details(
        "Physical activity improves memory."
    )
    assert details.queries == ["Physical activity improves memory."]
    assert details.extraction.status == "unavailable"
    assert "model_unavailable" in details.extraction.warnings[0]


def test_missing_annotations_is_extraction_failure() -> None:
    """A blank tokenizer alone is not represented as a trained POS model."""
    result = SpacyConceptExtractor(nlp=spacy.blank("en")).extract(
        "Exercise improves memory."
    )
    assert result.status == "failed"
    assert result.concepts == []


def test_no_nouns_and_parser_errors() -> None:
    """No-noun input and parser exceptions retain explicit failure states."""
    nlp = spacy.blank("en")
    doc = Doc(
        nlp.vocab, words=["Run", "quickly", "."],
        spaces=[True, False, False], heads=[0, 0, 0],
        deps=["ROOT", "advmod", "punct"], pos=["VERB", "ADV", "PUNCT"],
    )
    result = SpacyConceptExtractor(nlp=lambda _: doc).extract("Run quickly.")
    assert result.warnings == ["no_concepts_extracted"]
    result = SpacyConceptExtractor(
        nlp=Mock(meta={}, side_effect=RuntimeError("broken pipeline"))
    ).extract("A claim")
    assert result.status == "failed"
    assert result.warnings == ["extraction_error:RuntimeError"]


@pytest.mark.parametrize("claim", [
    'The "unclosed phrase', "The “unclosed phrase"
])
def test_unbalanced_quotes_do_not_invent_phrase_boundaries(claim: str) -> None:
    """Return original control instead of generating malformed query syntax."""
    result = SpacyConceptExtractor(nlp=Mock(meta={})).extract(claim)
    assert result.status == "failed"
    assert result.warnings == ["unbalanced_quotes:original_retained"]


def test_conflicting_spans_are_not_silently_applied() -> None:
    """The query builder rejects overlapping edits instead of losing text."""
    result = ExtractionResult(status="ok", concepts=[
        Concept(text="DNA", start=0, end=3, kind="noun_phrase"),
        Concept(text="DNA", start=0, end=3, kind="entity"),
    ])
    decision = build_focused_query("DNA repair", result)
    assert decision.query is None
    assert decision.reason == "invalid_or_overlapping_source_spans"
    assert result.warnings == []
