"""Verify the proposed boundary independently of live APIs or NLP models."""

import pytest
from pydantic import ValidationError

from ranking_evaluation.examples.mock_input import build_mock_input
from ranking_evaluation.schemas import RankedPaper, RankingInput


def test_round_trip_retains_upstream_metadata_and_provenance():
    original = build_mock_input()
    restored = RankingInput.model_validate_json(original.model_dump_json())
    assert restored == original
    ranked = RankedPaper(entry=restored.candidates.papers[0], rank=1, rrf_score=0)
    assert ranked.entry.source_records == original.candidates.papers[0].source_records
    assert ranked.entry.provenance == original.candidates.papers[0].provenance


@pytest.mark.parametrize("changes", [
    {"bm25_ranking": ["missing"]},
    {"sbert_ranking": ["synthetic:A", "synthetic:A"]},
    {"bm25_seconds": -1},
    {"sbert_seconds": float("nan")},
])
def test_invalid_rankings_and_times_rejected(changes):
    data = build_mock_input().model_dump()
    data.update(changes)
    with pytest.raises(ValidationError):
        RankingInput.model_validate(data)


def test_empty_rankings_allowed_for_empty_or_failed_branches():
    data = build_mock_input().model_dump()
    data.update(bm25_ranking=[], sbert_ranking=[])
    assert RankingInput.model_validate(data).bm25_ranking == []
    data["candidates"]["papers"] = []
    assert RankingInput.model_validate(data).candidates.papers == []


def test_duplicate_candidates_rejected():
    data = build_mock_input().model_dump()
    data["candidates"]["papers"].append(data["candidates"]["papers"][0])
    with pytest.raises(ValidationError, match="identities must be unique"):
        RankingInput.model_validate(data)


def test_claim_mismatch_rejected():
    data = build_mock_input().model_dump()
    data["candidates"]["papers"][0]["original_claim"] = "Different claim"
    with pytest.raises(ValidationError, match="claim must match"):
        RankingInput.model_validate(data)
