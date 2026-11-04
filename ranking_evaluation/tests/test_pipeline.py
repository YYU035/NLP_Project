"""Local module integration with Person 1's schemas and mocked Person 2 ranks."""

import pytest
from pydantic import ValidationError
from paper_search.retrieval_models import ProviderRun

from ranking_evaluation.examples.mock_input import build_mock_input
from ranking_evaluation.pipeline import (
    ModelDescription, RankingConfig, RankingOutput, rank_claim,
)


def description():
    return ModelDescription(model_id="synthetic-scorer", revision="test-only",
                            device="cpu", dtype="float64", max_length=512, batch_size=3)


class Scorer:
    def score_pairs(self, pairs):
        return [0, 1, 2][:len(pairs)]


def test_handoff_round_trip_preserves_all_upstream_records():
    inputs = build_mock_input()
    result = rank_claim(inputs, config=RankingConfig(output_top_k=2))
    restored = RankingOutput.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.inputs == inputs
    assert [row.entry.identity for row in result.papers] == ["synthetic:A", "synthetic:B"]
    assert result.ranked_union_size == 3
    assert result.rerank_scored_count == 0
    assert result.latency.end_to_end_seconds is None
    assert result.latency.rerank_seconds is None
    assert result.ranking_seconds >= result.latency.rrf_seconds


def test_truncation_happens_after_reranking_and_input_is_unchanged():
    inputs = build_mock_input()
    before = inputs.model_dump()
    result = rank_claim(inputs, config=RankingConfig(
        output_top_k=1, rerank_enabled=True, rerank_top_n=3,
    ), scorer=Scorer(), model=description())
    assert result.papers[0].entry.identity == "synthetic:C"
    assert result.papers[0].rank == 1
    assert result.rerank_scored_count == 3
    assert result.latency.rerank_seconds >= 0
    assert inputs.model_dump() == before


def test_partial_search_and_empty_rankings_are_visible():
    inputs = build_mock_input()
    inputs.bm25_ranking = []
    inputs.candidates.providers["arxiv"] = ProviderRun(
        provider="arxiv", status="provider_failure", error="synthetic timeout",
    )
    result = rank_claim(inputs)
    assert len(result.warnings) == 2
    assert result.inputs.candidates.providers["arxiv"].error == "synthetic timeout"
    inputs.sbert_ranking = []
    assert rank_claim(inputs).papers == []


def test_missing_text_warning_and_score_count():
    inputs = build_mock_input()
    inputs.candidates.papers[1].paper.title = None
    result = rank_claim(inputs, config=RankingConfig(rerank_enabled=True),
                        scorer=Scorer(), model=description())
    assert result.rerank_scored_count == 2
    assert "lacked title" in result.warnings[0]
    assert result.papers[1].entry.identity == "synthetic:B"


@pytest.mark.parametrize("kwargs", [
    {"config": RankingConfig(rerank_enabled=True)},
    {"scorer": Scorer()}, {"model": description()},
])
def test_inconsistent_model_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        rank_claim(build_mock_input(), **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"rrf_k": True}, {"output_top_k": -1}, {"rerank_top_n": 1.5},
])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValidationError):
        RankingConfig(**kwargs)


def test_zero_output_limit_keeps_diagnostics_and_union_count():
    result = rank_claim(build_mock_input(), config=RankingConfig(output_top_k=0))
    assert result.papers == []
    assert result.ranked_union_size == 3
    assert len(result.inputs.candidates.papers) == 3


def test_model_failure_propagates():
    class Broken:
        def score_pairs(self, pairs):
            raise RuntimeError("synthetic failure")
    with pytest.raises(RuntimeError, match="synthetic failure"):
        rank_claim(build_mock_input(), config=RankingConfig(rerank_enabled=True),
                   scorer=Broken(), model=description())
