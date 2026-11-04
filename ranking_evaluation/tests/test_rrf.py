"""Hand-computed fusion cases, boundaries and upstream reference integrity."""

import pytest
from pydantic import ValidationError

from ranking_evaluation.examples.mock_input import build_mock_input
from ranking_evaluation.rrf import reciprocal_rank_fusion
from ranking_evaluation.schemas import RankingInput


def make_input(bm25, sbert):
    data = build_mock_input().model_dump()
    data.update(bm25_ranking=bm25, sbert_ranking=sbert)
    return RankingInput.model_validate(data)


def test_hand_computed_scores_and_provenance():
    inputs = build_mock_input()
    result = reciprocal_rank_fusion(inputs)
    assert [row.entry.identity for row in result] == [
        "synthetic:A", "synthetic:B", "synthetic:C",
    ]
    assert [row.rrf_score for row in result] == pytest.approx([
        1 / 61 + 1 / 62, 1 / 63 + 1 / 61, 1 / 62 + 1 / 63,
    ])
    assert [(row.bm25_rank, row.sbert_rank) for row in result] == [
        (1, 2), (3, 1), (2, 3),
    ]
    assert [row.rank for row in result] == [1, 2, 3]
    assert result[0].entry == inputs.candidates.papers[0]
    assert result[0].cross_encoder_score is None


@pytest.mark.parametrize("branch", ["bm25", "sbert"])
def test_one_branch_and_unranked_candidates(branch):
    ids = ["synthetic:C", "synthetic:A"]
    inputs = make_input(ids if branch == "bm25" else [],
                        ids if branch == "sbert" else [])
    result = reciprocal_rank_fusion(inputs, rrf_k=0)
    assert [row.entry.identity for row in result] == ids
    assert [row.rrf_score for row in result] == [1, 0.5]
    assert getattr(result[0], "sbert_rank" if branch == "bm25" else "bm25_rank") is None


def test_partial_overlap_uses_union():
    result = reciprocal_rank_fusion(make_input(
        ["synthetic:A", "synthetic:B"], ["synthetic:C", "synthetic:A"],
    ))
    assert [row.entry.identity for row in result] == [
        "synthetic:A", "synthetic:C", "synthetic:B",
    ]


def test_exact_ties_use_identity_and_are_branch_symmetric():
    inputs = make_input(["synthetic:B", "synthetic:A"],
                        ["synthetic:A", "synthetic:B"])
    result = reciprocal_rank_fusion(inputs)
    assert [row.entry.identity for row in result] == ["synthetic:A", "synthetic:B"]
    swapped = make_input(inputs.sbert_ranking, inputs.bm25_ranking)
    assert [row.entry.identity for row in reciprocal_rank_fusion(swapped)] == [
        row.entry.identity for row in result
    ]


def test_empty_branches_and_empty_candidates():
    inputs = make_input([], [])
    assert reciprocal_rank_fusion(inputs) == []
    inputs.candidates.papers.clear()
    assert reciprocal_rank_fusion(inputs) == []


@pytest.mark.parametrize("limit,expected_count", [(0, 0), (1, 1), (3, 3), (10, 3)])
def test_output_limit_is_prefix(limit, expected_count):
    inputs = build_mock_input()
    result = reciprocal_rank_fusion(inputs, top_k=limit)
    assert len(result) == expected_count
    assert result == reciprocal_rank_fusion(inputs)[:limit]


@pytest.mark.parametrize("parameter,value", [
    ("rrf_k", -1), ("rrf_k", 1.5), ("rrf_k", True),
    ("top_k", -1), ("top_k", 1.5), ("top_k", False),
])
def test_invalid_parameters(parameter, value):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(build_mock_input(), **{parameter: value})


def test_rejects_mutated_invalid_input_and_does_not_change_valid_input():
    inputs = build_mock_input()
    before = inputs.model_dump()
    reciprocal_rank_fusion(inputs)
    assert inputs.model_dump() == before
    inputs.bm25_ranking.append("unknown")
    with pytest.raises(ValidationError):
        reciprocal_rank_fusion(inputs)
