"""Hand-computed metrics and safeguards against evaluation bias."""

from math import log2

import pytest

from ranking_evaluation.metrics import (
    evaluate_rankings, ndcg_at_k, recall_at_k, reciprocal_rank,
)


def test_hand_computed_metrics():
    ranking, gold = ["x", "a", "b"], {"a", "b", "unretrieved"}
    assert recall_at_k(ranking, gold, 2) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranking, gold) == 0.5
    assert reciprocal_rank(ranking, gold, k=1) == 0
    assert ndcg_at_k(ranking, gold, 3) == pytest.approx(
        (1 / log2(3) + 0.5) / (1 + 1 / log2(3) + 0.5)
    )


def test_perfect_empty_and_short_rankings():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 10) == 1
    assert recall_at_k([], {"a"}, 10) == 0
    assert ndcg_at_k([], {"a"}, 10) == 0
    assert reciprocal_rank([], {"a"}) == 0
    assert ndcg_at_k(["a"], {"a", "b"}, 2) == pytest.approx(1 / (1 + 1/log2(3)))


def test_no_gold_is_undefined():
    assert recall_at_k(["a"], [], 1) is None
    assert reciprocal_rank([], []) is None
    assert ndcg_at_k([], [], 1) is None


def test_macro_counts_missing_rankings_and_reports_exclusions():
    report = evaluate_rankings(
        {"c1": ["x", "a"]}, {"c1": {"a"}, "c2": {"b"}, "c3": set()}, ks=(1, 2),
    )
    assert report["total_claims"] == 3
    assert report["evaluated_claims"] == 2
    assert report["excluded_no_relevant"] == 1
    assert report["missing_rankings"] == 2
    assert report["macro"]["MRR"] == 0.25
    assert report["macro"]["Recall@2"] == 0.5
    assert report["macro"]["nDCG@2"] == pytest.approx(1 / log2(3) / 2)
    assert report["per_claim"]["c2"]["MRR"] == 0


@pytest.mark.parametrize("qrels", [{}, {"c": set()}])
def test_no_evaluable_claims_has_no_fabricated_mean(qrels):
    report = evaluate_rankings({}, qrels)
    assert all(value is None for value in report["macro"].values())


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_cutoffs(k):
    with pytest.raises(ValueError):
        recall_at_k([], {"a"}, k)


def test_rejects_duplicates_unknown_claims_and_invalid_ks():
    for metric in (recall_at_k, ndcg_at_k):
        with pytest.raises(ValueError):
            metric(["a", "a"], {"a"}, 2)
    with pytest.raises(ValueError):
        reciprocal_rank(["a", "a"], {"a"})
    with pytest.raises(ValueError):
        evaluate_rankings({"unknown": []}, {"c": {"a"}})
    for ks in ((), (1, 1), (0,)):
        with pytest.raises(ValueError):
            evaluate_rankings({}, {}, ks=ks)
