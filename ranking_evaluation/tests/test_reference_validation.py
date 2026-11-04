"""Additional reproducible checks against exact arithmetic and another library."""

from fractions import Fraction
from random import Random

import pytest

from ranking_evaluation.examples.mock_input import build_mock_input
from ranking_evaluation.metrics import ndcg_at_k
from ranking_evaluation.rrf import reciprocal_rank_fusion
from ranking_evaluation.schemas import RankingInput


def test_rrf_against_exact_fraction_arithmetic_over_100_inputs():
    random = Random(20261104)
    template = build_mock_input().model_dump()
    ids = [row["identity"] for row in template["candidates"]["papers"]]
    for _ in range(100):
        branches = [random.sample(ids, random.randrange(4)) for _ in range(2)]
        constant = random.choice([0, 1, 20, 60, 100])
        data = dict(template, bm25_ranking=branches[0], sbert_ranking=branches[1])
        output = reciprocal_rank_fusion(RankingInput.model_validate(data), rrf_k=constant)
        exact = {}
        for identity in ids:
            if any(identity in branch for branch in branches):
                exact[identity] = sum((Fraction(1, constant + branch.index(identity) + 1)
                                       for branch in branches if identity in branch), Fraction())
        expected = sorted(exact, key=lambda identity: (-exact[identity], identity))
        assert [row.entry.identity for row in output] == expected
        for row in output:
            assert row.rrf_score == pytest.approx(float(exact[row.entry.identity]))


def test_binary_ndcg_against_sklearn_over_100_cutoffs():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    random = Random(20261104)
    for _ in range(100):
        size = random.randint(2, 20)
        corpus = [str(index) for index in range(size)]
        ranking = random.sample(corpus, len(corpus))
        gold = set(random.sample(corpus, random.randint(1, size)))
        k = random.randint(1, size + 3)
        relevance = [[int(identity in gold) for identity in corpus]]
        scores = [[size - ranking.index(identity) for identity in corpus]]
        reference = sklearn_metrics.ndcg_score(relevance, scores, k=k, ignore_ties=True)
        assert ndcg_at_k(ranking, gold, k) == pytest.approx(reference)
