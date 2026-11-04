"""Synthetic arithmetic demo, not SciFact or measured retrieval performance."""

import json

from ranking_evaluation.examples.mock_input import build_mock_input
from ranking_evaluation.latency import LatencyRecord, measure_call
from ranking_evaluation.metrics import evaluate_rankings
from ranking_evaluation.rrf import reciprocal_rank_fusion


if __name__ == "__main__":
    inputs = build_mock_input()
    fused, seconds = measure_call(lambda: reciprocal_rank_fusion(inputs))
    gold = {inputs.claim_id: {"synthetic:B"}}  # Invented label for arithmetic only.
    rankings = {
        "bm25": inputs.bm25_ranking,
        "sbert": inputs.sbert_ranking,
        "rrf": [row.entry.identity for row in fused],
    }
    print(json.dumps({
        "example_type": "synthetic; labels and rankings are manually constructed",
        "metrics": {name: evaluate_rankings({inputs.claim_id: ids}, gold, ks=(1, 3))
                    for name, ids in rankings.items()},
        "latency": LatencyRecord(rrf_seconds=seconds).model_dump(),
    }, indent=2))
