"""Offline synthetic demo: python -m ranking_evaluation.examples.fuse_mock."""

import json

from ranking_evaluation.examples.mock_input import build_mock_input
from ranking_evaluation.rrf import reciprocal_rank_fusion


if __name__ == "__main__":
    inputs = build_mock_input()
    result = {
        "example_type": "synthetic; not a retrieval experiment",
        "claim_id": inputs.claim_id,
        "original_claim": inputs.candidates.original_claim,
        "rrf_k": 60,
        "papers": [
            paper.model_dump(mode="json")
            for paper in reciprocal_rank_fusion(inputs)
        ],
    }
    print(json.dumps(result, indent=2))
