"""Bridge to the pinned Person 3 runtime; never invent Person 2 rankings."""
import json
from pathlib import Path
import sys

from research_verifier.contracts.models import digest

REVISION = "7ed93fa4be69c661422b33b6c88b949dd2636e12"


def runtime():
    # The shared package uses absolute imports. Its unchanged runtime is scoped here.
    from research_verifier.retrieval.fusion_pipeline import RankingInput, RankingOutput, rank_claim
    return RankingInput, RankingOutput, rank_claim


def ranking_request(snapshot):
    """An unfilled contract, not an assertion that retrieval has run."""
    return {"claim_id": snapshot.fingerprint, "candidates": snapshot.result.model_dump(mode="json"),
            "bm25_ranking": [], "sbert_ranking": [], "bm25_seconds": None, "sbert_seconds": None}


def connect_ranking(raw, result):
    if len(raw) > 10_000_000:
        raise ValueError("Ranking file exceeds 10 MB")
    data = json.loads(raw)
    Input, Output, rank_claim = runtime()
    if not isinstance(data, dict):
        raise ValueError("Expected ranking object")
    output = Output.model_validate(data) if "inputs" in data else rank_claim(Input.model_validate(data))
    if digest(output.inputs.candidates.model_dump(mode="json")) != digest(result.model_dump(mode="json")):
        raise ValueError("Ranking candidates must exactly match the loaded search")
    # Until the local reranker environment is validated, accept only reproducible RRF.
    if output.configuration.rerank_enabled or output.model is not None:
        raise ValueError("Cross-encoder integration is pending; supply RRF-only output")
    expected = rank_claim(output.inputs, config=output.configuration)
    if ([p.model_dump(mode="json") for p in output.papers] !=
            [p.model_dump(mode="json") for p in expected.papers]):
        raise ValueError("Ranked rows do not match the supplied branch rankings")
    if output.ranked_union_size != expected.ranked_union_size or output.rerank_scored_count != 0:
        raise ValueError("Ranking counts are inconsistent")
    return output.model_dump(mode="json")


def ranked_candidates(candidates, ranking):
    if ranking is None:
        return candidates
    positions = {row["entry"]["identity"]: row["rank"] for row in ranking["papers"]}
    return sorted((p for p in candidates if p.identity in positions), key=lambda p: positions[p.identity])
