"""Reciprocal Rank Fusion of Person 2's ordered BM25 and SBERT results."""

from ranking_evaluation.schemas import RankedPaper, RankingInput


def reciprocal_rank_fusion(
    inputs: RankingInput, *, rrf_k: int = 60, top_k: int | None = None,
) -> list[RankedPaper]:
    """Fuse ranked IDs, preserving upstream metadata and one-based positions.

    Each present branch contributes 1 / (rrf_k + branch_rank). Candidates
    absent from both branches are excluded. Exact score ties use identity in
    ascending lexical order, independent of input branch/candidate ordering.
    top_k limits the fused output, not the input rankings. Empty branches are
    valid; this function does not infer whether upstream retrieval failed.
    """
    if type(rrf_k) is not int or rrf_k < 0:
        raise ValueError("rrf_k must be a non-negative integer")
    if top_k is not None and (type(top_k) is not int or top_k < 0):
        raise ValueError("top_k must be a non-negative integer or None")

    # Revalidate to catch accidental mutation of lists after model creation.
    inputs = RankingInput.model_validate(inputs.model_dump())
    papers = {entry.identity: entry for entry in inputs.candidates.papers}
    bm25 = {identity: rank for rank, identity in enumerate(inputs.bm25_ranking, 1)}
    sbert = {identity: rank for rank, identity in enumerate(inputs.sbert_ranking, 1)}
    scores = {
        identity: sum(
            1 / (rrf_k + branch[identity])
            for branch in (bm25, sbert) if identity in branch
        )
        for identity in bm25.keys() | sbert.keys()
    }
    ordered = sorted(scores, key=lambda identity: (-scores[identity], identity))
    return [
        RankedPaper(
            entry=papers[identity], rank=rank, rrf_score=scores[identity],
            bm25_rank=bm25.get(identity), sbert_rank=sbert.get(identity),
        )
        for rank, identity in enumerate(ordered[:top_k], 1)
    ]
