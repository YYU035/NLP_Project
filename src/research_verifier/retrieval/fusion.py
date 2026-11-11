"""Reciprocal Rank Fusion of Person 2's ordered BM25 and SBERT results."""

from research_verifier.retrieval.fusion_schemas import RankedPaper, RankingInput



def fuse_ordered_ids(bm25_ids, sbert_ids, *, rrf_k=60):
    """Lightweight ID/score fusion shared by the runtime and ablation tools."""
    if type(rrf_k) is not int or rrf_k < 0:
        raise ValueError('rrf_k must be a non-negative integer')
    for ids in (bm25_ids, sbert_ids):
        if len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError('Ranking IDs must be unique nonempty strings')
    branches=[{identity:rank for rank,identity in enumerate(ids,1)} for ids in (bm25_ids,sbert_ids)]
    scores={identity:sum(1/(rrf_k+branch[identity]) for branch in branches if identity in branch)
            for identity in branches[0].keys() | branches[1].keys()}
    return sorted(scores.items(),key=lambda row:(-row[1],row[0]))


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
    fused = fuse_ordered_ids(inputs.bm25_ranking, inputs.sbert_ranking, rrf_k=rrf_k)
    scores = dict(fused)
    ordered = [identity for identity, _ in fused]
    return [
        RankedPaper(
            entry=papers[identity], rank=rank, rrf_score=scores[identity],
            bm25_rank=bm25.get(identity), sbert_rank=sbert.get(identity),
        )
        for rank, identity in enumerate(ordered[:top_k], 1)
    ]
