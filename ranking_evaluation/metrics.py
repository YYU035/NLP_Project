"""Binary relevance metrics, with explicit handling of unlabeled claims."""

from collections.abc import Collection, Sequence
from math import log2


def _validate(ranking: Sequence[str], k: int | None = None) -> None:
    if len(ranking) != len(set(ranking)):
        raise ValueError("Ranking must not contain duplicate document IDs")
    if k is not None and (type(k) is not int or k <= 0):
        raise ValueError("k must be a positive integer")


def recall_at_k(
    ranking: Sequence[str], relevant: Collection[str], k: int,
) -> float | None:
    """Recall against ALL gold documents, not just retrieved candidates.

    None means there are no gold relevant documents; exclude from macro mean
    and report the excluded count. An empty ranking with gold documents is 0.
    """
    _validate(ranking, k)
    gold = set(relevant)
    return len(set(ranking[:k]) & gold) / len(gold) if gold else None


def reciprocal_rank(
    ranking: Sequence[str], relevant: Collection[str], *, k: int | None = None,
) -> float | None:
    """Per-claim RR (or RR@K); MRR is its mean over eligible claims.

    Without k, evaluate the complete supplied ranking, not an unseen corpus.
    """
    _validate(ranking, k)
    gold = set(relevant)
    if not gold:
        return None
    return next((1 / rank for rank, identity in enumerate(ranking[:k], 1)
                 if identity in gold), 0.0)


def ndcg_at_k(
    ranking: Sequence[str], relevant: Collection[str], k: int,
) -> float | None:
    """Binary gain (1 relevant, 0 otherwise), discount 1/log2(rank+1)."""
    _validate(ranking, k)
    gold = set(relevant)
    if not gold:
        return None
    dcg = sum(1 / log2(rank + 1)
              for rank, identity in enumerate(ranking[:k], 1) if identity in gold)
    ideal = sum(1 / log2(rank + 1) for rank in range(1, min(k, len(gold)) + 1))
    return dcg / ideal


def evaluate_rankings(
    rankings: dict[str, Sequence[str]],
    qrels: dict[str, Collection[str]],
    *, ks: Sequence[int] = (5, 10, 20),
) -> dict:
    """Evaluate every qrels claim; missing rankings count as empty results.

    Unknown claims are rejected so ID mismatches cannot silently inflate means.
    qrels must include claims with empty gold sets to report exclusions.
    """
    if not ks or len(ks) != len(set(ks)):
        raise ValueError("ks must be nonempty and unique")
    for k in ks:
        _validate([], k)
    if rankings.keys() - qrels.keys():
        raise ValueError("Rankings contain claim IDs absent from qrels")
    rows = {}
    for claim_id, gold in qrels.items():
        ranking = rankings.get(claim_id, [])
        row = {"MRR": reciprocal_rank(ranking, gold)}
        for k in ks:
            row[f"Recall@{k}"] = recall_at_k(ranking, gold, k)
            row[f"MRR@{k}"] = reciprocal_rank(ranking, gold, k=k)
            row[f"nDCG@{k}"] = ndcg_at_k(ranking, gold, k)
        rows[claim_id] = row
    eligible = [row for row in rows.values() if row["MRR"] is not None]
    names = ["MRR"] + [name for k in ks for name in (
        f"Recall@{k}", f"MRR@{k}", f"nDCG@{k}",
    )]
    return {
        "total_claims": len(qrels),
        "evaluated_claims": len(eligible),
        "excluded_no_relevant": len(qrels) - len(eligible),
        "missing_rankings": len(qrels.keys() - rankings.keys()),
        "macro": {name: sum(row[name] for row in eligible) / len(eligible)
                  if eligible else None for name in names},
        "per_claim": rows,
    }
