"""Retrieval metrics and the BM25 vs SBERT vs RRF comparison.

Metrics (binary relevance, SciFact evidence docs as qrels):
  Recall@K  fraction of a claim's relevant docs found in the top K
  MRR       1 / rank of first relevant doc (0 if none in the list)
  nDCG@K    DCG with gain 1 per relevant doc, normalised by the ideal ranking
Latency is query time only; index-building time is reported separately.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from .schema import RankedList


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr(ranked: list[str], relevant: set[str], k: int | None = None) -> float:
    for i, pid in enumerate(ranked[:k] if k else ranked, start=1):
        if pid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, pid in enumerate(ranked[:k], 1) if pid in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_method(runs: list[RankedList], qrels: dict[str, set[str]],
                    ks=(1, 3, 5, 10, 20, 50, 100)) -> dict:
    per_claim, agg = [], {}
    for run in runs:
        ids = [r.paper_id for r in run.results]
        rel = qrels[run.claim_id]
        row = {"claim_id": run.claim_id, "latency_ms": run.latency_ms,
               "mrr": mrr(ids, rel), "first_relevant_rank": next(
                   (i for i, p in enumerate(ids, 1) if p in rel), None)}
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(ids, rel, k)
            row[f"ndcg@{k}"] = ndcg_at_k(ids, rel, k)
        per_claim.append(row)
    n = len(per_claim)
    for key in per_claim[0]:
        if key in ("claim_id", "first_relevant_rank"):
            continue
        agg[key] = sum(r[key] for r in per_claim) / n
    lat = sorted(r["latency_ms"] for r in per_claim)
    agg["latency_ms_median"] = statistics.median(lat)
    agg["latency_ms_p95"] = lat[min(n - 1, int(round(0.95 * (n - 1))))]
    agg["n_claims"] = n
    return {"summary": agg, "per_claim": per_claim}


def failure_examples(runs_by_method: dict[str, list[RankedList]], claims: list[dict],
                     k: int = 10, limit: int = 15) -> dict[str, list[dict]]:
    """Claims where one branch finds a relevant doc in the top-k and the other doesn't.
    These are the examples to read for the 'where lexical and semantic differ' section."""
    by_id = {c["claim_id"]: c for c in claims}
    bm = {r.claim_id: r for r in runs_by_method["bm25"]}
    sb = {r.claim_id: r for r in runs_by_method["sbert"]}
    rrf = {r.claim_id: r for r in runs_by_method.get("rrf", [])}
    out = {"bm25_only": [], "sbert_only": [], "both_miss": []}

    def hit_rank(run, rel):
        return next((i for i, r in enumerate(run.results[:k], 1) if r.paper_id in rel), None)

    for cid, c in by_id.items():
        rel = c["relevant"]
        b, s = hit_rank(bm[cid], rel), hit_rank(sb[cid], rel)
        bucket = ("bm25_only" if b and not s else "sbert_only" if s and not b
                  else "both_miss" if not b and not s else None)
        if bucket and len(out[bucket]) < limit:
            out[bucket].append({
                "claim_id": cid, "claim": c["claim"], "relevant": sorted(rel),
                "bm25_hit_rank": b, "sbert_hit_rank": s,
                "rrf_hit_rank": hit_rank(rrf[cid], rel) if cid in rrf else None,
                "bm25_top3": [r.paper_id for r in bm[cid].results[:3]],
                "sbert_top3": [r.paper_id for r in sb[cid].results[:3]],
            })
    return out


def format_table(results: dict[str, dict], index_seconds: dict[str, float],
                 ks=(1, 5, 10, 20, 100)) -> str:
    cols = ["method", "MRR"] + [f"R@{k}" for k in ks] + ["nDCG@10", "lat_med_ms", "lat_p95_ms"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for m, res in results.items():
        s = res["summary"]
        vals = [m, f"{s['mrr']:.3f}"] + [f"{s.get(f'recall@{k}', float('nan')):.3f}" for k in ks]
        vals += [f"{s['ndcg@10']:.3f}", f"{s['latency_ms_median']:.2f}", f"{s['latency_ms_p95']:.2f}"]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("Index build time (s): " + ", ".join(
        f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in index_seconds.items()))
    return "\n".join(lines)


def save_runs(path: str | Path, runs: list[RankedList], papers=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for run in runs:
            f.write(json.dumps(run.to_dict(papers)) + "\n")
