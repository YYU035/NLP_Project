"""Rank Person 1's live-search candidates for one claim.

    python -m retrieval.search_live --papers live_search/snapshots/abc123.json \
        --claim "Vitamin D supplementation reduces fracture risk" --method rrf --k 10

Python use (what the integrated pipeline will call):
    from retrieval.search_live import rank_candidates
    ranked = rank_candidates(claim, paper_records, method="rrf", k=10)
    ranked.to_dict(papers)  # -> Person 3

Note: live candidate sets are small (tens of papers), so BM25's IDF is
computed over very few docs and is noisier than on SciFact's 5k corpus.
That's one reason the live branch is evaluated separately by manual review.
"""
from __future__ import annotations

import argparse
import json

from .data import load_live_papers
from .retriever import Retriever
from .schema import RankedList


def rank_candidates(claim: str, records, method: str = "rrf", k: int = 10,
                    claim_id: str = "live", sbert_model: str | None = None,
                    encoder=None) -> tuple[RankedList, dict]:
    """Returns (RankedList, papers_by_id). Papers without any text are kept out
    of the index but reported so Person 5 can show them as 'unavailable'."""
    from .sbert import DEFAULT_MODEL, SBERTRetriever
    papers = load_live_papers(records)
    indexable = [p for p in papers if p.index_text()]
    skipped = [p.paper_id for p in papers if not p.index_text()]
    sbert = (SBERTRetriever(sbert_model or DEFAULT_MODEL, cache_dir=None, encoder=encoder)
             if method in ("sbert", "rrf") else None)
    r = Retriever(method, sbert=sbert, corpus="live",
                  fusion_depth=max(len(indexable), 1)).index(indexable, use_cache=False)
    ranked = r.search(claim, k=k, claim_id=claim_id)
    ranked.settings["skipped_no_text"] = skipped
    ranked.settings["n_candidates"] = len(papers)
    return ranked, r.papers


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", required=True, help="JSON/JSONL of Person 1's paper records")
    ap.add_argument("--claim", required=True)
    ap.add_argument("--method", default="rrf", choices=["bm25", "sbert", "rrf"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sbert-model", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    ranked, papers = rank_candidates(args.claim, args.papers, args.method, args.k,
                                     sbert_model=args.sbert_model)
    d = ranked.to_dict(papers)
    for r in d["results"]:
        print(f"{r['rank']:>2}  {r['score']:.4f}  {r['paper_id']}  {r['paper']['title'][:80]}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)


if __name__ == "__main__":
    main()
