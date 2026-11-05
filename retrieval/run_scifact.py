"""BM25 vs SBERT vs RRF on SciFact.

    python -m retrieval.run_scifact --split dev
    python -m retrieval.run_scifact --split train --sbert-model sentence-transformers/allenai-specter

Writes to retrieval/outputs/<split>_<model>/:
    results.md          comparison table (paste into the report)
    metrics.json        summary + per-claim metrics per method
    runs_<method>.jsonl ranked lists (with paper text) -> Person 3's input
    failures.json       claims where BM25 and SBERT disagree
    config.json         every setting needed to reproduce the run
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

from .data import load_scifact_claims, load_scifact_corpus
from .evaluate import evaluate_method, failure_examples, format_table, save_runs
from .retriever import build_all
from .sbert import DEFAULT_MODEL


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main(argv=None, encoder=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/scifact")
    ap.add_argument("--split", default="dev", choices=["train", "dev"],
                    help="SciFact test labels are not public; use dev for reported numbers.")
    ap.add_argument("--sbert-model", default=DEFAULT_MODEL)
    ap.add_argument("--k", type=int, default=100, help="ranked-list depth saved per claim")
    ap.add_argument("--handoff-k", type=int, default=10, help="depth written for Person 3")
    ap.add_argument("--k-rrf", type=int, default=60)
    ap.add_argument("--fusion-depth", type=int, default=100)
    ap.add_argument("--k1", type=float, default=0.9)
    ap.add_argument("--b", type=float, default=0.4)
    ap.add_argument("--cache-dir", default="retrieval/cache")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    data = Path(args.data_dir)
    papers = load_scifact_corpus(data / "corpus.jsonl")
    claims = load_scifact_claims(data / f"claims_{args.split}.jsonl")
    qrels = {c["claim_id"]: c["relevant"] for c in claims}
    print(f"{len(papers)} papers, {len(claims)} claims with evidence ({args.split})")

    retrievers = build_all(papers, sbert_model=args.sbert_model, cache_dir=args.cache_dir,
                           encoder=encoder, k1=args.k1, b=args.b,
                           k_rrf=args.k_rrf, fusion_depth=args.fusion_depth)
    index_seconds = retrievers["rrf"].index_seconds
    by_id = retrievers["rrf"].papers

    # Warm-up query so model load / first-call overhead isn't counted as latency.
    for r in retrievers.values():
        r.search(claims[0]["claim"], k=args.k)

    runs = {m: [r.search(c["claim"], k=args.k, claim_id=c["claim_id"]) for c in claims]
            for m, r in retrievers.items()}
    results = {m: evaluate_method(rs, qrels) for m, rs in runs.items()}

    tag = f"{args.split}_{args.sbert_model.split('/')[-1]}"
    out = Path(args.out_dir or f"retrieval/outputs/{tag}")
    out.mkdir(parents=True, exist_ok=True)
    table = format_table(results, index_seconds)
    (out / "results.md").write_text(f"# SciFact {args.split} retrieval\n\n{table}\n")
    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    (out / "failures.json").write_text(json.dumps(failure_examples(runs, claims), indent=1))
    for m, rs in runs.items():
        for r in rs:
            r.results = r.results[: args.handoff_k] if m == "rrf" else r.results
        save_runs(out / f"runs_{m}.jsonl", rs, papers=by_id if m == "rrf" else None)
    (out / "config.json").write_text(json.dumps({
        "args": vars(args), "settings": {m: r.settings() for m, r in retrievers.items()},
        "index_seconds": index_seconds, "n_papers": len(papers), "n_claims": len(claims),
        "git_commit": git_commit(), "python": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, indent=1, default=str))
    print(table)
    print(f"\nSaved to {out}/")
    return results


if __name__ == "__main__":
    main()
