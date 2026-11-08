"""CLI for replaying fixed rankings; does not run BM25 or SBERT itself."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ranking_evaluation.benchmark import BenchmarkBundle
from ranking_evaluation.latency import LatencyRecord, measure_call
from ranking_evaluation.metrics import evaluate_rankings
from ranking_evaluation.reranker import LocalCrossEncoder, PairScorer, rerank
from ranking_evaluation.rrf import reciprocal_rank_fusion


def run_benchmark(
    bundle: BenchmarkBundle, *, ks: tuple[int, ...] = (5, 10, 20),
    rrf_k: int = 60, scorer: PairScorer | None = None, top_n: int = 20,
) -> dict:
    """Keep identical claims, gold labels and candidate pools across methods.

    Full RRF union is retained, so cutoff metrics and reranked prefix/tail are
    comparable. Branch timings are upstream observations, not replay timings.
    """
    bundle = BenchmarkBundle.model_validate(bundle.model_dump())
    methods = {name: {} for name in ("bm25", "sbert", "rrf")}
    if scorer is not None:
        methods["rrf_cross_encoder"] = {}
    results, latencies = {}, {}
    for claim in bundle.claims:
        fused, fusion_seconds = measure_call(
            lambda: reciprocal_rank_fusion(claim, rrf_k=rrf_k),
        )
        methods["bm25"][claim.claim_id] = claim.bm25_ranking
        methods["sbert"][claim.claim_id] = claim.sbert_ranking
        methods["rrf"][claim.claim_id] = [row.entry.identity for row in fused]
        results[claim.claim_id] = {
            "rrf": [row.model_dump(mode="json") for row in fused],
        }
        rerank_seconds = None
        if scorer is not None:
            reranked, rerank_seconds = measure_call(lambda: rerank(
                fused, claim=claim.candidates.original_claim,
                scorer=scorer, top_n=top_n,
            ))
            methods["rrf_cross_encoder"][claim.claim_id] = [
                row.entry.identity for row in reranked
            ]
            results[claim.claim_id]["rrf_cross_encoder"] = [
                row.model_dump(mode="json") for row in reranked
            ]
        latencies[claim.claim_id] = LatencyRecord(
            bm25_seconds=claim.bm25_seconds, sbert_seconds=claim.sbert_seconds,
            rrf_seconds=fusion_seconds, rerank_seconds=rerank_seconds,
        ).model_dump()
    return {
        "dataset": bundle.model_dump(exclude={"claims", "qrels", "corpus_doc_ids"}),
        "corpus_size": len(bundle.corpus_doc_ids),
        "configuration": {"ks": ks, "rrf_k": rrf_k,
                          "rerank_enabled": scorer is not None, "top_n": top_n},
        "timing_note": "Synchronous ranking replay; CUDA scorer synchronizes inference; no live search, retrieval replay, or end-to-end timing",
        "branch_depths": {claim.claim_id: {
            "bm25": len(claim.bm25_ranking), "sbert": len(claim.sbert_ranking),
        } for claim in bundle.claims},
        "metrics": {name: evaluate_rankings(rankings, bundle.qrels, ks=ks)
                    for name, rankings in methods.items()},
        "rankings": methods, "ranked_papers": results, "latency": latencies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON report path; existing files are never overwritten")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cpu-float64", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    if args.model_path and not (args.model_id and args.model_revision):
        parser.error("Reranking requires --model-id and --model-revision for provenance")
    if not args.model_path and (args.model_id or args.model_revision or args.cpu_float64):
        parser.error("Model options require --model-path")
    if args.cpu_float64 and args.device != "cpu":
        parser.error("--cpu-float64 requires --device cpu")
    if args.device != "cpu" and not args.model_path:
        parser.error("--device cuda requires --model-path")
    raw = args.input.read_bytes()
    bundle = BenchmarkBundle.model_validate_json(raw)
    scorer, load_seconds = None, None
    if args.model_path:
        scorer, load_seconds = measure_call(lambda: LocalCrossEncoder(
            args.model_path, device=args.device, max_length=args.max_length,
            batch_size=args.batch_size, cpu_float64=args.cpu_float64,
        ))
    report = run_benchmark(bundle, ks=tuple(args.ks), rrf_k=args.rrf_k,
                           scorer=scorer, top_n=args.top_n)
    root = Path(__file__).resolve().parents[1]
    source_hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in (root / "ranking_evaluation", root / "live_search/src/paper_search")
        for path in sorted(folder.rglob("*.py"))
    }
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              capture_output=True, text=True, check=False)
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                capture_output=True, text=True, check=False)
    report["provenance"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "git_head": git_head.stdout.strip() if git_head.returncode == 0 else None,
        "git_dirty": bool(git_status.stdout) if git_status.returncode == 0 else None,
        "source_sha256": source_hashes,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {dist.metadata["Name"]: dist.version
                     for dist in importlib.metadata.distributions()},
        "model": {"id": args.model_id, "revision": args.model_revision,
                  "identity_note": "ID/revision supplied by caller; local file hashes identify actual bytes",
                  "files_sha256": {
                      path.relative_to(args.model_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted(args.model_path.rglob("*"))
                      if path.is_file() and ".cache" not in path.relative_to(args.model_path).parts
                  },
                  "device": args.device, "dtype": "float64" if args.cpu_float64 else "default",
                  "max_length": args.max_length, "batch_size": args.batch_size,
                  "load_seconds": load_seconds} if scorer else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
