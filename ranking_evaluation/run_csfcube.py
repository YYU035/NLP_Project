"""Run Person 2 retrieval and Person 3 fusion/reranking on fixed CSFCube pools."""

import argparse
import csv
import importlib.metadata
import json
import platform
from math import isfinite
from pathlib import Path
from time import perf_counter

from paper_search.metadata import ScientificPaper
from paper_search.retrieval_models import RetrievedPaper, RetrievalResult
from retrieval.bm25 import BM25Retriever
from retrieval.sbert import SBERTRetriever
from retrieval.schema import Paper
from shared.datasets.csfcube import load, SPLITS, FACETS
from shared.datasets.setup import file_hash
from ranking_evaluation.evaluate_csfcube import evaluate
from ranking_evaluation.pipeline import ModelDescription, RankingConfig, rank_claim
from ranking_evaluation.reranker import LocalCrossEncoder
from ranking_evaluation.schemas import RankingInput


def ordered_ids(scores, ids):
    if len(scores) != len(ids) or not all(isfinite(float(score)) for score in scores):
        raise ValueError("Expected one finite retrieval score per candidate")
    return [ids[index] for index in sorted(range(len(ids)),
                                           key=lambda index: (-float(scores[index]), index))]


def run(dataset, *, encoder, model_name, scorer=None, model_description=None,
        device="cpu", top_n=20, rrf_k=60, batch_size=32, evaluator=evaluate, progress=None):
    """Score text only; judgments are consumed exclusively by the evaluator."""
    if (scorer is None) != (model_description is None):
        raise ValueError("Supply both reranker and its model description")
    synchronize = lambda: None
    if device.startswith("cuda"):
        import torch
        synchronize = lambda: torch.cuda.synchronize(device)
    elif device == "mps":
        import torch
        synchronize = torch.mps.synchronize
    documents = dataset["documents"]
    source = dataset["dataset"].lower()
    papers = [Paper(paper_id=identity, title=row["title"], abstract=row["abstract"], source=source)
              for identity, row in documents.items()]
    positions = {paper.paper_id: i for i, paper in enumerate(papers)}
    dense = SBERTRetriever(model_name=model_name, batch_size=batch_size,
                          cache_dir=None, device=device, encoder=encoder)
    synchronize()
    start = perf_counter()
    dense.index(papers, use_cache=False)
    synchronize()
    index_seconds = perf_counter() - start
    methods = {name: {} for name in ("bm25", "sbert", "rrf")}
    if scorer is not None:
        methods["rrf_cross_encoder"] = {}
    per_query = {}
    for query_index, (identity, query) in enumerate(dataset["queries"].items(), 1):
        candidate_ids, text = query["candidate_ids"], query["text"]
        lexical = BM25Retriever(stemmer="simple")
        pool = [papers[positions[pid]] for pid in candidate_ids]
        start = perf_counter()
        lexical.index(pool)
        bm25_index_seconds = perf_counter() - start
        start = perf_counter()
        bm25_ids = ordered_ids(lexical.score_all(text), candidate_ids)
        bm25_seconds = perf_counter() - start
        synchronize()
        start = perf_counter()
        all_scores = dense.score_all(text)
        sbert_ids = ordered_ids([all_scores[positions[pid]] for pid in candidate_ids], candidate_ids)
        synchronize()
        sbert_seconds = perf_counter() - start
        entries = []
        for pid in candidate_ids:
            record = documents[pid]
            paper = ScientificPaper(source=source, source_id=pid,
                                    title=record["title"] or None,
                                    abstract=" ".join(record["abstract"]) or None,
                                    landing_url=record.get("landing_url"),
                                    authors=record.get("authors", []))
            entries.append(RetrievedPaper(identity=pid, identity_aliases=[pid],
                original_claim=text, paper=paper, provenance=[], source_records=[paper]))
        inputs = RankingInput(claim_id=identity,
            candidates=RetrievalResult(original_claim=text, top_k=len(entries), papers=entries),
            bm25_ranking=bm25_ids, sbert_ranking=sbert_ids,
            bm25_seconds=bm25_seconds, sbert_seconds=sbert_seconds)
        baseline = rank_claim(inputs, config=RankingConfig(rrf_k=rrf_k))
        methods["bm25"][identity], methods["sbert"][identity] = bm25_ids, sbert_ids
        methods["rrf"][identity] = [row.entry.identity for row in baseline.papers]
        per_query[identity] = {"bm25_index_seconds": bm25_index_seconds,
                               "timings": baseline.latency.model_dump()}
        if scorer is not None:
            reranked = rank_claim(inputs, config=RankingConfig(
                rrf_k=rrf_k, rerank_enabled=True, rerank_top_n=top_n),
                scorer=scorer, model=model_description)
            methods["rrf_cross_encoder"][identity] = [row.entry.identity for row in reranked.papers]
            per_query[identity]["rerank_seconds"] = reranked.latency.rerank_seconds
            per_query[identity]["rerank_scored_count"] = reranked.rerank_scored_count
        if progress is not None:
            progress(query_index, len(dataset["queries"]), identity)
    reports = {name: evaluator(dataset, {"status": "complete", "dataset_version": dataset["version"],
               "method": name, "rankings": rankings}) for name, rankings in methods.items()}
    return {"dataset": dataset["dataset"], "version": dataset["version"],
            "task": "released pool ranking; not claim verification", "protocol": dataset["protocol"],
            "configuration": {"bm25": lexical.settings(), "sbert": dense.settings(),
                "rrf_k": rrf_k, "rerank_top_n": top_n,
                "reranker": model_description.model_dump() if model_description else None},
            "index_seconds": index_seconds, "query_count": len(per_query),
            "latency_note": "One synchronous run; no warmed throughput or total-system claim. SBERT includes query encoding and scoring all loaded docs before pool filtering.",
            "methods": reports, "rankings": methods, "per_query": per_query}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/csfcube"))
    parser.add_argument("--split", choices=SPLITS, default="fold1_dev")
    parser.add_argument("--facet", choices=[*FACETS, "all"], default="all")
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument("--encoder-id", required=True)
    parser.add_argument("--encoder-revision", required=True)
    parser.add_argument("--reranker-path", type=Path)
    parser.add_argument("--reranker-id")
    parser.add_argument("--reranker-revision")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cpu-float64", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory exists; choose a new path")
    if not args.encoder_path.is_dir():
        parser.error("Encoder path must be a provisioned local model directory")
    if args.cpu_float64 and args.device != "cpu":
        parser.error("CPU float64 requires --device cpu")
    if args.batch_size <= 0 or args.top_n < 0 or args.rrf_k < 0 or args.threads <= 0:
        parser.error("Invalid batch size, rerank depth or RRF constant")
    if args.reranker_path and not (args.reranker_id and args.reranker_revision):
        parser.error("Reranker ID/revision are required")
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(args.threads)
    start = perf_counter()
    encoder = SentenceTransformer(str(args.encoder_path), local_files_only=True,
                                  trust_remote_code=False, device=args.device)
    if args.cpu_float64:
        encoder.double()
    scorer = model = None
    if args.reranker_path:
        scorer = LocalCrossEncoder(args.reranker_path, device=args.device,
            cpu_float64=args.cpu_float64, batch_size=args.batch_size)
        model = ModelDescription(model_id=args.reranker_id, revision=args.reranker_revision,
            device=args.device, dtype=str(next(scorer.model.model.parameters()).dtype).removeprefix("torch."),
            max_length=512, batch_size=args.batch_size)
    load_seconds = perf_counter() - start
    report = run(load(args.data, split=args.split, facet=args.facet), encoder=encoder,
        model_name=args.encoder_id, scorer=scorer, model_description=model,
        device=args.device, top_n=args.top_n, rrf_k=args.rrf_k, batch_size=args.batch_size)
    report["model_load_seconds"] = load_seconds
    report["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                             "torch_threads": torch.get_num_threads(), "device": args.device}
    report["provenance"] = {"encoder_id": args.encoder_id,
        "encoder_revision": args.encoder_revision,
        "encoder_file_hashes": {p.relative_to(args.encoder_path).as_posix(): file_hash(p)
            for p in sorted(args.encoder_path.rglob("*")) if p.is_file()
            and ".cache" not in p.relative_to(args.encoder_path).parts},
        "runner_sha256": file_hash(Path(__file__)),
        "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "sentence-transformers", "transformers", "numpy")}}
    if args.reranker_path:
        report["provenance"]["reranker_file_hashes"] = {
            p.relative_to(args.reranker_path).as_posix(): file_hash(p)
            for p in sorted(args.reranker_path.rglob("*")) if p.is_file()
            and ".cache" not in p.relative_to(args.reranker_path).parts}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    rows = [{"method": name, "queries": report["query_count"],
             **value["binary_metrics"]["macro"], **value["graded_macro"]}
            for name, value in report["methods"].items()]
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output_dir / "summary.csv")


if __name__ == "__main__":
    main()
