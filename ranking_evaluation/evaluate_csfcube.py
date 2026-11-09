"""Evaluate real CSFCube pool rankings without inventing verification verdicts."""

import argparse
import json
from math import log2
from pathlib import Path

from ranking_evaluation.metrics import evaluate_rankings
from shared.datasets.csfcube import load, SPLITS, FACETS
from shared.datasets.setup import file_hash


def graded_ndcg(ranking: list[str], grades: dict[str, int], k: int) -> float | None:
    """Linear gain / log2(rank+1); supplemental metric, not official scorer parity."""
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if len(ranking) != len(set(ranking)) or set(ranking) - grades.keys():
        raise ValueError("Ranking contains duplicate or unjudged IDs")
    if any(type(g) is not int or g not in range(4) for g in grades.values()):
        raise ValueError("Grades must be integers from 0 to 3")
    ideal = sum(g / log2(i + 2) for i, g in enumerate(sorted(grades.values(), reverse=True)[:k]))
    return (sum(grades[pid] / log2(i + 2) for i, pid in enumerate(ranking[:k])) / ideal
            if ideal else None)


def evaluate(dataset: dict, export: dict, *, ks=(5, 10, 20), relevant_min_grade=2) -> dict:
    if export.get("status") != "complete" or export.get("dataset_version") != dataset["version"]:
        raise ValueError("Expected a completed export for the same dataset version/protocol")
    if type(relevant_min_grade) is not int or relevant_min_grade not in (1, 2, 3):
        raise ValueError("relevant_min_grade must be 1, 2 or 3")
    if not isinstance(export.get("method"), str) or not export["method"].strip():
        raise ValueError("Method provenance is required")
    rankings = export["rankings"]
    if set(rankings) != set(dataset["queries"]):
        raise ValueError("Every selected query must appear exactly once")
    for identity, ranking in rankings.items():
        if not isinstance(ranking, list) or len(ranking) != len(set(ranking)):
            raise ValueError("Rankings must be lists of unique IDs")
        if set(ranking) - set(dataset["queries"][identity]["candidate_ids"]):
            raise ValueError("A ranked paper is outside the released query pool")
    qrels = {identity: [pid for pid, grade in grades.items() if grade >= relevant_min_grade]
             for identity, grades in dataset["grades"].items()}
    binary = evaluate_rankings(rankings, qrels, ks=ks)
    per_query = {identity: {f"graded_nDCG@{k}": graded_ndcg(rankings[identity], grades, k)
                            for k in ks} for identity, grades in dataset["grades"].items()}
    macro = {}
    for k in ks:
        name = f"graded_nDCG@{k}"
        values = [row[name] for row in per_query.values() if row[name] is not None]
        macro[name] = sum(values) / len(values) if values else None
    return {"dataset": "CSFCube", "version": dataset["version"],
            "task": "released candidate-pool ranking; not claim verification or open retrieval",
            "method": export["method"], "protocol": dataset["protocol"],
            "relevant_min_grade": relevant_min_grade,
            "metric_note": "Supplemental metrics: binary threshold plus linear-gain log2 graded nDCG; not claimed identical to official aggregation",
            "query_count": len(rankings), "binary_metrics": binary,
            "graded_macro": macro, "graded_per_query": per_query,
            "excluded_no_positive_grade": sum(not any(g > 0 for g in values.values())
                                               for values in dataset["grades"].values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/csfcube"))
    parser.add_argument("--split", choices=SPLITS, default="fold1_dev")
    parser.add_argument("--facet", choices=[*FACETS, "all"], default="all")
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--relevant-min-grade", type=int, choices=[1, 2, 3], default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path")
    report = evaluate(load(args.data, split=args.split, facet=args.facet),
                      json.loads(args.rankings.read_text()), ks=args.ks,
                      relevant_min_grade=args.relevant_min_grade)
    report["provenance"] = {"ranking_file_sha256": file_hash(args.rankings),
                            "evaluator_sha256": file_hash(Path(__file__)),
                            "adapter_sha256": file_hash(Path(__file__).parents[1] / "shared/datasets/csfcube.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
