"""Score binary feasibility predictions; never reinterpret them as NLI labels."""

import argparse
import json
from datetime import date
from pathlib import Path

from shared.datasets.matter_of_fact import FILES, load
from shared.datasets.setup import file_hash


def evaluate(dataset: dict, export: dict) -> dict:
    if export.get("status") != "complete" or export.get("dataset_version") != dataset["version"]:
        raise ValueError("Expected completed predictions for the same dataset version/scope")
    if not isinstance(export.get("model"), str) or not export["model"].strip():
        raise ValueError("Model/revision/protocol description is required")
    records = export["predictions"]
    mapping = {row["case_id"]: row for row in records}
    if len(mapping) != len(records) or set(mapping) != set(dataset["gold"]):
        raise ValueError("Predictions must cover all selected case IDs exactly once")
    counts = {"true_positive": 0, "false_positive": 0, "true_negative": 0,
              "false_negative": 0, "abstained_positive": 0, "abstained_negative": 0}
    missing_evidence = 0
    per_case = {}
    for identity, gold in dataset["gold"].items():
        row = mapping[identity]
        predicted, actual = row["prediction"], gold["label"]
        if predicted is not None and type(predicted) is not bool:
            raise ValueError("Prediction must be boolean feasibility or null abstention")
        evidence = row.get("evidence", [])
        if not isinstance(evidence, list):
            raise ValueError("Evidence must be a list")
        cutoff = date.fromisoformat(gold["evidence_not_after"])
        for source in evidence:
            if not isinstance(source.get("reference"), str) or not source["reference"].strip():
                raise ValueError("Evidence reference is required")
            if date.fromisoformat(source["published_date"]) > cutoff:
                raise ValueError("Evidence postdates the feasibility cutoff")
        if not evidence:
            missing_evidence += 1
        if predicted is None:
            key = "abstained_positive" if actual else "abstained_negative"
        else:
            key = ("true_positive" if predicted else "false_negative") if actual else (
                "false_positive" if predicted else "true_negative")
        counts[key] += 1
        per_case[identity] = {"correct": predicted is not None and predicted == actual,
                              "abstained": predicted is None, "evidence_count": len(evidence)}
    tp, fp, tn, fn = [counts[k] for k in ("true_positive", "false_positive", "true_negative", "false_negative")]
    def f1(correct, wrong_positive, wrong_negative):
        denominator = 2 * correct + wrong_positive + wrong_negative
        return 2 * correct / denominator if denominator else 0.0
    total = len(mapping)
    if not total:
        raise ValueError("No cases to evaluate")
    return {"dataset": "Matter-of-Fact", "version": dataset["version"],
            "task": "binary feasibility; not Supported/Contradicted/NEI",
            "model": export["model"], "protocol": dataset["protocol"],
            "cases": total, "accuracy": (tp + tn) / total,
            "macro_f1": (f1(tp, fp, fn + counts["abstained_positive"]) +
                         f1(tn, fn, fp + counts["abstained_negative"])) / 2,
            "metric_policy": "Fixed two-class macro F1; abstentions incorrect and included in gold-class false negatives",
            "counts": counts, "per_case": per_case,
            "cases_without_evidence": missing_evidence,
            "temporal_check": "self-reported citation dates checked; publication dates and model training provenance not independently verified",
            "formal_temporal_protocol_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/matter_of_fact"))
    parser.add_argument("--split", choices=list(FILES), default="validation")
    parser.add_argument("--scope", choices=["eee", "all"], default="eee")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path")
    report = evaluate(load(args.data, split=args.split, scope=args.scope),
                      json.loads(args.predictions.read_text()))
    report["provenance"] = {"predictions_sha256": file_hash(args.predictions),
                            "evaluator_sha256": file_hash(Path(__file__)),
                            "adapter_sha256": file_hash(Path(__file__).parents[1] / "shared/datasets/matter_of_fact.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
