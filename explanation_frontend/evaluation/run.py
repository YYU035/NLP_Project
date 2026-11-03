"""Paired, blinded evaluation on identical frozen contexts.

python -m explanation_frontend.evaluation.run --mode fixture
python -m explanation_frontend.evaluation.run --mode llm --cases <manifest.jsonl>
python -m explanation_frontend.evaluation.run --reviews <review.csv>
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import statistics
import time

from ..export import import_snapshot
from ..llm import OpenAIClient, generate_llm
from ..models import make_context
from ..providers import FixtureProvider, replay_aggregate
from ..templates import build_template
from ..validation import validate_explanation

MODULE = Path(__file__).resolve().parents[1]
REVIEW_FIELDS = ["output_id", "reviewer", "factual_supported", "factual_total", "citation_supported", "citation_total",
                 "citation_complete", "citation_required", "verdict_consistent", "scope_preserved",
                 "correction_supported", "correction_total", "usefulness", "notes"]


class FixtureLLM:
    """A scripted generator for harness tests, explicitly not an actual model."""
    model = "scripted-fixture-not-an-llm"

    def __init__(self, context, fault=None):
        self.context, self.fault = context, fault

    def complete(self, messages):
        data = {"verdict": self.context.aggregation.aggregated_verdict,
                "statements": [{"text": p.text, "citation_ids": [p.passage_id]}
                               for p in self.context.aggregation.primary_evidence], "correction": None}
        if self.fault == "numbers" and data["statements"]:
            data["statements"][0]["text"] = data["statements"][0]["text"].replace("12%", "20%")
        elif self.fault == "citation" and data["statements"]:
            data["statements"][0]["citation_ids"] = ["fabricated"]
        elif self.fault == "verdict":
            data["verdict"] = "SUPPORTED" if data["verdict"] != "SUPPORTED" else "CONTRADICTED"
        elif self.fault == "invalid":
            return "Invalid JSON fixture"
        elif self.fault == "timeout":
            raise TimeoutError("Synthetic provider failure")
        return json.dumps(data, ensure_ascii=False)


def run_study(cases: list[dict], *, mode: str, seed: int = 19) -> dict:
    if mode not in {"fixture", "llm"}:
        raise ValueError("Unknown evaluation mode")
    ids = [case["case_id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case IDs")
    rng, packets, keys, records = random.Random(seed), [], [], []
    client = OpenAIClient.from_env() if mode == "llm" else None
    for index, case in enumerate(cases, 1):
        if "snapshot_file" in case:
            bundle = import_snapshot(Path(case["snapshot_file"]).read_bytes())
            snapshot = bundle.snapshot
            selected = case.get("selected", bundle.selected_paper_ids)
            aggregate = bundle.aggregation if sorted(selected) == bundle.selected_paper_ids else replay_aggregate(snapshot, selected)
        else:
            snapshot = FixtureProvider().run_fixture(case["fixture"])
            selected = case.get("selected", list(snapshot.papers))
            aggregate = replay_aggregate(snapshot, selected)
        context = make_context(snapshot, selected, aggregate)
        started = time.monotonic()
        template = build_template(context)
        template_seconds = time.monotonic() - started
        generated = generate_llm(context, client or FixtureLLM(context, case.get("fault")))
        variants = [("template", template.model_dump_json(), template, template_seconds),
                    ("llm", generated.raw or "[generation unavailable]", generated.displayed, generated.elapsed_seconds)]
        rng.shuffle(variants)
        for letter, (method, raw, displayed, elapsed) in zip("AB", variants):
            output_id = f"case-{index:03d}-{letter}"
            packets.append({"output_id": output_id, "case_id": case["case_id"], "claim": snapshot.claim,
                "authoritative_verdict": aggregate.aggregated_verdict,
                "evidence": [p.model_dump() for p in context.aggregation.primary_evidence], "raw_output": raw})
            keys.append({"output_id": output_id, "method": method})
            records.append({"output_id": output_id, "case_id": case["case_id"], "dataset": case.get("dataset", "unspecified"),
                "snapshot_hash": snapshot.fingerprint, "context_key": context.key, "method": method,
                "generation_status": generated.status if method == "llm" else "template",
                "raw": raw, "displayed": displayed.model_dump(mode="json"),
                "display_validation": validate_explanation(displayed, context).model_dump(),
                "generation_validation": generated.validation.model_dump() if method == "llm" and generated.validation else None,
                "elapsed_seconds": elapsed, "settings": generated.settings if method == "llm" else {},
                "model": generated.model if method == "llm" else None, "prompt_version": generated.prompt_version if method == "llm" else None,
                "issues": generated.issues if method == "llm" else []})
    statuses = dict(Counter(r["generation_status"] for r in records if r["method"] == "llm"))
    return {"summary": {"mode": mode, "seed": seed, "cases": len(cases), "outputs": len(records),
                "datasets": dict(Counter(case.get("dataset", "unspecified") for case in cases)),
                "llm_statuses": statuses, "fallback_count": sum(v for k, v in statuses.items() if k != "accepted"),
                "human_reviews_completed": 0, "note": "Scripted fixture results validate the harness only." if mode == "fixture" else "Human support/usefulness scores pending review."},
            "review_packets": packets, "private_key": keys, "records": records}


def summarize_reviews(rows: list[dict], method_key: list[dict] | None = None) -> dict:
    complete = [row for row in rows if row.get("usefulness", "").strip()]
    if len({(r["output_id"], r["reviewer"]) for r in complete}) != len(complete):
        raise ValueError("Duplicate output/reviewer score")
    for row in complete:
        for numerator, denominator in [("factual_supported", "factual_total"), ("citation_supported", "citation_total"),
                                        ("citation_complete", "citation_required"), ("correction_supported", "correction_total")]:
            if not 0 <= int(row[numerator]) <= int(row[denominator]):
                raise ValueError("Invalid review counts")
        if int(row["verdict_consistent"]) not in {0, 1} or int(row["scope_preserved"]) not in {0, 1} or not 1 <= int(row["usefulness"]) <= 5:
            raise ValueError("Invalid rating")
    def ratio(a, b):
        numerator, denominator = sum(int(r[a]) for r in complete), sum(int(r[b]) for r in complete)
        return {"numerator": numerator, "denominator": denominator, "rate": numerator / denominator if denominator else None}
    summary = {"completed_reviews": len(complete), "pending_reviews": len(rows) - len(complete),
            "factual_support": ratio("factual_supported", "factual_total"),
            "citation_support": ratio("citation_supported", "citation_total"),
            "citation_completeness": ratio("citation_complete", "citation_required"),
            "correction_support": ratio("correction_supported", "correction_total"),
            "verdict_consistency": {"numerator": sum(int(r["verdict_consistent"]) for r in complete), "denominator": len(complete)},
            "scope_preserved": {"numerator": sum(int(r["scope_preserved"]) for r in complete), "denominator": len(complete)},
            "usefulness_mean": statistics.mean(int(r["usefulness"]) for r in complete) if complete else None,
            "usefulness_counts": dict(Counter(r["usefulness"] for r in complete))}
    if method_key is not None:
        methods = {entry["output_id"]: entry["method"] for entry in method_key}
        if any(row["output_id"] not in methods for row in rows):
            raise ValueError("Review output ID missing from private method key")
        summary["by_method"] = {method: summarize_reviews([row for row in rows if methods[row["output_id"]] == method])
                                for method in sorted(set(methods.values()))}
    paired = {}
    for row in complete:
        paired.setdefault(row["output_id"], []).append(row)
    pairs = [reviews for reviews in paired.values() if len(reviews) == 2]
    summary["verdict_reviewer_agreement"] = {
        "agreements": sum(pair[0]["verdict_consistent"] == pair[1]["verdict_consistent"] for pair in pairs),
        "paired_outputs": len(pairs)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fixture", "llm"], default="fixture")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.jsonl"))
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--key", type=Path, help="Private method key; defaults to private_key.jsonl beside the review CSV")
    parser.add_argument("--out", type=Path, default=MODULE / "outputs" / "evaluation")
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    output = args.out.resolve()
    if not output.is_relative_to(MODULE):
        parser.error("Output must remain within explanation_frontend")
    output.mkdir(parents=True, exist_ok=True)
    if args.reviews:
        with args.reviews.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        key_path = args.key or args.reviews.with_name("private_key.jsonl")
        method_key = [json.loads(line) for line in key_path.read_text(encoding="utf-8").splitlines() if line.strip()] if key_path.exists() else None
        summary = summarize_reviews(rows, method_key=method_key)
        (output / "human_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
        # Resolve real snapshot inputs relative to the manifest, not the execution directory.
        for case in cases:
            if "snapshot_file" in case:
                case["snapshot_file"] = str((args.cases.parent / case["snapshot_file"]).resolve())
        result = run_study(cases, mode=args.mode, seed=args.seed)
        summary = result["summary"]
        for key in ("records", "review_packets", "private_key"):
            (output / f"{key}.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in result[key]) + "\n", encoding="utf-8")
        with (output / "review.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            for packet in result["review_packets"]:
                for reviewer in ("r1", "r2"):
                    writer.writerow({"output_id": packet["output_id"], "reviewer": reviewer})
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
