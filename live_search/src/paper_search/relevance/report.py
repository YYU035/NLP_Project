"""Join blinded labels to provenance only after judging; no retrieval calls."""

import json
from pathlib import Path

from paper_search.relevance.cache import Cache
from paper_search.relevance.dataset import cache_key, csv_write


def generate_report(
    manifest: dict, cache: Cache, output: Path
) -> dict:
    """Report both label thresholds; incomplete metrics stay null, not zero."""
    judgments = {}
    exported = []
    for pair in manifest["pairs"]:
        item = cache.get(cache_key(pair, manifest["model"]))
        if item and item[0] == "done":
            judgments[(pair["claim_id"], pair["identity"])] = item[
                1
            ]["label"]
        exported.append(
            {
                "id": pair["id"],
                "claim_id": pair["claim_id"],
                "identity": pair["identity"],
                "missing_abstract": pair["missing_abstract"],
                "status": item[0] if item else "pending",
                "label": item[1]["label"]
                if item and item[0] == "done"
                else "",
                "justification": item[1].get("justification", "")
                if item and item[1]
                else "",
                "quote": item[1].get("quote", "")
                if item and item[1]
                else "",
            }
        )
    csv_write(output / "judgments.csv", exported)
    metrics = []
    comparisons = []
    rows = manifest["rows"]
    calls = manifest["experiment"]["calls"]
    providers = sorted({c["provider"] for c in calls})
    for provider in [*providers, "combined"]:
        selected = [
            r
            for r in rows
            if provider == "combined" or r["provider"] == provider
        ]
        relevant_sets = {}
        for threshold in (1, 2):
            for arm in ("original", "alternative"):
                arm_rows = [r for r in selected if r["arm"] == arm]
                keys = {
                    (r["claim_id"], r["identity"]) for r in arm_rows
                }
                complete = keys.issubset(judgments)
                relevant = {
                    key
                    for key in keys
                    if judgments.get(key, -1) >= threshold
                }
                relevant_sets[(arm, threshold)] = (
                    relevant,
                    complete,
                )
                pdf = {
                    (r["claim_id"], r["identity"])
                    for r in arm_rows
                    if json.loads(r["pdf_urls"] or "[]")
                }
                valid_calls = [
                    c
                    for c in calls
                    if c["kind"] == arm
                    and (
                        provider == "combined"
                        or c["provider"] == provider
                    )
                    and c["status"] in {"success", "zero_results"}
                ]
                failures = sum(
                    c["status"] not in {"success", "zero_results"}
                    for c in calls
                    if c["kind"] == arm
                    and (
                        provider == "combined"
                        or c["provider"] == provider
                    )
                )
                per_claim = []
                for call in valid_calls:
                    subset = [
                        r
                        for r in arm_rows
                        if r["claim_id"] == call["claim_id"]
                        and r["provider"] == call["provider"]
                    ]
                    ids = {
                        (r["claim_id"], r["identity"])
                        for r in subset
                    }
                    score = (
                        len(ids & relevant) / 10
                        if ids.issubset(judgments)
                        else None
                    )
                    if provider != "combined":
                        per_claim.append(
                            {
                                "claim_id": call["claim_id"],
                                "precision_at_10": score,
                            }
                        )
                precision = (
                    sum(p["precision_at_10"] for p in per_claim)
                    / len(per_claim)
                    if per_claim
                    and all(
                        p["precision_at_10"] is not None
                        for p in per_claim
                    )
                    else None
                )
                metrics.append(
                    {
                        "provider": provider,
                        "arm": arm,
                        "label_threshold": threshold,
                        "complete_for_retrieved_pairs": complete,
                        "assessed_pairs": len(
                            keys & judgments.keys()
                        ),
                        "retrieved_pairs": len(keys),
                        "provider_failures": failures,
                        "precision_at_10": precision,
                        "valid_queries": len(valid_calls),
                        "per_claim_precision": per_claim,
                        "relevant_pair_yield": len(relevant)
                        if complete
                        else None,
                        "relevant_pairs_with_pdf_url": len(
                            relevant & pdf
                        )
                        if complete
                        else None,
                        "relevant_pdf_fraction": len(relevant & pdf)
                        / len(relevant)
                        if complete and relevant
                        else None,
                    }
                )
            left, complete_left = relevant_sets[
                ("original", threshold)
            ]
            right, complete_right = relevant_sets[
                ("alternative", threshold)
            ]
            complete = complete_left and complete_right
            comparisons.append(
                {
                    "provider": provider,
                    "label_threshold": threshold,
                    "complete": complete,
                    "relevant_concept_additions": len(right - left)
                    if complete
                    else None,
                    "relevant_original_displaced": len(left - right)
                    if complete
                    else None,
                    "net_relevant_yield_change": len(right)
                    - len(left)
                    if complete
                    else None,
                    "relevant_overlap": len(left & right)
                    if complete
                    else None,
                    "two_query_union_yield": len(left | right)
                    if complete
                    else None,
                }
            )
    result = {
        "model": manifest["model"],
        "judged_pairs": len(judgments),
        "total_pairs": len(manifest["pairs"]),
        "metrics": metrics,
        "displacement": comparisons,
        "interpretation": "LLM-estimated relevance, not ground-truth accuracy",
    }
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    flat = [
        {k: v for k, v in m.items() if k != "per_claim_precision"}
        for m in metrics
    ]
    csv_write(output / "metrics.csv", flat)
    csv_write(output / "displacement.csv", comparisons)

    def display(value: object) -> str:
        return "pending" if value is None else str(value)

    lines = [
        "# Blinded Gemini relevance assessment",
        "",
        f"Model: `{manifest['model']}`. Judged: **{len(judgments)} / "
        f"{len(manifest['pairs'])}** pairs.",
        "",
        "LLM estimates only. Labels 1+2 measure topical relevance; label 2 "
        "measures potential evidence, including contradictions. A judgment "
        "does not establish claim truth. Missing abstracts remain flagged.",
        "",
        "| Provider | Arm | Threshold | P@10 | Relevant yield | "
        "Relevant PDFs |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        lines.append(
            f"| {metric['provider']} | {metric['arm']} | "
            f"{metric['label_threshold']} | "
            f"{display(metric['precision_at_10'])} | "
            f"{display(metric['relevant_pair_yield'])} | "
            f"{display(metric['relevant_pairs_with_pdf_url'])} |"
        )
    lines.extend(
        [
            "",
            "P@10 uses ten slots per valid provider query and macro-averages "
            "over claims, including valid empty results as zero. "
            "Provider failures are excluded and counted separately. "
            "Unknown labels are never zero. Combined-provider P@10 is "
            "undefined because there is no shared ranked top ten. "
            "Combined yield deduplicates claim-paper pairs across sources.",
            "",
            "Compare original and alternative at their matched request "
            "budgets (ten requests per provider arm in the saved experiment). "
            "The union costs twice as many requests and may include "
            "twenty papers per claim/provider; it is not an equal-cost gain. "
            "Displacement and union counts are in displacement.csv. "
            "PDFs are metadata URLs, not verified downloads. "
            "There is no corpus Recall@K without exhaustive "
            "relevance labels. These LLM labels require human validation.",
            "",
            "manual_validation_30.csv contains three fixed examples per "
            "claim without provenance or LLM predictions. Use two independent "
            "human assessors, adjudicate disagreements, "
            "then compare agreement, "
            "weighted Cohen's kappa and especially false label-2 judgments. "
            "This small sample cannot establish reliability "
            "for every discipline.",
        ]
    )
    (output / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result
