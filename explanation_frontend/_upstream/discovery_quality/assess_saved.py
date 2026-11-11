"""Assess saved retrieval outputs offline; never contact providers."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from .abstract_quality import (
    AbstractQualityAssessment, AbstractQualityRecord, assess_retrieval,
)
from paper_search.retrieval import merge_records
from paper_search.retrieval_models import RetrievalResult


def evaluate(input_dir: Path, output_dir: Path, depths: list[int]) -> dict:
    """Write a separate report; refuse existing outputs and input overlap."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("Outputs must be separate from saved input artifacts")
    files = sorted(input_dir.glob("c[0-9][0-9].json"))
    if not files:
        raise ValueError("No saved cNN.json retrieval results found")
    rows = []
    manifest = []
    for path in files:
        payload = path.read_bytes()
        saved = RetrievalResult.model_validate_json(payload)
        # Detect any future schema normalization that could change saved text.
        raw = json.loads(payload)
        for provider, run in saved.providers.items():
            for parsed, original in zip(
                run.records, raw["providers"][provider]["records"], strict=True
            ):
                if parsed.paper.abstract != original["paper"]["abstract"]:
                    raise ValueError(
                        f"Abstract changed during loading: {path}"
                    )
        for parsed, original in zip(saved.papers, raw["papers"], strict=True):
            if parsed.paper.abstract != original["paper"]["abstract"]:
                raise ValueError(f"Merged abstract changed loading: {path}")
        manifest.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        for depth in depths:
            if depth <= 0 or depth > saved.top_k:
                raise ValueError(
                    "Depth must be within the saved retrieval limit"
                )
            result = saved.model_copy(deep=True)
            if depth != saved.top_k:
                # Reuse existing merge rules on the prefix, avoiding metadata
                # leakage from papers found only below the selected depth.
                for run in result.providers.values():
                    run.records = [r for r in run.records if r.rank <= depth]
                result.papers = merge_records(
                    result.original_claim,
                    [
                        (provider, record)
                        for provider, run in result.providers.items()
                        for record in run.records
                    ],
                )
            result.top_k = depth
            before = result.model_dump_json()
            for record in assess_retrieval(result, claim_id=path.stem):
                rows.append({"top_k": depth, **record.model_dump()})
            if result.model_dump_json() != before:
                raise AssertionError("Assessment mutated retrieval results")
    counts = []
    for depth in depths:
        for scope in ("provider_record", "deduplicated"):
            subset = [
                r for r in rows if r["top_k"] == depth and r["scope"] == scope
            ]
            providers = sorted({r["provider"] for r in subset})
            for provider in ["ALL", *providers]:
                selected = [
                    r
                    for r in subset
                    if provider == "ALL" or r["provider"] == provider
                ]
                counts.append(
                    {
                        "top_k": depth,
                        "scope": scope,
                        "provider": provider,
                        "total": len(selected),
                        "nonempty_before": sum(
                            bool((r["assessment"]["text"] or "").strip())
                            for r in selected
                        ),
                        "categories": dict(
                            Counter(
                                r["assessment"]["status"] for r in selected
                            )
                        ),
                    }
                )
    report = {"inputs": manifest, "counts": counts, "assessments": rows}
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "assessment.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "assessment.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        flat = []
        for row in rows:
            item = {k: v for k, v in row.items() if k != "assessment"}
            item.update(row["assessment"])
            for key in ("identity_aliases", "provenance", "reason_codes"):
                item[key] = json.dumps(item[key], ensure_ascii=False)
            flat.append(item)
        fieldnames = [
            "top_k",
            *[name for name in AbstractQualityRecord.model_fields
              if name != "assessment"],
            *AbstractQualityAssessment.model_fields,
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)
    lines = [
        "# Offline abstract-quality review",
        "",
        "Heuristic flags only: likely_valid does not verify completeness, "
        "relevance or evidence sufficiency. No abstracts or retrieval outputs "
        "were edited, filtered or reranked. No live requests were made.",
        "",
        "Top-10 prefixes are remerged using the existing deduplication rules; "
        "top-30 uses the final saved merged results. Counts across depths "
        "must not be added together. Provider records are occurrences; "
        "deduplicated rows are unique claim–paper pairs within each claim.",
        "",
        "## Availability versus heuristic categories",
        "",
        "For merged rows, provider denotes the selected record's source, "
        "not an exclusive provider contribution.",
        "",
        "| Depth | Scope | Provider | Total | Nonempty before | Missing | "
        "Likely valid | Incomplete | Body-like | Uncertain |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for count in counts:
        categories = count["categories"]
        values = [
            categories.get(k, 0)
            for k in (
                "missing",
                "likely_valid",
                "suspected_incomplete",
                "body_like",
                "uncertain",
            )
        ]
        lines.append(
            f"| {count['top_k']} | {count['scope']} | {count['provider']} "
            f"| {count['total']} | {count['nonempty_before']} | "
            + " | ".join(map(str, values))
            + " |"
        )
    lines += [
        "",
        "## Nonempty flagged examples",
        "",
        "| Depth | Scope | Claim | Source ID | Title | Status | Reasons |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        assessment = row["assessment"]
        if assessment["status"] in {"missing", "likely_valid"}:
            continue
        title = (row["title"] or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {row['top_k']} | {row['scope']} | {row['claim_id']} "
            f"| {row['source_id']} | {title} | {assessment['status']} "
            f"| {', '.join(assessment['reason_codes'])} |"
        )
    lines += [
        "",
        "## Limitations",
        "",
        "- No human quality labels were used; these are review flags, "
        "not measured classification accuracy.",
        "- Long legitimate abstracts can be flagged. Subtle fragments, "
        "fluent non-abstract summaries and translations can be missed.",
        "- Language checks use explicit repeated headings only. Scientific "
        "symbols and non-English text alone do not establish a defect.",
        "- Exact stored normalized text is retained. Provider parsing "
        "previously removed markup/whitespace; raw archives remain unchanged.",
        "- Prefixes are saved-result views, not new frontend executions. "
        "No downstream verification or retrieval-quality claim is made.",
        "",
        "See assessment.json for input SHA-256 hashes and full unchanged "
        "text; assessment.csv contains the same assessment rows "
        "and provenance.",
    ]
    (output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    """Run against saved results without initializing providers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", default=[10, 30])
    args = parser.parse_args()
    report = evaluate(args.input, args.output, sorted(set(args.depths)))
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
