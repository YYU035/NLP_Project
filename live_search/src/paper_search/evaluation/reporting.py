"""CSV/JSON and concise Markdown artifacts for preliminary coverage studies."""

import csv
import json
from pathlib import Path
from typing import Any

from paper_search.evaluation.analysis import (
    compare_calls,
    summarize_provider,
)
from paper_search.evaluation.models import SearchObservation


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_results(manifest: dict[str, Any]) -> dict[str, Any]:
    """Analyze saved observations without contacting any API."""
    calls = [
        SearchObservation.model_validate(row)
        for row in manifest["calls"]
    ]
    pairs = []
    for provider in manifest["providers"]:
        for case in manifest["cases"]:
            selected = [
                c
                for c in calls
                if c.provider == provider
                and c.claim_id == case["id"]
            ]
            natural = compare_calls(selected)
            matched = (
                compare_calls(selected, matched=True)
                if (manifest["options"]["matched_budget"])
                else None
            )
            alternatives = [
                c for c in selected if c.kind == "alternative"
            ]
            pairs.append(
                {
                    "provider": provider,
                    "claim_id": case["id"],
                    "domain": case["domain"],
                    "natural": natural,
                    "matched": matched,
                    "alternative_emitted": bool(alternatives),
                    "call_statuses": {
                        c.kind: c.status for c in selected
                    },
                }
            )
    summaries = {
        provider: summarize_provider(
            [c for c in calls if c.provider == provider]
        )
        for provider in manifest["providers"]
    }
    claims = []
    for case in manifest["cases"]:
        selected = [c for c in calls if c.claim_id == case["id"]]
        claims.append(
            {
                "claim_id": case["id"],
                "domain": case["domain"],
                "comparison": compare_calls(selected),
            }
        )
    helped = [
        p
        for p in pairs
        if p["natural"]["status"] == "complete"
        and (p["natural"]["additional"] or 0) > 0
    ]
    same = [
        p
        for p in pairs
        if p["natural"]["status"] == "complete"
        and p["natural"]["additional"] == 0
    ]
    harmed = [
        p
        for p in pairs
        if p["matched"]
        and p["matched"]["status"] == "complete"
        and (p["matched"]["lost"] or 0) > 0
    ]
    cost_only = [p for p in same if p["alternative_emitted"]]
    return {
        **manifest,
        "provider_summary": summaries,
        "comparisons": pairs,
        "claim_comparisons": claims,
        "examples": {
            "coverage_gain_not_relevance": helped[:3],
            "no_coverage_change": same[:3],
            "apparent_matched_coverage_loss": harmed[:3],
            "extra_cost_without_coverage_gain": cost_only[:3],
        },
        "precision_at_k": None,
        "recall_at_k": None,
        "accuracy_statement": "No reliable relevance labels; coverage only.",
    }


def write_results(
    manifest: dict[str, Any], output: Path
) -> dict[str, Any]:
    """Save normalized records, comparisons, and a Markdown report."""
    results = build_results(manifest)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    calls = [
        SearchObservation.model_validate(row)
        for row in manifest["calls"]
    ]
    _csv(
        output / "calls.csv",
        [
            {
                "call_id": c.id,
                "provider": c.provider,
                "claim_id": c.claim_id,
                "kind": c.kind,
                "query": c.query,
                "status": c.status,
                "result_count": c.result_count,
                "raw_result_count": c.raw_result_count,
                "latency_seconds": c.elapsed_seconds,
                "wire_requests": len(c.attempts),
                "http_statuses": json.dumps(
                    [a["status"] for a in c.attempts]
                ),
                "rate_limit_responses": sum(
                    a["status"] == 429 for a in c.attempts
                ),
                "metadata": json.dumps(
                    c.metadata, ensure_ascii=False
                ),
                "error": c.error,
                "authentication": c.authentication,
            }
            for c in calls
        ],
    )
    _csv(
        output / "papers.csv",
        [
            {
                "call_id": c.id,
                "claim_id": c.claim_id,
                "provider": c.provider,
                "query_kind": c.kind,
                "source_id": p.source_id,
                "doi": p.doi,
                "title": p.title,
                "authors": json.dumps(
                    p.authors, ensure_ascii=False
                ),
                "publication_date": p.publication_date,
                "year": p.year,
                "abstract_available": bool(p.abstract),
                "landing_url": p.landing_url,
                "pdf_urls": json.dumps(
                    [str(url) for url in p.pdf_urls]
                ),
            }
            for c in calls
            for p in c.papers
        ],
    )
    pair_rows = []
    for pair in results["comparisons"]:
        for arm in ("natural", "matched"):
            data = pair[arm]
            if data is None:
                continue
            pair_rows.append(
                {
                    "provider": pair["provider"],
                    "claim_id": pair["claim_id"],
                    "domain": pair["domain"],
                    "budget": arm,
                    **{
                        key: value
                        for key, value in data.items()
                        if key != "identity_comparison"
                        and not isinstance(value, list)
                    },
                }
            )
    _csv(output / "comparisons.csv", pair_rows)
    _csv(output / "claim_comparisons.csv", [{
        "claim_id": row["claim_id"], "domain": row["domain"],
        **{key: value for key, value in row["comparison"].items()
           if key != "identity_comparison" and not isinstance(value, list)},
    } for row in results["claim_comparisons"]])
    _csv(
        output / "provider_summary.csv",
        [
            {
                "provider": name,
                **{
                    key: json.dumps(value)
                    if isinstance(value, dict)
                    else value
                    for key, value in summary.items()
                },
            }
            for name, summary in results["provider_summary"].items()
        ],
    )
    (output / "report.md").write_text(
        _markdown(results), encoding="utf-8"
    )
    return results


def _value(value: Any) -> str:
    return "unavailable" if value is None else str(value)


def _rate(numerator: int, denominator: int) -> str:
    if not denominator:
        return "unavailable"
    return f"{numerator}/{denominator} ({numerator / denominator:.0%})"


def _markdown(results: dict[str, Any]) -> str:
    options = results["options"]
    lines = [
        "# Live scientific search: preliminary query comparison",
        "",
        f"Started (UTC): {results['started_at_utc']}. "
        f"Finished (UTC): {results.get('finished_at_utc', 'interrupted')}.",
        "",
        f"Claims: {len(results['cases'])}; limit: {options['max_results']} "
        f"per query; dates: {options['from_publication_date']} to "
        f"{options['to_publication_date']}.",
        "",
        "**Retrieval coverage only. No ground-truth relevance labels are "
        "available; Precision@K and Recall@K are not computed. Extra papers "
        "must not be interpreted as relevant papers or improved accuracy.**",
        "",
        "## Live validation and provider coverage",
        "",
        'The original query for each of the '
        'ten claims is the validation call. '
        'Successful empty responses are '
        'distinct from failures. Metadata counts '
        "below use original-call records, before deduplication.",
        "",
        "| Provider | Original OK / failed | All calls OK / failed | HTTP "
        'requests | 429s | Original / reform '
        'observed unique | Added observed |',
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, data in results["provider_summary"].items():
        available = data["original_successes"] > 0
        baseline = (
            data["observed_original_unique"] if available else None
        )
        reform = (
            data["observed_reformulated_unique"]
            if data["natural_successful_calls"]
            else None
        )
        added = (
            data["observed_additional_unique"]
            if available
            else None
        )
        lines.append(
            f"| {name} "
            f"| {data['original_successes']} / {data['original_failures']} "
            f"| {data['successful_calls']} / {data['failures']} "
            f"| {data['wire_requests']} | {data['http_429_responses']} "
            f"| {_value(baseline)} / {_value(reform)} | {_value(added)} |"
        )
    lines.extend(
        [
            "",
            'Observed unique totals are '
            'deduplicated across claims per provider; '
            'missing provider calls make totals '
            'partial. Only complete per-claim '
            "pairs support additional/overlap comparisons below.",
            "",
            '| Provider | Records | Titles | '
            'Authors | DOIs | Dates | Abstracts | PDF URLs '
            "| Date violations | Mean original latency (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, data in results["provider_summary"].items():
        counts = data["original_metadata_present"]
        total = data["original_metadata_record_count"]
        fields = (
            " | ".join(
                str(counts[key])
                for key in (
                    "title",
                    "authors",
                    "doi",
                    "publication_date",
                    "abstract",
                    "pdf_urls",
                )
            )
            if data["original_successes"]
            else "unavailable | " * 5 + "unavailable"
        )
        count_calls = (
            data["original_successes"] + data["original_failures"]
        )
        latency = (
            data["original_latency_seconds"] / count_calls
            if count_calls
            else 0
        )
        lines.append(
            f"| {name} "
            f"| {total if data['original_successes'] else 'unavailable'} "
            f"| {fields} | {data['date_filter_violations']} | {latency:.2f} |"
        )
    lines.extend(
        [
            "",
            "Dates are provider publication dates (arXiv original submission "
            "dates). Missing dates remain unknown; PDF availability means URL "
            "metadata only. No PDFs or landing pages were downloaded.",
            "",
            "## Per-provider, per-claim natural comparisons",
            "",
            "| Provider | Claim | Status | Original unique | Reform union | "
            "Overlap | Added | Wire requests original / reform | Zero O / R |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for pair in results["comparisons"]:
        data = pair["natural"]
        lines.append(
            f"| {pair['provider']} | {pair['claim_id']} | {data['status']} "
            f"| {_value(data['original_unique'])} "
            f"| {_value(data['reformulated_unique'])} "
            f"| {_value(data['overlap'])} | {_value(data['additional'])} "
            f"| {data['original_api_requests']} "
            f"/ {data['reformulated_api_requests']} "
            f"| {_value(data['original_zero'])} "
            f"/ {_value(data['reformulated_zero'])} |"
        )
    lines.extend([
        "", "## Zero-result rates and measured costs", "",
        "Rates below use successful claim outcomes only; unavailable calls "
        "are excluded, never counted as empty responses.", "",
        "| Provider | Original zero rate | Reform union zero rate | "
        "Original latency sum (s) | Extra-query latency sum (s) |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, data in results["provider_summary"].items():
        extra_latency = sum(
            row["natural"]["extra_query_latency_seconds"]
            for row in results["comparisons"] if row["provider"] == name
        )
        original_rate = _rate(
            data["original_zero_claims"], data["original_zero_denominator"]
        )
        reform_rate = _rate(
            data["reformulated_zero_claims"],
            data["reformulated_zero_denominator"],
        )
        lines.append(
            f"| {name} | "
            f"{original_rate} | {reform_rate} | "
            f"{data['original_latency_seconds']:.2f} | {extra_latency:.2f} |"
        )
    lines.extend([
        "", "## Per-claim coverage across available providers", "",
        "All observations below are partial if a provider failed. "
        "Counts do not replace missing outcomes from unavailable providers.",
        "",
        "| Claim | Status | Observed original unique | "
        "Observed reform unique |",
        "|---|---|---:|---:|",
    ])
    for row in results["claim_comparisons"]:
        comparison = row["comparison"]
        lines.append(
            f"| {row['claim_id']} | {comparison['status']} | "
            f"{comparison['observed_original_unique']} | "
            f"{comparison['observed_reformulated_unique']} |"
        )
    lines.extend(
        [
            "",
            "## Matched-request-budget comparison and natural cost",
            "",
            'The common original response is '
            'fetched once and reused by both modes. '
            'For claims with an alternative, '
            'matched budget uses two logical calls '
            'per arm: original + fresh repeated '
            'original versus original + concept '
            'query. Extra-call order is '
            'counterbalanced by claim index. With no '
            'alternative both arms use one '
            'call. Actual wire attempts can differ '
            'due to retries; those mismatches '
            'are reported, not called matched.',
            "",
            '| Provider | Complete matched '
            'pairs | Logical / wire matched pairs '
            "| Original / reform unique sum | Lost from matched baseline sum "
            "| Natural extra wire requests | Repeat-control overhead |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, summary in results["provider_summary"].items():
        matched = [
            p["matched"]
            for p in results["comparisons"]
            if p["provider"] == name and p["matched"]
        ]
        complete = [p for p in matched if p["status"] == "complete"]
        original_sum = (
            sum(p["original_unique"] for p in complete) if complete else None
        )
        reform_sum = (
            sum(p["reformulated_unique"] for p in complete)
            if complete else None
        )
        lost_sum = sum(p["lost"] for p in complete) if complete else None
        lines.append(
            f"| {name} | {len(complete)} | "
            f"{sum(p['logical_budget_matched'] for p in matched)} / "
            f"{sum(p['wire_budget_matched'] for p in matched)} | "
            f"{_value(original_sum)} / {_value(reform_sum)} | "
            f"{_value(lost_sum)} | "
            f"{summary['extra_query_wire_requests']} | "
            f"{summary['matched_repeat_overhead_requests']} |"
        )
    lines.extend(
        [
            "",
            'Sums above are per-claim unique '
            'counts, not global unique papers. '
            'Natural union includes the '
            'identical original response, so losing '
            'original papers in that union is '
            'impossible by design. Fresh repeats '
            'can expose variability and may '
            'add papers to the matched baseline. '
            "The experiment's physical HTTP total counts each shared original "
            'only once. Per-arm latency is a sum '
            'of client-call durations, including '
            "pacing/retries, not independent wall-clock A/B experiments.",
            "",
            "## Examples",
            "",
        ]
    )
    labels = {
        "coverage_gain_not_relevance": 'Helped observed '
        'coverage, not relevance',
        "no_coverage_change": "Made no observed coverage difference",
        "apparent_matched_coverage_loss": 'Appeared harmful '
        'to matched coverage',
        "extra_cost_without_coverage_gain": 'Appeared harmful '
        'in extra cost only',
    }
    for key, label in labels.items():
        examples = results["examples"][key]
        description = "; ".join(
            f"{x['provider']} {x['claim_id']}" for x in examples
        )
        lines.append(
            f"- {label}: {description or 'not observed in complete pairs'}."
        )
    corrections = [c for c in results["calls"] if c.get("parser_correction")]
    lines.extend([
        "", "## Actual API failures and parsing observations", "",
    ])
    for name in results["providers"]:
        calls = [c for c in results["calls"] if c["provider"] == name]
        failures = {}
        statuses = {}
        for call in calls:
            if call["error"]:
                failures[call["error"]] = failures.get(call["error"], 0) + 1
            for attempt in call["attempts"]:
                key = str(attempt["status"])
                statuses[key] = statuses.get(key, 0) + 1
        auth = sorted({c["authentication"] for c in calls})
        lines.append(f"- {name}: auth={','.join(auth)}; HTTP statuses "
                     f"{json.dumps(statuses)}; errors {json.dumps(failures)}.")
    if corrections:
        lines.append(
            f"- {len(corrections)} saved HTTP-200 Semantic Scholar responses "
            "reported total=0 and offset=0 without a data array. A confirmed "
            "client fix was verified against those exact archives offline. "
            "They are corrected to zero_results; original runtime statuses "
            "and errors remain in calls.jsonl and parser_correction. No new "
            "live requests were made for this correction."
        )
    controls = [
        c for c in results["calls"] if c["provider"] == "semantic_scholar"
        and c["kind"] == "original_repeat" and c["status"] == "success"
    ]
    if controls:
        lines.append(
            "- A successful Semantic Scholar repeat control parsed "
            f"{sum(len(c['papers']) for c in controls)} real papers. It does "
            "not replace failed original/alternative outcomes or establish "
            "a complete paired Semantic Scholar comparison."
        )
    lines.extend(
        [
            "",
            "## Claims and generated queries",
            "",
        ]
    )
    for case in results["cases"]:
        generation = results["generation"][case["id"]]
        lines.extend(
            [
                f"- **{case['id']} ({case['domain']})** "
                f"Original: {case['claim']}",
                "  Alternative: "
                + (
                    generation["queries"][1]
                    if len(generation["queries"]) > 1
                    else "none"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Limitations and reproducibility",
            "",
            '- One truncated page per query; at '
            'most the configured result limit. '
            'Provider ordering, index updates, '
            'proxy caches, and quota state vary.',
            '- No labels, relevance judgments, '
            'significance tests, or accuracy '
            'conclusions. Gains mean extra '
            'identities only; zero results do not '
            'refute a claim. Removing framing '
            'changes service-side top-k ordering.',
            '- DOI and arXiv identifiers are '
            'used when exposed by metadata. Exact '
            'long-title matching additionally '
            'requires the same known year and a '
            'shared full author; conflicting '
            'identifiers do not merge. Missing '
            'identifiers may leave duplicates; '
            'no fuzzy title matching is used.',
            '- Provider failures and rate '
            'limits are unavailable outcomes, not '
            'zero results. Zero-rate '
            'denominators exclude failed calls; detailed '
            'counts/denominators are in '
            'provider_summary.csv and results.json.',
            '- Authentication mode/header '
            'checks are recorded without secrets. '
            'A keyless run cannot validate keyed '
            'account access or quota increases.',
            '- Raw HTTP bodies, safe headers, '
            'exact parameters, timings, hashes '
            'and normalized papers are saved '
            'locally. Response bodies are decoded '
            'by httpx; credentials reflected '
            'in bodies are redacted and flagged.',
            '- results.json contains per-claim '
            'cross-provider observed coverage, '
            'identity groups, and all IDs. '
            'calls.csv/papers.csv/comparisons.csv '
            'provide detailed exports. '
            'calls.jsonl is checkpointed after each call.',
            '- No LLM, BM25/SBERT ranking, PDF '
            'download, or verification is run.',
            "",
        ]
    )
    return "\n".join(lines)
