"""Opt-in matched-budget backend concept ablation; archive metadata only."""

import argparse
import asyncio
import csv
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx

from paper_search.clients.arxiv import ArxivClient
from paper_search.clients.openalex import OpenAlexClient
from paper_search.config import Settings
from paper_search.deduplication import group_papers
from paper_search.evaluation.dataset import CLAIMS
from paper_search.evaluation.models import ExperimentOptions, SearchObservation
from paper_search.evaluation.runner import ResponseRecorder, _search
from paper_search.query_processing import DeterministicQueryGenerator
from paper_search.query_rendering import render_query


def compare(calls: list[SearchObservation]) -> dict[str, Any]:
    """Compare coverage using globally consistent conservative identities."""
    records = [paper for call in calls for paper in call.papers]
    grouped = group_papers(records)
    offset = 0
    identities: dict[str, set[int]] = {"original": set(), "alternative": set()}
    profiles: dict[str, Any] = {}
    for kind in identities:
        chosen = [call for call in calls if call.kind == kind]
        profiles[kind] = {
            "calls": len(chosen),
            "valid_calls": sum(call.succeeded for call in chosen),
            "failures": sum(not call.succeeded for call in chosen),
            "zeros": sum(call.status == "zero_results" for call in chosen),
            "zero_rate_valid": (
                sum(call.status == "zero_results" for call in chosen)
                / sum(call.succeeded for call in chosen)
                if any(call.succeeded for call in chosen) else None
            ),
            "http_requests": sum(len(call.attempts) for call in chosen),
            "seconds": sum(call.elapsed_seconds for call in chosen),
            "records": sum(len(call.papers) for call in chosen),
            "with_abstract": sum(bool(p.abstract) for c in chosen
                                 for p in c.papers),
            "with_pdf_url": sum(bool(p.pdf_urls) for c in chosen
                                for p in c.papers),
        }
    for call in calls:
        identities[call.kind].update(
            grouped.assignments[offset:offset + len(call.papers)]
        )
        offset += len(call.papers)
    left, right = identities["original"], identities["alternative"]
    for kind in identities:
        profiles[kind]["unique"] = len(identities[kind])
    return {
        "arms": profiles,
        "overlap": len(left & right),
        "concept_only": len(right - left),
        "original_only": len(left - right),
        "union": len(left | right),
        "identity_assignments": grouped.assignments,
    }


def write_artifacts(manifest: dict[str, Any], output: Path) -> None:
    """Save raw observations, per-claim comparisons and manual-review CSV."""
    calls = [SearchObservation.model_validate(c) for c in manifest["calls"]]
    providers = {
        name: compare([c for c in calls if c.provider == name])
        for name in ("openalex", "arxiv")
    }
    claims = [
        {"provider": name, "claim_id": case.id,
         **compare([c for c in calls if c.provider == name
                    and c.claim_id == case.id])}
        for name in providers for case in CLAIMS
    ]
    combined = compare(calls)
    manifest.update(provider_summary=providers, comparisons=claims,
                    combined=combined)
    (output / "results.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    all_ids = combined["identity_assignments"]
    rows = []
    offset = 0
    for call in calls:
        for rank, paper in enumerate(call.papers, 1):
            rows.append({
                "identity": all_ids[offset], "claim_id": call.claim_id,
                "claim": next(
                    c.claim for c in CLAIMS if c.id == call.claim_id
                ),
                "provider": call.provider, "arm": call.kind, "rank": rank,
                "query": call.query, "source_id": paper.source_id,
                "doi": paper.doi, "title": paper.title,
                "abstract": paper.abstract,
                "publication_date": paper.publication_date,
                "paper_url": str(paper.landing_url or ""),
                "pdf_urls": json.dumps([str(url) for url in paper.pdf_urls]),
                "manual_relevance": "", "manual_notes": "",
            })
            offset += 1
    with (output / "manual_assessment.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        fields = list(rows[0]) if rows else ["title", "abstract"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# arXiv concept-rendering comparison — 31 October 2026", "",
        "Coverage only: there are no relevance labels or accuracy metrics.",
        "",
        "## Protocol", "",
        "Ten fixed claims; 2020-01-01–2026-10-30; top 10 per query. "
        "Each provider has a hard budget of 20 HTTP attempts, zero retries, "
        "one original and one concept request per claim. Order alternates "
        "by claim. Stop that provider after 429/401/403. No PDF downloads.",
        "",
        "The original arm sends the intact claim to OpenAlex and the legacy "
        "every-token AND rendering to arXiv. The concept arm uses the same "
        "NLP-selected surface phrases on both providers: space-separated "
        "plain text on OpenAlex, supported phrase syntax on arXiv. This is "
        "a backend-rendering ablation, not a change to QueryGenerator or its "
        "original_only/original_plus_concepts modes. Both generator outputs "
        "are saved in results.json; they are not additional API requests.",
        "",
        "The matched comparison is original versus concept, one request "
        "each. Their union is the natural extra-query case: two requests per "
        "claim, potentially 20 candidates rather than 10. It is not an "
        "equal-cost gain. Latency sums include pacing; attempt timings "
        "separate HTTP time. Shared originals are counted once in the union.",
        "",
        "Identities use the existing DOI/arXiv ID/conservative exact-title "
        "matcher, not fuzzy matching. Counts exclude failed requests from "
        "valid-zero denominators. Metadata URLs are unverified.", "",
        "## Per-provider coverage", "",
        "| Provider | Arm | Requests | Unique | Zeros/valid | Failures | "
        "Seconds | Abstracts/records | PDF URLs/records |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in providers.items():
        for arm, profile in summary["arms"].items():
            p = profile
            report.append(
                f"| {name} | {arm} | {p['http_requests']} | {p['unique']} | "
                f"{p['zeros']}/{p['valid_calls']} | {p['failures']} | "
                f"{p['seconds']:.2f} | {p['with_abstract']}/{p['records']} | "
                f"{p['with_pdf_url']}/{p['records']} |"
            )
    for name, summary in providers.items():
        report.extend([
            "", f"{name}: overlap **{summary['overlap']}**, concept-only "
            f"**{summary['concept_only']}**, original-only "
            f"**{summary['original_only']}**, two-query union "
            f"**{summary['union']}** unique papers.", "",
        ])
    report.extend([
        "Combined across providers after deduplication: original "
        f"**{combined['arms']['original']['unique']}**, concepts "
        f"**{combined['arms']['alternative']['unique']}**, overlap "
        f"**{combined['overlap']}**, union **{combined['union']}**. "
        "Each matched arm costs 20 total requests; the union costs 40.", "",
    ])
    report.extend([
        "## Per-claim matched comparison", "",
        "| Provider | Claim | Original unique | Concepts unique | Overlap | "
        "Concept-only | Original-only | Union |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in claims:
        report.append(
            f"| {row['provider']} | {row['claim_id']} | "
            f"{row['arms']['original']['unique']} | "
            f"{row['arms']['alternative']['unique']} | {row['overlap']} | "
            f"{row['concept_only']} | {row['original_only']} | "
            f"{row['union']} |"
        )
    report.extend([
        "", "## Interpretation examples", "",
        "Use the paired table to identify expanded coverage (for example "
        "an original zero becoming a nonzero concept result), unchanged "
        "coverage, and displacement (original-only identities). Expanded "
        "coverage is not relevance improvement. Original-only identities "
        "are candidates to inspect for possible recall loss. PDF metadata "
        "completeness can also deteriorate despite additional identities.",
        "", "## Rendered arXiv queries and omitted text", "",
    ])
    for case in CLAIMS:
        plan = manifest["rendering"][case.id]
        report.extend([
            f"- **{case.id}:** `{plan['text']}`",
            "  Omitted: " + json.dumps(plan["omitted_spans"],
                                       ensure_ascii=False),
        ])
    report.extend([
        "", "## Limitations", "",
        "Selected phrases are discovery constraints, not equivalent claims. "
        "Negation, comparisons, populations and numerical conditions may be "
        "omitted and must be assessed against the intact claim. Keeping only "
        "two concepts can lose important outcomes; exact phrases can miss "
        "wording variants. spaCy's small English model is not scientific NER. "
        "There are no synonym expansions or model downloads at runtime.",
        "",
        "More candidates or PDFs do not imply relevance. Original-only "
        "papers identify possible displacement under the matched budget, "
        "not proven relevance loss. Truncation, API variability, caching and "
        "different index/date semantics limit interpretation (arXiv uses "
        "submission dates). No relevance judgments, ranking or full-text "
        "verification were performed. See manual_assessment.csv for titles, "
        "abstracts and blank judgment columns, calls.jsonl for failures and "
        "attempts, and responses/ for credential-redacted raw responses.",
    ])
    (output / "report.md").write_text("\n".join(report) + "\n",
                                      encoding="utf-8")


async def run(output: Path) -> None:
    """Execute at most forty attempts and preserve every response."""
    if output.exists():
        raise FileExistsError("Choose a new output directory")
    output.mkdir(parents=True)
    settings = Settings.from_environment()
    options = ExperimentOptions(
        max_results=10, from_publication_date=date(2020, 1, 1),
        to_publication_date=date(2026, 10, 30), matched_budget=True,
    )
    generator = DeterministicQueryGenerator.from_settings(
        settings.model_copy(update={"query_mode": "original_plus_concepts"})
    )
    plans = {case.id: render_query(case.claim, "arxiv") for case in CLAIMS}
    manifest = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "options": options.model_dump(mode="json"),
        "cases": [case.model_dump() for case in CLAIMS],
        "generation": {
            case.id: generator.generate_with_details(case.claim).model_dump()
            for case in CLAIMS
        },
        "control_queries": {
            case.id: DeterministicQueryGenerator(mode="original_only")
            .generate(case.claim) for case in CLAIMS
        },
        "rendering": {key: plan.model_dump() for key, plan in plans.items()},
        "http_budget_per_provider": 20, "max_retries": 0,
        "relevance_labels": None,
    }
    (output / "run.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    async def provider(name: str) -> list[SearchObservation]:
        recorder = ResponseRecorder(output, name)
        rows = []
        async with httpx.AsyncClient(event_hooks={
            "request": [recorder.on_request],
            "response": [recorder.on_response],
        }) as http:
            client = (
                ArxivClient(http, max_retries=0)
                if name == "arxiv" else OpenAlexClient(
                    http, max_retries=0,
                    api_key=settings.openalex_api_key.get_secret_value()
                    if settings.openalex_api_key else None,
                )
            )
            recorder.endpoint = client.endpoint
            recorder.auth = client.auth_headers()
            for index, case in enumerate(CLAIMS):
                jobs = ["original", "alternative"]
                for kind in jobs if index % 2 == 0 else reversed(jobs):
                    query = case.claim
                    if isinstance(client, ArxivClient):
                        client.query_strategy = (
                            "literal" if kind == "original" else "concepts"
                        )
                    elif (kind == "alternative" and
                          plans[case.id].selected_concepts):
                        query = " ".join(plans[case.id].selected_concepts)
                    row = await _search(client, recorder, case, kind,
                                        query, options)
                    rows.append(row)
                    print(f"{row.id}: {row.status} {row.result_count}",
                          flush=True)
                    if any(a["status"] in {429, 401, 403}
                           for a in row.attempts):
                        return rows
                    if name == "openalex":
                        await asyncio.sleep(1)
        return rows

    groups = await asyncio.gather(provider("arxiv"), provider("openalex"))
    manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
    manifest["calls"] = [row.model_dump(mode="json")
                         for group in groups for row in group]
    write_artifacts(manifest, output)


def main() -> None:
    """Require deliberate network opt-in; no new services are contacted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-live", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/arxiv-concepts-20261031"
    ))
    args = parser.parse_args()
    if not args.run_live:
        parser.error("--run-live required; no network requests made")
    asyncio.run(run(args.output))


if __name__ == "__main__":
    main()
