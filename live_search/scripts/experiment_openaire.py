"""Bounded public OpenAIRE experiment against a frozen retrieval baseline."""

import argparse
import asyncio
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from paper_search.clients.openaire import OpenAIREClient
from paper_search.deduplication import group_papers
from paper_search.evaluation.models import (
    ClaimCase,
    ExperimentOptions,
    SearchObservation,
)
from paper_search.evaluation.runner import ResponseRecorder, _search

if __package__:
    from scripts.compare_arxiv_concepts import compare
else:
    from compare_arxiv_concepts import compare


def write_csv(
    path: Path, rows: list[dict], fields: list[str]
) -> None:
    """Always retain headers, including a valid experiment with no novelty."""
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    manifest: dict[str, Any], output: Path
) -> dict[str, Any]:
    """Compare identities, not relevance; keep new-paper labels blank."""
    baseline = [
        SearchObservation.model_validate(c)
        for c in manifest["baseline"]["calls"]
    ]
    fresh = [
        SearchObservation.model_validate(c)
        for c in manifest["calls"]
    ]
    source_metadata = {}
    query_diagnostics = []
    for call in fresh:
        reported_total = None
        for attempt in call.attempts:
            if attempt.get("status") != 200:
                continue
            path = output / attempt["response_file"]
            if not path.exists():
                continue
            payload = json.loads(path.read_bytes())
            reported_total = payload.get("header", {}).get(
                "numFound"
            )
            for record in payload.get("results", []):
                source_metadata[record.get("id")] = record
        query_diagnostics.append(
            {
                "id": call.id,
                "claim_id": call.claim_id,
                "arm": call.kind,
                "reported_total": reported_total,
                "returned": call.result_count,
                "status": call.status,
                "capped": (reported_total > 10)
                if isinstance(reported_total, int)
                and call.result_count == 10
                else False,
            }
        )
    cases = {c["id"]: c for c in manifest["baseline"]["cases"]}
    entries = [
        (call, rank, paper)
        for call in baseline + fresh
        for rank, paper in enumerate(call.papers, 1)
    ]
    grouped = group_papers([paper for _, _, paper in entries])
    old = {
        (call.claim_id, identity)
        for (call, _, _), identity in zip(
            entries, grouped.assignments, strict=True
        )
        if call.provider != "openaire"
    }
    retrieved = []
    novel: dict[tuple[str, int], list[dict]] = {}
    for (call, rank, paper), identity in zip(
        entries, grouped.assignments, strict=True
    ):
        if call.provider != "openaire":
            continue
        pair = (call.claim_id, identity)
        row = {
            "identity": identity,
            "claim_id": call.claim_id,
            "claim": cases[call.claim_id]["claim"],
            "provider": "openaire",
            "arm": call.kind,
            "rank": rank,
            "query": call.query,
            "source_id": paper.source_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "doi": paper.doi,
            "authors": json.dumps(paper.authors),
            "publication_date": paper.publication_date,
            "paper_url": str(paper.landing_url or ""),
            "pdf_urls": json.dumps(
                [str(link) for link in paper.pdf_urls]
            ),
            "new_vs_saved_union": pair not in old,
            "description_caveat": (
                "OpenAIRE description, not guaranteed abstract"
            ),
            "source_language_code": (
                source_metadata.get(paper.source_id, {}).get(
                    "language"
                )
                or {}
            ).get("code"),
            "description_characters": len(paper.abstract or ""),
            "manual_relevance": "",
            "manual_notes": "",
        }
        retrieved.append(row)
        novel.setdefault(pair, []).append(row)
    observed = set(novel)
    novel = {
        key: values
        for key, values in novel.items()
        if key not in old
    }
    new_rows = []
    for _key, versions in sorted(novel.items()):
        chosen = max(
            versions, key=lambda r: len(r["abstract"] or "")
        )
        row = dict(chosen)
        row["provenance"] = json.dumps(
            [
                {
                    k: v[k]
                    for k in (
                        "provider",
                        "arm",
                        "rank",
                        "query",
                        "source_id",
                    )
                }
                for v in versions
            ],
            ensure_ascii=False,
        )
        row["pdf_urls"] = json.dumps(
            sorted(
                {
                    link
                    for v in versions
                    for link in json.loads(v["pdf_urls"])
                }
            )
        )
        new_rows.append(row)
    fields = [
        "identity",
        "claim_id",
        "claim",
        "provider",
        "arm",
        "rank",
        "query",
        "source_id",
        "title",
        "abstract",
        "doi",
        "authors",
        "publication_date",
        "paper_url",
        "pdf_urls",
        "new_vs_saved_union",
        "description_caveat",
        "source_language_code",
        "description_characters",
        "manual_relevance",
        "manual_notes",
    ]
    write_csv(output / "openaire_papers.csv", retrieved, fields)
    write_csv(
        output / "new_pairs_for_assessment.csv",
        new_rows,
        fields + ["provenance"],
    )
    preview = [
        {
            **row,
            "abstract": (row["abstract"] or "")[:12000],
            "abstract_truncated": len(row["abstract"] or "")
            > 12000,
        }
        for row in new_rows
    ]
    write_csv(
        output / "new_pairs_preview.csv",
        preview,
        fields + ["provenance", "abstract_truncated"],
    )
    providers = {
        name: compare(
            [c for c in baseline + fresh if c.provider == name]
        )
        for name in ("openalex", "arxiv", "openaire")
    }
    per_claim = []
    for case_id in cases:
        for provider in providers:
            selected = [
                c
                for c in baseline + fresh
                if c.claim_id == case_id and c.provider == provider
            ]
            per_claim.append(
                {
                    "claim_id": case_id,
                    "provider": provider,
                    "original": sum(
                        len(c.papers)
                        for c in selected
                        if c.kind == "original" and c.succeeded
                    ),
                    "concepts": sum(
                        len(c.papers)
                        for c in selected
                        if c.kind == "alternative" and c.succeeded
                    ),
                    "statuses": {
                        arm: next(
                            (
                                c.status
                                for c in selected
                                if c.kind == arm
                            ),
                            "not_attempted",
                        )
                        for arm in ("original", "alternative")
                    },
                    "unique_union": compare(selected)["union"],
                    "new_pairs": sum(
                        key[0] == case_id for key in novel
                    )
                    if provider == "openaire"
                    else None,
                }
            )
    result = {
        **manifest,
        "provider_summary": providers,
        "query_diagnostics": query_diagnostics,
        "per_claim": per_claim,
        "novelty": {
            "saved_claim_paper_pairs": len(old),
            "openaire_claim_paper_pairs": len(observed),
            "overlapping_pairs": len(observed & old),
            "new_pairs": len(novel),
            "combined_pairs": len(old | observed),
            "new_with_description": sum(
                bool(r["abstract"]) for r in new_rows
            ),
            "new_with_pdf_url": sum(
                bool(json.loads(r["pdf_urls"])) for r in new_rows
            ),
        },
        "relevance": "Not judged; novelty is not relevance or accuracy",
    }
    (output / "results.json").write_text(
        json.dumps(
            result, ensure_ascii=False, indent=2, default=str
        ),
        encoding="utf-8",
    )
    flat_claims = [
        {**row, "statuses": json.dumps(row["statuses"])}
        for row in per_claim
    ]
    write_csv(
        output / "per_claim.csv",
        flat_claims,
        list(flat_claims[0]) if flat_claims else ["claim_id"],
    )
    n = result["novelty"]
    request_count = sum(len(c.attempts) for c in fresh)
    failures = sum(not c.succeeded for c in fresh)
    total_records = sum(len(c.papers) for c in fresh)
    lines = [
        "# OpenAIRE discovery experiment",
        "",
        f"Completed UTC: {manifest.get('finished_at_utc', 'partial run')}. "
        "OpenAlex/arXiv are saved 31 October results, not fresh requests.",
        "",
        "## Protocol and API",
        "",
        "Same ten claims, exact saved original and space-separated spaCy "
        "concept queries; inclusive dates 2020-01-01–2026-10-30; top 10, "
        "one page, native relevance order. No model/query regeneration. "
        "Original/concept order alternates by claim. Anonymous public API "
        "only, at most 20 physical attempts, zero retries, serialized starts "
        "at least three seconds apart; stop on the first failed query. "
        "No PDF/landing-page downloads or relevance scoring.",
        "",
        "[OpenAIRE v3 "
        "search](https://graph.openaire.eu/docs/apis/graph-"
        "api/research-products/) "
        "supports keyword `search`, `type=publication`, date filters and "
        "pageSize 1–100. [Terms](https://graph.openaire.eu/docs/apis/terms/) "
        "allow free access: anonymous up to 60/hour, authenticated up to "
        "7,200/hour. This bounded run uses no credentials or paid services. "
        "Coordinate any other users of the same anonymous quota.",
        "",
        "[Query syntax](https://graph.openaire.eu/docs/apis"
        "/graph-api/research-products/filtering/) "
        "supports quoted Boolean operands; this experiment uses plain "
        "keyword input without invented synonyms or phrase constraints. "
        "The adapter quotes an entire literal input if it contains an "
        "uppercase reserved operator, so claim text cannot "
        "become NOT exclusion.",
        "",
        "## Observed results",
        "",
        f"OpenAIRE: **{len(fresh)} logical calls / {request_count} HTTP "
        f"attempts**, **{failures} failures**, **{total_records} returned "
        f"record instances**, **{n['openaire_claim_paper_pairs']} unique "
        "claim–paper pairs across its two query arms.",
        "",
        "| Provider | Arm | Calls | Records | Unique papers | Valid zeros | "
        "Failures | Descriptions/abstracts | PDF URLs | Seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for provider, summary in providers.items():
        for arm, a in summary["arms"].items():
            lines.append(
                f"| {provider} | {arm} | {a['http_requests']} | "
                f"{a['records']} | {a['unique']} | {a['zeros']}/"
                f"{a['valid_calls']} | {a['failures']} | "
                f"{a['with_abstract']} | "
                f"{a['with_pdf_url']} | {a['seconds']:.2f} |"
            )
    capped = sum(q["capped"] for q in query_diagnostics)
    oversized = sum(
        row["description_characters"] > 12000 for row in new_rows
    )
    lines.extend(
        [
            "",
            f"**{capped} OpenAIRE responses are confirmed top-10 capped** "
            "because their reported total exceeds ten; query_diagnostics in "
            "results.json preserves those totals. Short/empty responses are "
            "not treated as API failures. Latency sums include pacing.",
        ]
    )
    lines.extend(
        [
            "",
            f"Against the saved **{n['saved_claim_paper_pairs']}**-pair "
            f"OpenAlex/arXiv union, OpenAIRE overlaps "
            f"**{n['overlapping_pairs']}** "
            f"pairs and contributes **{n['new_pairs']} new pairs**; combined "
            f"coverage is **{n['combined_pairs']} pairs**. New pairs include "
            f"**{n['new_with_description']} descriptions** and "
            f"**{n['new_with_pdf_url']} explicit PDF URL metadata records**. "
            "These new pairs have no assigned relevance labels.",
            "",
            "## Per-claim/provider comparison",
            "",
            "Counts are saved returned records (top 10), not "
            "all indexed hits. "
            "`pending`/`not_attempted` and failures must not "
            "be read as zeros.",
            "",
            "| Claim | Provider | Original returned | Concepts returned | "
            "Unique union | New vs saved two-provider union | Status O / C |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in per_claim:
        counts = []
        for arm, key in (
            ("original", "original"),
            ("alternative", "concepts"),
        ):
            counts.append(
                str(row[key])
                if row["statuses"][arm]
                in {"success", "zero_results"}
                else "unavailable"
            )
        lines.append(
            f"| {row['claim_id']} | {row['provider']} | {counts[0]} | "
            f"{counts[1]} | {row['unique_union']} | "
            f"{row['new_pairs'] if row['new_pairs'] is not None else '—'} | "
            f"{row['statuses']['original']} / "
            f"{row['statuses']['alternative']} |"
        )
    lines.extend(
        [
            "",
            "## Decision and limitations",
            "",
            f"Of the novel pairs, **{oversized} descriptions exceed 12,000 "
            "characters**. new_pairs_preview.csv truncates these with an "
            "explicit flag; the full CSV, results and raw responses retain "
            "the complete text. Some descriptions are much longer than "
            "normal abstracts, so description availability is not abstract "
            "quality. Language codes are retained but can be undetermined.",
            "",
            (
                "Retain as an **optional candidate source pending "
                "manual relevance "
                "assessment**, rather than replacing "
                "OpenAlex/arXiv: it supplies "
                "new identities, but their usefulness is not "
                "established. Review "
                "new_pairs_for_assessment.csv before adding "
                "routine request cost."
                if n["new_pairs"]
                else "No additional claim–paper pairs were observed. "
                "Defer routine use; "
                "failure/zero outcomes and top-10 truncation limit "
                "this conclusion."
            ),
            "",
            "Descriptions are not guaranteed abstracts. The adapter strips "
            "markup and retains the first nonempty description; it does not "
            "translate or infer scientific evidence. Metadata "
            "can be multilingual, "
            "incomplete, duplicated, or a publication subtype other than a "
            "research article. Explicit PDF-shaped URLs alone count as PDF "
            "metadata; DOI/OA flags do not prove downloadable full text.",
            "",
            "Deduplication reuses DOI/arXiv IDs and "
            "conservative exact-title/year/"
            "author matching, with known conflicting "
            "identifiers kept separate. "
            "A pair is new only if absent for that same claim "
            "from **both arms "
            "of both saved providers**. Novelty is relative to "
            "those saved top-10 "
            "lists, not to either provider's complete corpus. "
            "Different run dates, "
            "index updates and default ranking prevent a "
            "simultaneous controlled "
            "provider-quality comparison. Top-10 lists may be truncated; raw "
            "header.numFound is archived. No precision, "
            "accuracy or recall gain "
            "is inferred from additional papers.",
            "",
            "## Evidence",
            "",
            "- run.json freezes claims, saved queries, dates "
            "and baseline hash.",
            "- calls.jsonl and responses/*.body retain "
            "requests, errors, timing "
            "and raw responses with SHA-256 hashes; no "
            "credentials are supplied.",
            "- results.json / per_claim.csv contain counts and comparisons.",
            "- openaire_papers.csv retains all normalized "
            "OpenAIRE occurrences.",
            "- new_pairs_for_assessment.csv has one row per novel claim–paper "
            "pair, title/description, URLs and all retrieval provenance. "
            "manual_relevance and manual_notes are intentionally blank.",
            "- new_pairs_preview.csv has the same identities and provenance, "
            "with a 12,000-character description preview for easier review.",
        ]
    )
    (output / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


async def run(saved: Path, output: Path) -> dict:
    """Run twenty or fewer public requests; never retry or switch endpoints."""
    if output.exists():
        raise FileExistsError(
            "Use a fresh directory to preserve prior evidence"
        )
    baseline = json.loads(saved.read_text(encoding="utf-8"))
    if (
        len(baseline["cases"]) != 10
        or baseline["options"]["max_results"] != 10
    ):
        raise ValueError(
            "Expected the saved ten-claim, top-10 experiment"
        )
    options = ExperimentOptions.model_validate(baseline["options"])
    for case in baseline["cases"]:
        arms = [
            c["kind"]
            for c in baseline["calls"]
            if c["provider"] == "openalex"
            and c["claim_id"] == case["id"]
        ]
        if sorted(arms) != ["alternative", "original"]:
            raise ValueError(
                "Expected exactly two saved query arms per claim"
            )
    output.mkdir(parents=True)
    manifest = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "baseline_file": str(saved),
        "baseline_sha256": hashlib.sha256(
            saved.read_bytes()
        ).hexdigest(),
        "baseline": baseline,
        "max_http_attempts": 20,
        "authentication": "anonymous",
        "max_retries": 0,
        "options": options.model_dump(mode="json"),
    }
    (output / "run.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    recorder = ResponseRecorder(output, "openaire")
    recorder.endpoint = OpenAIREClient.endpoint
    calls = []
    async with httpx.AsyncClient(
        event_hooks={
            "request": [recorder.on_request],
            "response": [recorder.on_response],
        }
    ) as http:
        client = OpenAIREClient(
            http, max_retries=0, timeout_seconds=30
        )
        for index, raw_case in enumerate(baseline["cases"]):
            case = ClaimCase.model_validate(raw_case)
            jobs = [
                c
                for c in baseline["calls"]
                if c["provider"] == "openalex"
                and c["claim_id"] == case.id
            ]
            jobs.sort(key=lambda c: c["kind"] != "original")
            if index % 2:
                jobs.reverse()
            for job in jobs:
                if len(calls) >= 20:
                    break
                call = await _search(
                    client,
                    recorder,
                    case,
                    job["kind"],
                    job["query"],
                    options,
                )
                calls.append(call)
                print(
                    f"{call.id}: {call.status}; results={call.result_count}",
                    flush=True,
                )
                if not call.succeeded:
                    break
            if calls and not calls[-1].succeeded:
                break
    manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
    manifest["calls"] = [c.model_dump(mode="json") for c in calls]
    return analyze(manifest, output)


def main() -> None:
    """Require opt-in for live search; support offline report regeneration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-live", action="store_true")
    parser.add_argument("--summarize-existing", type=Path)
    parser.add_argument(
        "--saved",
        type=Path,
        default=Path(
            "artifacts/arxiv-concepts-20261031/results.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/openaire-experiment-20261101"),
    )
    args = parser.parse_args()
    if args.summarize_existing:
        output = args.summarize_existing
        manifest = json.loads(
            (output / "run.json").read_text(encoding="utf-8")
        )
        manifest["calls"] = [
            json.loads(line)
            for line in (output / "calls.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        if (output / "results.json").exists():
            manifest["finished_at_utc"] = json.loads(
                (output / "results.json").read_text(
                    encoding="utf-8"
                )
            ).get("finished_at_utc")
        result = analyze(manifest, output)
    elif args.run_live:
        result = asyncio.run(run(args.saved, args.output))
    else:
        parser.error("--run-live required; no requests made")
    print(json.dumps(result["novelty"]))


if __name__ == "__main__":
    main()
