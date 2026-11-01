"""Opt-in local KeyBERT preparation and a matched public retrieval study."""

import argparse
import asyncio
import csv
import hashlib
import importlib.metadata
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import httpx

from paper_search.clients.openaire import OpenAIREClient
from paper_search.clients.openalex import OpenAlexClient
from paper_search.deduplication import group_papers
from paper_search.evaluation.models import (
    ClaimCase,
    ExperimentOptions,
    SearchObservation,
)
from paper_search.evaluation.runner import ResponseRecorder, _search
from paper_search.keybert_queries import (
    MODEL_ID,
    MODEL_REVISION,
    KeyBERTQueryGenerator,
)
from paper_search.model_setup import download_model

OLD = Path("artifacts/arxiv-concepts-20261031")
OPENAIRE = Path("artifacts/openaire-experiment-20261101")


def save_json(path: Path, value: dict) -> None:
    """Save readable metadata without serializing credentials."""
    path.write_text(
        json.dumps(
            value, ensure_ascii=False, indent=2, default=str
        ),
        encoding="utf-8",
    )


def save_csv(
    path: Path, rows: list[dict], fields: list[str]
) -> None:
    """Keep a usable header even when no new papers are observed."""
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def prepare(output: Path, model_path: Path) -> dict:
    """Freeze ten queries and their qualitative caveats before any searches."""
    if (output / "queries.json").exists():
        raise FileExistsError(
            "Use a fresh output directory for preparation"
        )
    baseline = json.loads(
        (OLD / "results.json").read_text(encoding="utf-8")
    )
    prior = json.loads(
        (OPENAIRE / "results.json").read_text(encoding="utf-8")
    )
    if (
        model_path / "pinned_revision.txt"
    ).read_text().strip() != MODEL_REVISION:
        raise ValueError("Pinned model revision is not verified")
    generator = KeyBERTQueryGenerator(model_path)
    plans = {}
    for case in baseline["cases"]:
        detail = generator.generate_with_details(case["claim"])
        if detail.status != "built" or len(detail.queries) != 2:
            raise ValueError(
                f"No useful KeyBERT alternative for {case['id']}"
            )
        spacy = next(
            call["query"]
            for call in baseline["calls"]
            if call["provider"] == "openalex"
            and call["claim_id"] == case["id"]
            and call["kind"] == "alternative"
        )
        plans[case["id"]] = {
            "spacy": spacy,
            "keybert": detail.queries[1],
            "extraction": detail.model_dump(),
        }
    manifest = {
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "settings": {
            "phrase_limit": 3,
            "mmr": True,
            "diversity": 0.3,
            "candidate_ngram_range": [1, 4],
            "device": "cpu",
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in [
                "keybert",
                "sentence-transformers",
                "torch",
                "transformers",
                "scikit-learn",
                "numpy",
            ]
        },
        "model_sha256": {
            str(path.relative_to(model_path)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in model_path.rglob("*")
            if path.is_file() and ".cache" not in path.parts
        },
        "historical_sources": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                OLD / "results.json",
                OPENAIRE / "results.json",
                OLD / "manual_assessment_provisional_250.csv",
            ]
        },
        "baseline": baseline,
        "prior_openaire_calls": prior["calls"],
        "plans": plans,
        "options": baseline["options"],
    }
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "queries.json", manifest)
    lines = [
        "# Pre-retrieval query review",
        "",
        "Local KeyBERT/MiniLM versus the exact saved spaCy concept text. "
        "No retrieval results were consulted to tune these queries.",
        "",
        "| Claim | spaCy | KeyBERT | Retained phrases |",
        "|---|---|---|---|",
    ]
    for case_id, plan in plans.items():
        lines.append(
            f"| {case_id} | {plan['spacy']} | {plan['keybert']} | "
            f"{'; '.join(plan['extraction']['selected_phrases'])} |"
        )
    lines.extend(
        [
            "",
            "## Qualitative caveats",
            "",
            "KeyBERT ranks literal candidate phrases, not scientific "
            "entities or evidence. Compared with spaCy, it can retain "
            "different outcomes/conditions, overemphasize a measurement "
            "or omit a population/comparator. Neither discovery query "
            "is equivalent to the original claim. Negation and numeric "
            "conditions may be omitted; all omitted source spans and "
            "phrase cosine scores are in queries.json. Those scores "
            "are not relevance scores for retrieved papers.",
        ]
    )
    (output / "query_review.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return manifest


def report(manifest: dict, output: Path) -> dict:
    """Conservative identity-based novelty and partial label reuse."""
    for path, expected in manifest.get(
        "historical_sources", {}
    ).items():
        if (
            hashlib.sha256(Path(path).read_bytes()).hexdigest()
            != expected
        ):
            raise ValueError(
                "Historical source changed; cannot reuse labels"
            )
    baseline = manifest["baseline"]
    historic = [
        SearchObservation.model_validate(c)
        for c in baseline["calls"]
        + manifest["prior_openaire_calls"]
    ]
    calls = [
        SearchObservation.model_validate(c)
        for c in manifest["calls"]
    ]
    diagnostics = []
    for call in calls:
        total = None
        for attempt in call.attempts:
            if attempt.get("status") != 200:
                continue
            path = output / attempt["response_file"]
            if path.exists():
                payload = json.loads(path.read_bytes())
                total = (
                    payload.get("meta", {}).get("count")
                    if call.provider == "openalex"
                    else payload.get("header", {}).get("numFound")
                )
        diagnostics.append(
            {
                "claim_id": call.claim_id,
                "provider": call.provider,
                "strategy": "spacy"
                if call.kind == "original"
                else "keybert",
                "returned": call.result_count,
                "reported_total": total,
                "status": call.status,
                "potentially_capped": call.result_count == 10,
                "confirmed_capped": isinstance(total, int)
                and total > 10,
            }
        )
    entries = [
        (c, rank, p)
        for c in historic + calls
        for rank, p in enumerate(c.papers, 1)
    ]
    groups = group_papers([p for _, _, p in entries])
    historical = {
        (c.claim_id, identity)
        for (c, _, _), identity in zip(
            entries[: sum(len(c.papers) for c in historic)],
            groups.assignments,
            strict=False,
        )
    }
    labels = {}
    with (OLD / "manual_assessment_provisional_250.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        for row in csv.DictReader(file):
            value = row["manual_relevance"].strip()
            if not value:
                continue
            number = Decimal(value)
            if number not in {Decimal(0), Decimal(1), Decimal(2)}:
                raise ValueError("Invalid provisional label")
            key = (row["claim_id"], int(row["identity"]))
            if key in labels and labels[key] != int(number):
                raise ValueError("Conflicting historical labels")
            labels[key] = int(number)
    cases = {case["id"]: case for case in baseline["cases"]}
    rows = []
    offset = sum(len(c.papers) for c in historic)
    for (call, rank, paper), identity in zip(
        entries[offset:], groups.assignments[offset:], strict=True
    ):
        key = (call.claim_id, identity)
        rows.append(
            {
                "claim_id": call.claim_id,
                "identity": identity,
                "claim": cases[call.claim_id]["claim"],
                "provider": call.provider,
                "strategy": "spacy"
                if call.kind == "original"
                else "keybert",
                "query": call.query,
                "rank": rank,
                "source_id": paper.source_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "doi": paper.doi,
                "publication_date": paper.publication_date,
                "pdf_urls": json.dumps(
                    [str(url) for url in paper.pdf_urls]
                ),
                "paper_url": str(paper.landing_url or ""),
                "reused_label": labels.get(key, ""),
                "label_source": "provisional_250"
                if key in labels
                else "",
                "new_vs_history": key not in historical,
                "manual_relevance": "",
                "manual_notes": "",
            }
        )
    fields = (
        list(rows[0])
        if rows
        else ["claim_id", "identity", "title", "abstract"]
    )
    save_csv(output / "retrieved_papers.csv", rows, fields)
    pair_rows = {}
    for row in rows:
        pair_rows.setdefault(
            (row["claim_id"], row["identity"]), []
        ).append(row)
    new_rows = []
    for key, versions in pair_rows.items():
        if key in historical:
            continue
        chosen = dict(
            max(versions, key=lambda r: len(r["abstract"] or ""))
        )
        chosen["provenance"] = json.dumps(
            [
                {
                    k: r[k]
                    for k in (
                        "provider",
                        "strategy",
                        "query",
                        "rank",
                        "source_id",
                    )
                }
                for r in versions
            ],
            ensure_ascii=False,
        )
        chosen["keybert_present"] = any(
            r["strategy"] == "keybert" for r in versions
        )
        chosen["spacy_present"] = any(
            r["strategy"] == "spacy" for r in versions
        )
        new_rows.append(chosen)
    new_fields = fields + [
        "provenance",
        "keybert_present",
        "spacy_present",
    ]
    save_csv(
        output / "new_pairs_for_assessment.csv",
        new_rows,
        new_fields,
    )
    save_csv(
        output / "new_pairs_preview.csv",
        [
            {
                **r,
                "abstract": (r["abstract"] or "")[:12000],
                "abstract_truncated": len(r["abstract"] or "")
                > 12000,
            }
            for r in new_rows
        ],
        new_fields + ["abstract_truncated"],
    )
    summaries = []
    comparisons = []
    for provider in ("openalex", "openaire"):
        for case_id in [None, *cases]:
            sets = {}
            for strategy, kind in (
                ("spacy", "original"),
                ("keybert", "alternative"),
            ):
                rs = [
                    r
                    for r in rows
                    if r["provider"] == provider
                    and r["strategy"] == strategy
                    and (
                        case_id is None or r["claim_id"] == case_id
                    )
                ]
                cs = [
                    c
                    for c in calls
                    if c.provider == provider
                    and c.kind == kind
                    and (case_id is None or c.claim_id == case_id)
                ]
                keys = {(r["claim_id"], r["identity"]) for r in rs}
                sets[strategy] = keys
                summary = {
                    "provider": provider,
                    "claim_id": case_id or "ALL",
                    "strategy": strategy,
                    "records": len(rs),
                    "unique_pairs": len(keys),
                    "new_vs_history": len(keys - historical),
                    "known_labels": sum(
                        key in labels for key in keys
                    ),
                    "known_relevant": sum(
                        labels.get(key, -1) >= 1 for key in keys
                    ),
                    "known_evidence": sum(
                        labels.get(key, -1) == 2 for key in keys
                    ),
                    "unlabelled": sum(
                        key not in labels for key in keys
                    ),
                    "abstract_records": sum(
                        bool(r["abstract"]) for r in rs
                    ),
                    "pdf_records": sum(
                        bool(json.loads(r["pdf_urls"])) for r in rs
                    ),
                    "requests": sum(len(c.attempts) for c in cs),
                    "valid_zero_queries": sum(
                        c.status == "zero_results" for c in cs
                    ),
                    "failures": sum(not c.succeeded for c in cs),
                    "seconds": sum(c.elapsed_seconds for c in cs),
                    "status": {c.claim_id: c.status for c in cs},
                }
                summaries.append(summary)
            left, right = sets["spacy"], sets["keybert"]
            comparisons.append(
                {
                    "provider": provider,
                    "claim_id": case_id or "ALL",
                    "overlap": len(left & right),
                    "keybert_only": len(right - left),
                    "spacy_only": len(left - right),
                    "keybert_only_new_vs_history": len(
                        (right - left) - historical
                    ),
                    "known_relevant_added": sum(
                        labels.get(k, -1) >= 1 for k in right - left
                    ),
                    "known_relevant_displaced": sum(
                        labels.get(k, -1) >= 1 for k in left - right
                    ),
                }
            )
    result = {
        **manifest,
        "summaries": summaries,
        "query_diagnostics": diagnostics,
        "comparisons": comparisons,
        "new_pairs": len(new_rows),
        "historical_pairs": len(historical),
        "new_keybert_pairs": sum(
            r["keybert_present"] for r in new_rows
        ),
        "label_policy": "Reuse provisional_250 only; no new judgments",
    }
    save_json(output / "results.json", result)
    if diagnostics:
        save_csv(
            output / "request_diagnostics.csv",
            diagnostics,
            list(diagnostics[0]),
        )
    save_csv(
        output / "per_query.csv",
        [
            {**s, "status": json.dumps(s["status"])}
            for s in summaries
        ],
        list(summaries[0]),
    )
    lines = [
        "# KeyBERT versus spaCy retrieval",
        "",
        "## Method",
        "",
        "Same ten original claims; dates 2020-01-01–2026-10-30; top 10; "
        "fresh spaCy and KeyBERT searches on OpenAlex/OpenAIRE, one request "
        "per method/claim/provider, counterbalanced order. Maximum 20 "
        "physical attempts per provider, no retries; a first failure "
        "stops that provider. No ranking, paid inference, PDFs "
        "or LLM judgments.",
        "",
        "KeyBERT uses pinned local all-MiniLM-L6-v2 CPU embeddings, "
        "source-preserving 1–4-word candidates, cosine/MMR selection "
        "(diversity 0.3) and at most three nonoverlapping phrases. It "
        "retains the original claim through the existing QueryGenerator "
        "interface but this experiment submits only the extracted "
        "alternative. spaCy uses the exact saved concept queries; its "
        "behavior is unchanged. See query_review.md, generated before "
        "retrieval, for the ten-query qualitative comparison.",
        "",
        "## Provider totals",
        "",
        "| Provider | Method | Requests | Records / unique pairs | "
        "New vs history | Abstracts/descriptions | PDF URLs | "
        "Zeros | Failures |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        if s["claim_id"] == "ALL":
            lines.append(
                f"| {s['provider']} | {s['strategy']} | {s['requests']} | "
                f"{s['records']} / {s['unique_pairs']} | "
                f"{s['new_vs_history']} | "
                f"{s['abstract_records']} | {s['pdf_records']} | "
                f"{s['valid_zero_queries']} | {s['failures']} |"
            )
    for comp in comparisons:
        if comp["claim_id"] == "ALL":
            lines.extend(
                [
                    "",
                    f"{comp['provider']}: overlap **{comp['overlap']}**, "
                    f"KeyBERT-only **{comp['keybert_only']}**, "
                    f"spaCy-only **{comp['spacy_only']}** pairs.",
                ]
            )
    lines.extend(
        [
            "",
            f"Historical union: **{len(historical)} claim–paper pairs** "
            "from both old query arms of OpenAlex/arXiv/OpenAIRE. "
            f"Fresh retrieval contributes **{len(new_rows)} new pairs**, "
            f"of which **{result['new_keybert_pairs']} occur in KeyBERT** "
            "(including pairs shared with fresh spaCy).",
            "",
            "| Provider | Claim | spaCy records / unique | "
            "KeyBERT records / unique | "
            "Overlap | KB-only | KB-only new vs history |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for comp in comparisons:
        if comp["claim_id"] == "ALL":
            continue
        pair = [
            s
            for s in summaries
            if s["provider"] == comp["provider"]
            and s["claim_id"] == comp["claim_id"]
        ]
        a, b = pair

        def count_text(summary: dict) -> str:
            if not summary["requests"]:
                return "not attempted"
            if summary["failures"]:
                return "API failure"
            return f"{summary['records']}/{summary['unique_pairs']}"

        lines.append(
            f"| {comp['provider']} | {comp['claim_id']} | "
            f"{count_text(a)} | {count_text(b)} | "
            f"{comp['overlap']} | {comp['keybert_only']} | "
            f"{comp['keybert_only_new_vs_history']} |"
        )
    lines.extend(
        [
            "",
            f"{sum(q['potentially_capped'] for q in diagnostics)} queries "
            "returned exactly ten and are potentially capped; "
            f"{sum(q['confirmed_capped'] for q in diagnostics)} have "
            "archived API totals greater than ten. A zero result is valid "
            "only when the API call succeeded; failures/unattempted calls "
            "remain separate. Identical c10 queries are a repeated-query "
            "control, not evidence of a strategy difference.",
            "",
            "## Relevance and decision",
            "",
            "Only exact claim/identity matches inherit the existing "
            "provisional "
            "AI-assisted labels. They are not validated ground truth. "
            "New/unmatched papers stay unlabelled; comparing label counts "
            "on this selectively labelled subset cannot establish full "
            "retrieval quality. No scores or labels are invented.",
            "",
            "| Provider | Method | Known relevant / evidence | "
            "Known labels | Unlabelled |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for s in summaries:
        if s["claim_id"] == "ALL":
            lines.append(
                f"| {s['provider']} | {s['strategy']} | "
                f"{s['known_relevant']}/{s['known_evidence']} | "
                f"{s['known_labels']} | {s['unlabelled']} |"
            )
    lines.extend(
        [
            "",
            "**Retain spaCy as the default and proceed with integration.** "
            "Keep KeyBERT optional for this ablation until new candidate "
            "relevance is assessed. Additional identities establish "
            "coverage novelty, not a meaningful relevance advantage. "
            "Review new_pairs_preview.csv/full CSV, prioritizing "
            "keybert_present=True and spacy_present=False.",
            "",
            "## Limits and evidence",
            "",
            "The model is general English, not scientific NER; a small "
            "fixed candidate filter can omit useful qualifiers. Query "
            "similarity scores are not paper relevance. Different query "
            "lengths may affect conjunction matching. Top-10 truncation, "
            "native search semantics, index variability, conservative "
            "deduplication and incomplete provisional labels limit this "
            "small ten-claim study. OpenAIRE descriptions may be long, "
            "multilingual or not abstracts. PDF URLs are unverified.",
            "",
            "queries.json freezes model revision, weights hashes, package "
            "versions, selection details, source hashes and baseline. "
            "calls.jsonl/responses retain failures and raw responses. "
            "per_query.csv/results.json contain counts, overlaps, metadata, "
            "requests and latency (including pacing). retrieved_papers.csv "
            "includes reused labels and provenance; new pair CSVs have "
            "blank manual labels and all retrieval provenance. Preview "
            "descriptions are truncated to 12,000 characters with a flag; "
            "full text remains in the full CSV and results. Neither list "
            "is full-text scientific verification.",
        ]
    )
    (output / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


async def run(output: Path) -> dict:
    """Run a frozen plan without loading models or regenerating queries."""
    manifest = json.loads(
        (output / "queries.json").read_text(encoding="utf-8")
    )
    if (output / "calls.jsonl").exists():
        raise FileExistsError(
            "Existing run: summarize it, do not repeat calls"
        )
    for path, digest in manifest["historical_sources"].items():
        if (
            hashlib.sha256(Path(path).read_bytes()).hexdigest()
            != digest
        ):
            raise ValueError(
                "Historical source changed after query review"
            )
    options = ExperimentOptions.model_validate(manifest["options"])

    async def provider(name: str) -> list:
        recorder = ResponseRecorder(output, name)
        observations = []
        async with httpx.AsyncClient(
            event_hooks={
                "request": [recorder.on_request],
                "response": [recorder.on_response],
            }
        ) as http:
            client = (
                OpenAIREClient(http, max_retries=0)
                if name == "openaire"
                else OpenAlexClient(http, max_retries=0)
            )
            recorder.endpoint = client.endpoint
            for index, raw in enumerate(
                manifest["baseline"]["cases"]
            ):
                case = ClaimCase.model_validate(raw)
                jobs = [
                    (
                        "original",
                        manifest["plans"][case.id]["spacy"],
                    ),
                    (
                        "alternative",
                        manifest["plans"][case.id]["keybert"],
                    ),
                ]
                for kind, query in (
                    jobs if index % 2 == 0 else reversed(jobs)
                ):
                    call = await _search(
                        client, recorder, case, kind, query, options
                    )
                    observations.append(call)
                    print(
                        f"{name} {case.id} {kind}: {call.status}, "
                        f"papers={call.result_count}",
                        flush=True,
                    )
                    if not call.succeeded:
                        return observations
                    if name == "openalex":
                        await asyncio.sleep(1)
        return observations

    manifest["started_at_utc"] = datetime.now(UTC).isoformat()
    groups = await asyncio.gather(
        provider("openalex"), provider("openaire")
    )
    manifest["calls"] = [
        c.model_dump(mode="json") for group in groups for c in group
    ]
    manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
    return report(manifest, output)


def main() -> None:
    """Separate explicit model download, local preparation and live opt-in."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("download-model", "prepare", "run", "report"),
    )
    parser.add_argument("--run-live", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/keybert-experiment-20261101"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/local-models/all-MiniLM-L6-v2"),
    )
    args = parser.parse_args()
    if args.action == "download-model":
        download_model(args.model_path)
    elif args.action == "prepare":
        prepare(args.output, args.model_path)
    elif args.action == "run":
        if not args.run_live:
            parser.error(
                "--run-live required; no retrieval requests made"
            )
        run_result = asyncio.run(run(args.output))
        print(f"New pairs: {run_result['new_pairs']}")
    else:
        manifest = json.loads(
            (args.output / "queries.json").read_text(
                encoding="utf-8"
            )
        )
        manifest["calls"] = [
            json.loads(line)
            for line in (args.output / "calls.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        if (args.output / "results.json").exists():
            prior = json.loads(
                (args.output / "results.json").read_text(
                    encoding="utf-8"
                )
            )
            for name in ("started_at_utc", "finished_at_utc"):
                manifest[name] = prior.get(name)
        report(manifest, args.output)


if __name__ == "__main__":
    main()
