"""Offline identity/label joins and summaries for the frozen readiness runs."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from paper_search.deduplication import arxiv_ids, group_papers, normalized_doi
from paper_search.metadata import ScientificPaper
from paper_search.retrieval_models import RetrievalResult

ROOT = Path("artifacts")


def load(path: Path) -> dict:
    """Read a saved JSON artifact, without network access."""
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict]:
    """Retain the existing long OpenAIRE description fields."""
    csv.field_size_limit(5_000_000)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    """Never overwrite an assessment export."""
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def resolve_labels(observations: dict[str, list[int]]) -> tuple[dict, set]:
    """Keep conflicting judgments unknown; never invent a label for a blank."""
    labels, conflicts = {}, set()
    for key, values in observations.items():
        unique = set(values)
        if not unique <= {0, 1, 2}:
            raise ValueError("Invalid relevance label")
        if len(unique) == 1:
            labels[key] = next(iter(unique))
        elif len(unique) > 1:
            conflicts.add(key)
    return labels, conflicts


def stats(records: list[dict], labels: dict) -> dict:
    """Count claim-pair yield, using metadata from these occurrences only."""
    keys = {r["key"] for r in records}
    return {
        "records": len(records),
        "unique": len(keys),
        "relevant_known": sum(labels.get(k) in {1, 2} for k in keys),
        "evidence_known": sum(labels.get(k) == 2 for k in keys),
        "unknown": sum(k not in labels for k in keys),
        "abstract_unique": len(
            {r["key"] for r in records if r["paper"].abstract}
        ),
        "pdf_unique": len(
            {
                r["key"]
                for r in records
                if r["paper"].pdf_url or r["paper"].pdf_urls
            }
        ),
    }


def analyze(depth_dir: Path, fallback_dir: Path, output: Path) -> dict:
    """Validate archives, reconcile current labels, and compare offline."""
    if output.exists():
        raise FileExistsError("Use a fresh analysis directory")
    latest_path = ROOT / "arxiv-keybert-20261101/results.json"
    latest = load(latest_path)
    previous = latest["previous"]
    historical_calls = (
        previous["baseline"]["calls"]
        + previous["prior_openaire_calls"]
        + previous["calls"]
        + latest["calls"]
    )
    history = [
        (c["claim_id"], ScientificPaper.model_validate(p))
        for c in historical_calls
        for p in c["papers"]
    ]
    frozen_history = group_papers([p for _, p in history])
    historical_keys = {
        f"{c}:{i}"
        for (c, _), i in zip(history, frozen_history.assignments, strict=True)
    }
    observations = defaultdict(list)
    manifest_path = ROOT / "gemini-relevance-20261031/manifest.json"
    manifest = load(manifest_path)
    mapping = {
        p["id"]: f"{p['claim_id']}:{int(p['identity'])}"
        for p in manifest["pairs"]
    }
    sources = [
        (
            ROOT
            / "arxiv-concepts-20261031/manual_assessment_provisional_250.csv",
            "manual_relevance",
        ),
        (
            ROOT
            / "keybert-experiment-20261101"
            / "keybert_new_pairs_provisional_labels.csv",
            "manual_relevance",
        ),
        (
            ROOT / "gemini-relevance-20261031/manual_validation_30_filled.csv",
            "human_label",
        ),
    ]
    # These sources are unlabelled at preparation time, but accept actual later
    # judgments if supplied. An exported blank never implies label 0.
    for path in (
        ROOT / "openaire-experiment-20261101/new_pairs_for_assessment.csv",
        ROOT / "arxiv-keybert-20261101/new_pairs_for_assessment.csv",
    ):
        data = csv_rows(path)
        if data and "manual_relevance" in data[0]:
            sources.append((path, "manual_relevance"))
    for path, field in sources:
        for row in csv_rows(path):
            key = (
                mapping[row["id"]]
                if field == "human_label"
                else (f"{row['claim_id']}:{int(row['identity'])}")
            )
            if key not in historical_keys:
                raise ValueError(f"Unmatched historical label key {key}")
            raw = row[field].strip()
            if raw:
                value = Decimal(raw)
                if value not in {Decimal(0), Decimal(1), Decimal(2)}:
                    raise ValueError("Invalid provisional label")
                observations[key].append(int(value))
    labels, conflicts = resolve_labels(observations)
    runs, records, queries, inputs = [], [], [], [latest_path, manifest_path]
    inputs.extend(p for p, _ in sources)
    archive_count = 0
    for mode, folder in (("depth", depth_dir), ("fallback", fallback_dir)):
        plan = load(folder / "plan.json")
        original_plan = (
            ROOT / "integration-readiness-20261101" / f"{mode}_plan.json"
        )
        if plan != load(original_plan):
            raise ValueError(
                "Executed plan differs from the approved frozen plan"
            )
        inputs.append(original_plan)
        inputs.append(folder / "plan.json")
        attempts = 0
        for path in sorted(folder.glob("c??.json")):
            result = RetrievalResult.model_validate(load(path))
            inputs.append(path)
            claim_id = path.stem
            for provider, outcome in result.providers.items():
                runs.append(
                    {
                        "experiment": mode,
                        "claim_id": claim_id,
                        "claim": result.original_claim,
                        "provider": provider,
                        "outcome": outcome,
                        "result": result,
                    }
                )
                for index, record in enumerate(outcome.records):
                    records.append(
                        {
                            "experiment": mode,
                            "claim_id": claim_id,
                            "claim": result.original_claim,
                            "provider": provider,
                            "query_id": record.query_id,
                            "rank": record.rank,
                            "record_index": index,
                            "paper": record.paper,
                        }
                    )
                for query in outcome.queries:
                    for attempt in query.attempts:
                        attempts += 1
                        if attempt.response_file:
                            archive = folder / attempt.response_file
                            if (
                                hashlib.sha256(
                                    archive.read_bytes()
                                ).hexdigest()
                                != attempt.sha256
                            ):
                                raise ValueError(
                                    f"Archive checksum mismatch: {archive}"
                                )
                            archive_count += 1
                            inputs.append(archive)
                    queries.append(
                        {
                            "experiment": mode,
                            "claim_id": claim_id,
                            "provider": provider,
                            "query_id": query.id,
                            "stage": query.stage,
                            "query": query.query,
                            "status": query.status,
                            "returned": query.returned,
                            "seconds": query.seconds,
                            "attempts": len(query.attempts),
                            "http_seconds": sum(
                                a.seconds for a in query.attempts
                            ),
                            "error": query.error or "",
                            "notes": json.dumps(
                                query.notes, ensure_ascii=False
                            ),
                            "parameters": json.dumps(
                                [a.parameters for a in query.attempts],
                                ensure_ascii=False,
                            ),
                        }
                    )
            if result.http_attempts != sum(
                o.http_attempts for o in result.providers.values()
            ):
                raise ValueError("Inconsistent HTTP accounting")
        budget = (
            plan["max_http_attempts"]
            if mode == "depth"
            else plan["max_logical_requests"]
        )
        if attempts > budget or any(
            q["attempts"] > 1 for q in queries if q["experiment"] == mode
        ):
            raise ValueError("Frozen zero-retry experimental budget exceeded")
    grouping = group_papers(
        [p for _, p in history] + [r["paper"] for r in records]
    )
    if grouping.assignments[: len(history)] != frozen_history.assignments:
        raise ValueError("Historical identity assignments changed")
    for record, identity in zip(
        records, grouping.assignments[len(history) :], strict=True
    ):
        record["key"] = f"{record['claim_id']}:{identity}"
    by_run = defaultdict(list)
    for record in records:
        by_run[
            (record["experiment"], record["claim_id"], record["provider"])
        ].append(record)
    depth_rows, fallback_rows = [], []
    retained_fallback = []
    for run in runs:
        mode, case, provider, outcome = (
            run[k] for k in ("experiment", "claim_id", "provider", "outcome")
        )
        items = by_run[(mode, case, provider)]
        if mode == "depth":
            for k in (10, 20, 30):
                selected = items[:k]
                depth_rows.append(
                    {
                        "claim_id": case,
                        "provider": provider,
                        "depth": k,
                        "status": outcome.status,
                        **stats(selected, labels),
                        "potentially_capped": len(selected) == k,
                        "shared_top30_seconds": outcome.seconds,
                        "shared_http_attempts": outcome.http_attempts,
                    }
                )
        else:
            primary_id = outcome.queries[0].id
            primary = [r for r in items if r["query_id"] == primary_id]
            relaxed = [r for r in items if r["query_id"] != primary_id]
            primary_keys = {r["key"] for r in primary}
            relaxed_keys = {r["key"] for r in relaxed}
            keep = set()
            for record in items:
                if len(keep) < run["result"].top_k:
                    keep.add(record["key"])
            retained = [r for r in items if r["key"] in keep]
            retained_fallback.extend(retained)
            extra = [r for r in items if r["key"] not in primary_keys]
            fallback_rows.append(
                {
                    "claim_id": case,
                    "status": outcome.status,
                    "primary": stats(primary, labels),
                    "union": stats(items, labels),
                    "retained_union": stats(retained, labels),
                    "additional": stats(extra, labels),
                    "primary_fallback_overlap": len(
                        primary_keys & relaxed_keys
                    ),
                    "primary_seconds": outcome.queries[0].seconds,
                    "fallback_seconds": sum(
                        q.seconds for q in outcome.queries[1:]
                    ),
                    "http_attempts": outcome.http_attempts,
                    "fallback_attempts": sum(
                        len(q.attempts) for q in outcome.queries[1:]
                    ),
                    "relaxation": [q.query for q in outcome.queries[1:]],
                }
            )
    provider_depth = []
    for provider in ("openalex", "openaire", "arxiv", "combined"):
        for k in (10, 20, 30):
            relevant_runs = [
                r
                for r in runs
                if r["experiment"] == "depth"
                and (provider == "combined" or r["provider"] == provider)
            ]
            selected = [
                item
                for r in relevant_runs
                for item in by_run[("depth", r["claim_id"], r["provider"])][:k]
            ]
            provider_depth.append(
                {
                    "provider": provider,
                    "depth": k,
                    **stats(selected, labels),
                    "valid_zero_queries": sum(
                        r["outcome"].status == "zero_results"
                        for r in relevant_runs
                    ),
                    "failed_queries": sum(
                        r["outcome"].status not in {"success", "zero_results"}
                        for r in relevant_runs
                    ),
                    "shared_top30_provider_seconds": sum(
                        r["outcome"].seconds for r in relevant_runs
                    ),
                    "shared_http_attempts": sum(
                        r["outcome"].http_attempts for r in relevant_runs
                    ),
                }
            )
    # Saved contemporary controls: KeyBERT OA/OpenAIRE and spaCy arXiv.
    reference = defaultdict(set)
    history_offset = 0
    # Call IDs recur across historical runs: select the actual run objects.
    eligible = {id(c) for c in previous["calls"] if c["kind"] == "alternative"}
    eligible.update(id(c) for c in latest["calls"] if c["kind"] == "original")
    for call in historical_calls:
        if id(call) in eligible:
            for index in range(len(call["papers"])):
                identity = frozen_history.assignments[history_offset + index]
                reference[(call["claim_id"], call["provider"])].add(
                    f"{call['claim_id']}:{identity}"
                )
        history_offset += len(call["papers"])
    variability = []
    for run in runs:
        if run["experiment"] == "depth":
            case, provider = run["claim_id"], run["provider"]
            fresh = {r["key"] for r in by_run[("depth", case, provider)][:10]}
            old = reference[(case, provider)]
            variability.append(
                {
                    "claim_id": case,
                    "provider": provider,
                    "historical_top10_unique": len(old),
                    "fresh_top10_unique": len(fresh),
                    "overlap": len(old & fresh),
                    "fresh_only": len(fresh - old),
                    "historical_only": len(old - fresh),
                }
            )
    all_pairs = defaultdict(list)
    for record in records:
        all_pairs[record["key"]].append(record)
    export = []
    query_lookup = {(q["experiment"], q["query_id"]): q for q in queries}
    for key, members in sorted(all_pairs.items()):
        representative = max(
            members, key=lambda r: len(r["paper"].abstract or "")
        )
        paper = representative["paper"]
        aliases = sorted(
            {f"{r['paper'].source}:{r['paper'].source_id}" for r in members}
            | {
                f"doi:{d}"
                for r in members
                if (d := normalized_doi(r["paper"].doi))
            }
            | {f"arxiv:{a}" for r in members for a in arxiv_ids(r["paper"])}
        )
        export.append(
            {
                "claim_id": representative["claim_id"],
                "identity": key.split(":", 1)[1],
                "pair_key": key,
                "claim": representative["claim"],
                "title": paper.title or "",
                "abstract": paper.abstract or "",
                "identity_aliases": json.dumps(aliases),
                "doi": paper.doi or "",
                "landing_url": str(paper.landing_url or ""),
                "pdf_urls": json.dumps(
                    sorted(
                        {
                            str(p)
                            for r in members
                            for p in [r["paper"].pdf_url, *r["paper"].pdf_urls]
                            if p
                        }
                    )
                ),
                "assessment_status": (
                    "new_claim_paper_pair"
                    if key not in historical_keys
                    else "conflicting_existing_labels"
                    if key in conflicts
                    else "existing_unresolved"
                    if key not in labels
                    else "existing_nonconflicting_provisional"
                ),
                "existing_relevance": labels.get(key, ""),
                "manual_relevance": "",
                "notes": "",
                "provenance": json.dumps(
                    [
                        {
                            "experiment": r["experiment"],
                            "provider": r["provider"],
                            "query": query_lookup[
                                (r["experiment"], r["query_id"])
                            ]["query"],
                            "query_id": r["query_id"],
                            "rank": r["rank"],
                        }
                        for r in members
                    ],
                    ensure_ascii=False,
                ),
            }
        )
    assessment = [row for row in export if row["existing_relevance"] == ""]
    depth_records = [r for r in records if r["experiment"] == "depth"]
    fallback_records = [r for r in records if r["experiment"] == "fallback"]
    primary_ids = {
        q["query_id"]
        for q in queries
        if q["experiment"] == "fallback" and q["stage"] == "primary"
    }
    primary_records = [
        r for r in fallback_records if r["query_id"] in primary_ids
    ]
    primary_keys = {r["key"] for r in primary_records}
    depth_keys = {r["key"] for r in depth_records}
    fallback_keys = {r["key"] for r in fallback_records}
    summary = {
        "labels": {
            "historical_pairs": len(historical_keys),
            "nonconflicting": len(labels),
            "conflicts": sorted(conflicts),
        },
        "depth": provider_depth,
        "depth_per_claim": depth_rows,
        "fallback_per_claim": fallback_rows,
        "fallback": {
            "primary": stats(primary_records, labels),
            "union": stats(fallback_records, labels),
            "retained_union": stats(retained_fallback, labels),
            "additional": stats(
                [r for r in fallback_records if r["key"] not in primary_keys],
                labels,
            ),
            "primary_zero_claims": [
                r["claim_id"]
                for r in fallback_rows
                if not r["primary"]["unique"]
                and r["status"] != "provider_failure"
            ],
            "union_zero_claims": [
                r["claim_id"]
                for r in fallback_rows
                if not r["union"]["unique"]
                and r["status"] != "provider_failure"
            ],
            "http_attempts": sum(r["http_attempts"] for r in fallback_rows),
            "extra_attempts": sum(
                r["fallback_attempts"] for r in fallback_rows
            ),
            "primary_seconds": sum(
                r["primary_seconds"] for r in fallback_rows
            ),
            "extra_seconds": sum(r["fallback_seconds"] for r in fallback_rows),
        },
        "combined_experiments": stats(records, labels),
        "overlap_depth_fallback": len(depth_keys & fallback_keys),
        "assessment": {
            "total": len(assessment),
            "new_pairs": sum(
                r["assessment_status"] == "new_claim_paper_pair"
                for r in assessment
            ),
            "existing_unresolved": sum(
                r["assessment_status"] == "existing_unresolved"
                for r in assessment
            ),
            "existing_conflicts": sum(
                r["assessment_status"] == "conflicting_existing_labels"
                for r in assessment
            ),
        },
        "archive_response_hashes_verified": archive_count,
        "historical_top10_overlap": variability,
        "queries": queries,
        "source_hashes": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in inputs
        },
    }
    output.mkdir(parents=True)
    (output / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(output / "retrieved_pairs.csv", export, list(export[0]))
    write_csv(
        output / "new_unresolved_for_assessment.csv",
        assessment,
        list(export[0]),
    )
    write_csv(
        output / "new_pairs_for_assessment.csv",
        [
            r
            for r in assessment
            if r["assessment_status"] == "new_claim_paper_pair"
        ],
        list(export[0]),
    )
    write_csv(
        output / "fallback_new_for_assessment.csv",
        [
            r
            for r in assessment
            if r["pair_key"] in fallback_keys - primary_keys
        ],
        list(export[0]),
    )
    write_csv(output / "depth_per_claim.csv", depth_rows, list(depth_rows[0]))
    write_csv(output / "query_diagnostics.csv", queries, list(queries[0]))
    write_csv(
        output / "historical_top10_overlap.csv",
        variability,
        list(variability[0]),
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.depth, args.fallback, args.output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "assessment",
                    "fallback",
                    "archive_response_hashes_verified",
                )
            },
            indent=2,
        )
    )
