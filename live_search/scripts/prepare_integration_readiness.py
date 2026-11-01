"""Offline audit, blinded review sample and frozen plans; no API use."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

from paper_search.arxiv_fallback import quote_phrases, relaxation_plan
from paper_search.deduplication import group_papers
from paper_search.evaluation.models import SearchObservation
from paper_search.query_rendering import render_query

ROOT = Path("artifacts")


def load(path: Path) -> dict:
    """Read saved evidence only."""
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict]:
    """Read full descriptions without truncating input evidence."""
    csv.field_size_limit(5_000_000)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(path: Path, records: list[dict], fields: list[str]) -> None:
    """Write a new export, never overwriting labels or an earlier sample."""
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(records)


def save_json(path: Path, value: dict) -> None:
    """Save deterministic, inspectable plans."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def prepare(output: Path) -> dict:
    """Resolve canonical pairs and prepare 47 blind reviews."""
    if output.exists():
        raise FileExistsError(
            "Use a fresh directory; existing evidence is immutable"
        )
    latest_path = ROOT / "arxiv-keybert-20261101/results.json"
    latest = load(latest_path)
    previous = latest["previous"]
    base = previous["baseline"]
    calls = [
        SearchObservation.model_validate(c)
        for c in (
            base["calls"]
            + previous["prior_openaire_calls"]
            + previous["calls"]
            + latest["calls"]
        )
    ]
    entries = [(c, p) for c in calls for p in c.papers]
    grouping = group_papers([p for _, p in entries])
    pairs = {}
    for (call, paper), identity in zip(
        entries, grouping.assignments, strict=True
    ):
        key = f"{call.claim_id}:{identity}"
        record = pairs.setdefault(
            key,
            {
                "key": key,
                "claim_id": call.claim_id,
                "identity": identity,
                "title": paper.title or "",
                "abstract": paper.abstract or "",
                "doi": paper.doi or "",
                "source_id": paper.source_id,
                "providers": set(),
                "labels": [],
                "label_sources": [],
            },
        )
        record["providers"].add(call.provider)
        if len(paper.abstract or "") > len(record["abstract"]):
            record["abstract"] = paper.abstract
    bulk = (
        ROOT / "arxiv-concepts-20261031/manual_assessment_provisional_250.csv"
    )
    keybert = (
        ROOT
        / "keybert-experiment-20261101"
        / "keybert_new_pairs_provisional_labels.csv"
    )
    check = ROOT / "gemini-relevance-20261031/manual_validation_30_filled.csv"
    manifest_path = ROOT / "gemini-relevance-20261031/manifest.json"
    manifest = load(manifest_path)
    mapping = {
        p["id"]: f"{p['claim_id']}:{int(p['identity'])}"
        for p in manifest["pairs"]
    }
    for path, field in (
        (bulk, "manual_relevance"),
        (keybert, "manual_relevance"),
        (check, "human_label"),
    ):
        for row in rows(path):
            raw = row[field].strip()
            key = (
                mapping[row["id"]]
                if path == check
                else (f"{row['claim_id']}:{int(row['identity'])}")
            )
            if key not in pairs:
                raise ValueError(f"Unmatched label identity: {key}")
            if raw:
                value = Decimal(raw)
                if value not in {Decimal(0), Decimal(1), Decimal(2)}:
                    raise ValueError("Invalid provisional label")
                pairs[key]["labels"].append(int(value))
                pairs[key]["label_sources"].append(path.as_posix())
    for record in pairs.values():
        values = set(record["labels"])
        record["status"] = (
            "conflict"
            if len(values) > 1
            else ("unresolved" if not values else "consensus_provisional")
        )
        record["label"] = next(iter(values)) if len(values) == 1 else None
    # Verify historical identities against the frozen mapping as well.
    for pair in manifest["pairs"]:
        key = f"{pair['claim_id']}:{int(pair['identity'])}"
        if (
            key not in pairs
            or pair.get("title", pairs[key]["title"]) != pairs[key]["title"]
        ):
            raise ValueError(
                "Canonical mapping no longer agrees with saved manifest"
            )
    conflicts = [r for r in pairs.values() if r["status"] == "conflict"]
    openaire_export = rows(
        ROOT / "openaire-experiment-20261101/new_pairs_for_assessment.csv"
    )
    unresolved_openaire = []
    for row in openaire_export:
        record = pairs[f"{row['claim_id']}:{int(row['identity'])}"]
        if record["status"] == "unresolved":
            unresolved_openaire.append(record)
    new_arxiv = [r for r in pairs.values() if r["identity"] >= 448]
    selected = {r["key"]: (r, "conflicting_labels") for r in conflicts}
    # Minimum three reviews/claim, considering conflicts first.
    for case in base["cases"]:
        count = sum(r["claim_id"] == case["id"] for r, _ in selected.values())
        candidates = sorted(
            (r for r in unresolved_openaire if r["claim_id"] == case["id"]),
            key=lambda r: r["key"],
        )
        for record in candidates[: max(0, 3 - count)]:
            selected[record["key"]] = (record, "unresolved_openaire")
    for label in (0, 1, 2):
        candidates = sorted(
            (r for r in pairs.values() if r["label"] == label),
            key=lambda r: (r["claim_id"], r["key"]),
        )
        # Round-robin claims for the consensus checks.
        chosen = []
        while candidates and len(chosen) < 5:
            for case in base["cases"]:
                record = next(
                    (r for r in candidates if r["claim_id"] == case["id"]),
                    None,
                )
                if record and len(chosen) < 5:
                    chosen.append(record)
                    candidates.remove(record)
        for record in chosen:
            selected[record["key"]] = (record, f"provisional_label_{label}")
    for record in new_arxiv:
        selected[record["key"]] = (record, "new_arxiv_unlabelled")
    claims = {c["id"]: c["claim"] for c in base["cases"]}
    blind, audit = [], []
    for key, (record, stratum) in sorted(
        selected.items(),
        key=lambda item: hashlib.sha256(item[0].encode()).hexdigest(),
    ):
        review_id = hashlib.sha256(f"readiness-v1:{key}".encode()).hexdigest()[
            :16
        ]
        blind.append(
            {
                "review_id": review_id,
                "claim": claims[record["claim_id"]],
                "title": record["title"],
                "abstract": record["abstract"][:12000],
                "abstract_truncated": len(record["abstract"]) > 12000,
                "reviewer_1_label": "",
                "reviewer_2_label": "",
                "adjudicated_label": "",
                "notes": "",
            }
        )
        audit.append(
            {
                "review_id": review_id,
                "stratum": stratum,
                **record,
                "providers": ";".join(sorted(record["providers"])),
                "labels": json.dumps(record["labels"]),
                "label_sources": ";".join(record["label_sources"]),
            }
        )
    plans = []
    fallback = []
    for case in base["cases"]:
        phrases = base["rendering"][case["id"]]["selected_concepts"]
        for provider in ("openalex", "openaire", "arxiv"):
            query = (
                quote_phrases(phrases)
                if provider == "arxiv"
                else previous["plans"][case["id"]]["keybert"]
            )
            plans.append(
                {
                    "claim_id": case["id"],
                    "claim": case["claim"],
                    "provider": provider,
                    "strategy": "spacy" if provider == "arxiv" else "keybert",
                    "query": query,
                    "required_phrases": phrases if provider == "arxiv" else [],
                }
            )
        fallback.append(
            {
                "claim_id": case["id"],
                "primary": quote_phrases(phrases),
                "required_phrases": phrases,
                "optional_phrases": [],
                "attempts": [
                    {
                        "query": q.query,
                        "reason": q.reason,
                        "omitted_phrases": list(q.omitted_phrases),
                        "rendered": render_query(
                            q.query, "arxiv", arxiv_strategy="literal"
                        ).text,
                    }
                    for q in relaxation_plan(phrases, [])
                ],
            }
        )
    output.mkdir(parents=True)
    fields = [
        "key",
        "claim_id",
        "identity",
        "title",
        "abstract",
        "doi",
        "source_id",
        "providers",
        "labels",
        "label_sources",
        "status",
        "label",
    ]

    def export_records(name: str, records: list[dict]) -> None:
        save_csv(
            output / name,
            [
                {
                    **r,
                    "providers": ";".join(sorted(r["providers"])),
                    "labels": json.dumps(r["labels"]),
                    "label_sources": ";".join(r["label_sources"]),
                }
                for r in records
            ],
            fields,
        )

    export_records("conflicting_labels_12.csv", conflicts)
    export_records("unresolved_openaire.csv", unresolved_openaire)
    save_csv(output / "manual_validation_47.csv", blind, list(blind[0]))
    save_csv(
        output / "manual_validation_audit.csv",
        audit,
        ["review_id", "stratum", *fields],
    )
    save_json(
        output / "depth_plan.json",
        {
            "status": "prepared_not_run",
            "top_k": 30,
            "prefixes": [10, 20, 30],
            "from_publication_date": "2020-01-01",
            "to_publication_date": "2026-10-30",
            "logical_requests": len(plans),
            "max_http_attempts": len(plans),
            "max_retries": 0,
            "arxiv_fallback": False,
            "plans": plans,
        },
    )
    save_json(
        output / "fallback_plan.json",
        {
            "status": "prepared_not_run",
            "top_k": 10,
            "trigger_unique_below": 5,
            "max_fallbacks": 2,
            "max_retries": 0,
            "from_publication_date": "2020-01-01",
            "to_publication_date": "2026-10-30",
            "max_logical_requests": sum(
                1 + len(p["attempts"]) for p in fallback
            ),
            "plans": fallback,
        },
    )
    summary = {
        "historical_pairs": len(pairs),
        "status_counts": dict(Counter(r["status"] for r in pairs.values())),
        "conflicts": sorted(r["key"] for r in conflicts),
        "unresolved_openaire": len(unresolved_openaire),
        "sample_size": len(blind),
        "sample_strata": dict(Counter(r["stratum"] for r in audit)),
        "source_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (latest_path, bulk, keybert, check, manifest_path)
        },
    }
    save_json(output / "audit_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args().output), indent=2))
