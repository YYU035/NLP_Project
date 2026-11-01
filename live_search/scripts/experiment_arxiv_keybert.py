"""Frozen arXiv phrase comparison; no production pipeline edits."""

import argparse
import asyncio
import csv
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx

from paper_search.clients.arxiv import ArxivClient
from paper_search.deduplication import group_papers
from paper_search.evaluation.models import (
    ClaimCase,
    ExperimentOptions,
    SearchObservation,
)
from paper_search.evaluation.runner import ResponseRecorder, _search
from paper_search.query_rendering import render_query

ROOT = Path("artifacts")
PREVIOUS = ROOT / "keybert-experiment-20261101"
DATE_FILTER = "submittedDate:[202001010000 TO 202610302359]"


def read_csv(path: Path) -> list[dict]:
    """Read complete existing descriptions, including unusually long fields."""
    csv.field_size_limit(5000000)
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(
    path: Path, rows: list[dict], fields: list[str]
) -> None:
    """Retain headers even if the experiment finds no novel pairs."""
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def json_write(path: Path, data: dict) -> None:
    """Save human-readable reproducibility evidence."""
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def phrase_input(phrases: list[str]) -> str:
    """Quote existing surface phrases; do not re-extract or invent terms."""
    if not phrases or any(
        not p.strip() or '"' in p or "\\" in p for p in phrases
    ):
        raise ValueError(
            "Expected nonempty unescaped source phrases"
        )
    return " ".join('"' + phrase + '"' for phrase in phrases)


def prepare(output: Path) -> dict:
    """Freeze equal-budget phrase plans and exact syntax before requests."""
    if output.exists():
        raise FileExistsError(
            "Use a fresh directory; preserve previous evidence"
        )
    source = PREVIOUS / "results.json"
    previous = json.loads(source.read_text(encoding="utf-8"))
    base = previous["baseline"]
    if (
        len(base["cases"]) != 10
        or base["options"]["max_results"] != 10
    ):
        raise ValueError(
            "Expected frozen ten-claim top-10 baseline"
        )
    if (
        base["options"]["from_publication_date"] != "2020-01-01"
        or base["options"]["to_publication_date"] != "2026-10-30"
    ):
        raise ValueError(
            "Expected the frozen 2020-01-01 to 2026-10-30 window"
        )
    plans = {}
    for case in base["cases"]:
        case_id = case["id"]
        spacy = base["rendering"][case_id]["selected_concepts"]
        keybert = previous["plans"][case_id]["extraction"][
            "selected_phrases"
        ]
        plans[case_id] = {}
        for strategy, phrases in (
            ("spacy", spacy),
            ("keybert", keybert),
        ):
            query = phrase_input(phrases)
            rendered = render_query(
                query, "arxiv", arxiv_strategy="literal"
            )
            plans[case_id][strategy] = {
                "phrases": phrases,
                "client_input": query,
                "rendered": rendered.text,
                "wire_query": f"({rendered.text}) AND {DATE_FILTER}",
            }
        if (
            plans[case_id]["spacy"]["rendered"]
            != base["rendering"][case_id]["text"]
        ):
            raise ValueError(
                "spaCy control no longer matches saved arXiv syntax"
            )
    bulk_path = (
        ROOT
        / "arxiv-concepts-20261031/manual_assessment_provisional_250.csv"
    )
    new_path = PREVIOUS / "keybert_new_pairs_provisional_labels.csv"
    check_path = (
        ROOT
        / "gemini-relevance-20261031/manual_validation_30_filled.csv"
    )
    map_path = ROOT / "gemini-relevance-20261031/manifest.json"
    observations: dict[str, list[int]] = defaultdict(list)
    for path in (bulk_path, new_path):
        for row in read_csv(path):
            raw = row["manual_relevance"].strip()
            if raw:
                value = Decimal(raw)
                if value not in {
                    Decimal(0),
                    Decimal(1),
                    Decimal(2),
                }:
                    raise ValueError("Invalid provisional label")
                observations[
                    f"{row['claim_id']}:{int(row['identity'])}"
                ].append(int(value))
    label_map = json.loads(map_path.read_text(encoding="utf-8"))
    ids = {
        p["id"]: f"{p['claim_id']}:{int(p['identity'])}"
        for p in label_map["pairs"]
    }
    for row in read_csv(check_path):
        if row["human_label"].strip():
            value = Decimal(row["human_label"])
            if value not in {Decimal(0), Decimal(1), Decimal(2)}:
                raise ValueError("Invalid sample label")
            observations[ids[row["id"]]].append(int(value))
    labels = {
        key: values[0]
        for key, values in observations.items()
        if len(set(values)) == 1
    }
    conflicts = [
        key
        for key, values in observations.items()
        if len(set(values)) > 1
    ]
    manifest = {
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "previous": previous,
        "source_hashes": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                source,
                bulk_path,
                new_path,
                check_path,
                map_path,
            )
        },
        "options": base["options"],
        "plans": plans,
        "translation": "existing literal mode with explicit source phrases",
        "labels": labels,
        "conflicting_label_keys": conflicts,
        "max_requests": 20,
        "max_retries": 0,
    }
    output.mkdir(parents=True)
    json_write(output / "run.json", manifest)
    return manifest


def analyze(manifest: dict, output: Path) -> dict:
    """Join saved labels by canonical pair and calculate union without I/O."""
    previous = manifest["previous"]
    history_calls = (
        previous["baseline"]["calls"]
        + previous["prior_openaire_calls"]
        + previous["calls"]
    )
    calls = [
        SearchObservation.model_validate(c)
        for c in manifest["calls"]
    ]
    history = [
        SearchObservation.model_validate(c) for c in history_calls
    ]
    entries = [
        (c, rank, p)
        for c in history + calls
        for rank, p in enumerate(c.papers, 1)
    ]
    assigned = group_papers([p for _, _, p in entries]).assignments
    split = sum(len(c.papers) for c in history)
    historic = {
        (c.claim_id, identity)
        for (c, _, _), identity in zip(
            entries[:split], assigned[:split], strict=True
        )
    }
    cases = {c["id"]: c for c in previous["baseline"]["cases"]}
    labels = manifest["labels"]
    conflicts = set(manifest["conflicting_label_keys"])
    rows = []
    for (c, rank, p), identity in zip(
        entries[split:], assigned[split:], strict=True
    ):
        key = f"{c.claim_id}:{identity}"
        rows.append(
            {
                "claim_id": c.claim_id,
                "identity": identity,
                "claim": cases[c.claim_id]["claim"],
                "provider": "arxiv",
                "strategy": "spacy"
                if c.kind == "original"
                else "keybert",
                "rank": rank,
                "query": c.query,
                "wire_query": c.attempts[0]["params"][
                    "search_query"
                ],
                "source_id": p.source_id,
                "title": p.title,
                "abstract": p.abstract,
                "doi": p.doi,
                "publication_date": p.publication_date,
                "paper_url": str(p.landing_url or ""),
                "pdf_urls": json.dumps(
                    [str(u) for u in p.pdf_urls]
                ),
                "reused_label": labels.get(key, ""),
                "label_status": "conflict"
                if key in conflicts
                else "known_provisional"
                if key in labels
                else "unknown",
                "historically_new": (c.claim_id, identity)
                not in historic,
                "manual_relevance": "",
                "manual_notes": "",
            }
        )
    fields = [
        "claim_id",
        "identity",
        "claim",
        "provider",
        "strategy",
        "rank",
        "query",
        "wire_query",
        "source_id",
        "title",
        "abstract",
        "doi",
        "publication_date",
        "paper_url",
        "pdf_urls",
        "reused_label",
        "label_status",
        "historically_new",
        "manual_relevance",
        "manual_notes",
    ]
    write_csv(output / "retrieved_papers.csv", rows, fields)
    pairs = defaultdict(list)
    for row in rows:
        pairs[(row["claim_id"], row["identity"])].append(row)
    new_rows = []
    unknown_rows = []
    for key, versions in sorted(pairs.items()):
        row = dict(
            max(versions, key=lambda r: len(r["abstract"] or ""))
        )
        row["provenance"] = json.dumps(
            [
                {
                    k: r[k]
                    for k in (
                        "strategy",
                        "rank",
                        "wire_query",
                        "source_id",
                    )
                }
                for r in versions
            ],
            ensure_ascii=False,
        )
        if key not in historic:
            new_rows.append(row)
        if row["reused_label"] == "":
            unknown_rows.append(row)
    write_csv(
        output / "new_pairs_for_assessment.csv",
        new_rows,
        fields + ["provenance"],
    )
    write_csv(
        output / "unresolved_pairs_for_assessment.csv",
        unknown_rows,
        fields + ["provenance"],
    )

    def summary(strategy: str, claim: str | None = None) -> dict:
        selected = [
            r
            for r in rows
            if (strategy == "union" or r["strategy"] == strategy)
            and (claim is None or r["claim_id"] == claim)
        ]
        keys = {(r["claim_id"], r["identity"]) for r in selected}
        return {
            "records": len(selected),
            "unique_pairs": len(keys),
            "relevant": sum(
                labels.get(f"{c}:{i}", -1) >= 1 for c, i in keys
            ),
            "evidence": sum(
                labels.get(f"{c}:{i}", -1) == 2 for c, i in keys
            ),
            "unknown": sum(
                f"{c}:{i}" not in labels for c, i in keys
            ),
            "with_abstract": sum(
                any(
                    r["abstract"]
                    for r in selected
                    if (r["claim_id"], r["identity"]) == key
                )
                for key in keys
            ),
            "with_pdf_url": sum(
                any(
                    json.loads(r["pdf_urls"])
                    for r in selected
                    if (r["claim_id"], r["identity"]) == key
                )
                for key in keys
            ),
        }

    totals = {s: summary(s) for s in ("spacy", "keybert", "union")}
    per_claim = []
    for case_id in cases:
        per_claim.append(
            {
                "claim_id": case_id,
                **{
                    s: summary(s, case_id)
                    for s in ("spacy", "keybert", "union")
                },
            }
        )
    sets = {
        s: {
            (r["claim_id"], r["identity"])
            for r in rows
            if r["strategy"] == s
        }
        for s in ("spacy", "keybert")
    }
    novelty = {
        "historical_pairs": len(historic),
        "new_pairs": len(new_rows),
        "overlap": len(sets["spacy"] & sets["keybert"]),
        "keybert_only": len(sets["keybert"] - sets["spacy"]),
        "spacy_only": len(sets["spacy"] - sets["keybert"]),
    }
    diagnostics = []
    for c in calls:
        total = None
        for attempt in c.attempts:
            if attempt.get("status") == 200:
                try:
                    root = ET.fromstring(
                        (
                            output / attempt["response_file"]
                        ).read_bytes()
                    )
                    total = int(
                        root.findtext(
                            "{http://a9.com/-/spec/opensearch/1.1/}totalResults"
                        )
                    )
                except (ET.ParseError, TypeError, ValueError):
                    pass
        strategy = "spacy" if c.kind == "original" else "keybert"
        diagnostics.append(
            {
                "claim_id": c.claim_id,
                "strategy": strategy,
                "query": c.query,
                "wire_query": manifest["plans"][c.claim_id][
                    strategy
                ]["wire_query"],
                "status": c.status,
                "returned": c.result_count,
                "reported_total": total,
                "http_attempts": len(c.attempts),
                "seconds": c.elapsed_seconds,
                "capped": c.result_count == 10
                and isinstance(total, int)
                and total > 10,
            }
        )
    write_csv(
        output / "per_query.csv",
        diagnostics,
        list(diagnostics[0]) if diagnostics else ["claim_id"],
    )
    result = {
        **manifest,
        "totals": totals,
        "per_claim": per_claim,
        "novelty": novelty,
        "diagnostics": diagnostics,
    }
    json_write(output / "results.json", result)
    write_report(result, output)
    return result


def write_report(result: dict, output: Path) -> None:
    """Report the frozen syntax, observed failures/zeros and uncertainty."""
    lines = [
        "# arXiv spaCy versus KeyBERT: frozen-phrase comparison",
        "",
        "## Method and translation",
        "",
        "Same ten claims, dates 2020-01-01–2026-10-30, top 10, one "
        "request per method/claim. Order alternates by claim. At most "
        "20 public arXiv requests, three-second minimum pacing, no retries; "
        "stop on a failed query. No OpenAlex/OpenAIRE, model inference, "
        "relevance API or PDF requests. Union is calculated offline.",
        "",
        "**Important:** default ArxivClient concepts mode re-extracts "
        "spaCy concepts and could overwrite KeyBERT choices. This experiment "
        "explicitly uses the ALREADY EXISTING `query_strategy=literal` "
        "mode for BOTH methods. Frozen source phrases are individually "
        "quoted before the existing renderer joins them using AND. "
        "No production code/default changed. The spaCy wire expressions "
        "were checked against all ten saved 31 October expressions; they "
        "match exactly. KeyBERT uses the frozen previously tested phrases, "
        "not a spaCy re-selection or regenerated model output.",
        "",
        "[Official arXiv syntax]"
        "(https://info.arxiv.org/help/api/user-manual.html) "
        "supports all: fields, exact phrases and AND conjunctions. "
        "all: searches indexed metadata, not article-body full text. "
        "The full UTC submission-date suffix is shown below; it is "
        "not necessarily a journal publication-date restriction.",
        "",
        "## Aggregate results",
        "",
        "| Method | Records / unique pairs | Abstracts | PDF URLs | "
        "Known relevant / evidence | Unresolved |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strategy, s in result["totals"].items():
        lines.append(
            f"| {strategy} | {s['records']} / {s['unique_pairs']} | "
            f"{s['with_abstract']} | {s['with_pdf_url']} | "
            f"{s['relevant']} / {s['evidence']} | {s['unknown']} |"
        )
    n = result["novelty"]
    lines.extend(
        [
            "",
            f"Overlap: **{n['overlap']}**; KeyBERT-only: "
            f"**{n['keybert_only']}**; spaCy-only: **{n['spacy_only']}**. "
            f"Historically new pairs: **{n['new_pairs']}**, relative to "
            f"the full previous **{n['historical_pairs']}**-pair collection.",
            "",
            "Union records count all returned occurrences; unique pairs "
            "are deduplicated by claim and the existing DOI/arXiv-ID/"
            "conservative-title matcher. Union needs no extra requests.",
            "",
            "| Method | HTTP attempts | Valid zero queries | "
            "Failures | Latency sum |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for strategy in ("spacy", "keybert"):
        ds = [
            d
            for d in result["diagnostics"]
            if d["strategy"] == strategy
        ]
        failures = sum(
            d["status"] not in {"success", "zero_results"} for d in ds
        )
        lines.append(
            f"| {strategy} | {sum(d['http_attempts'] for d in ds)} | "
            f"{sum(d['status'] == 'zero_results' for d in ds)} | "
            f"{failures}"
            " | "
            f"{sum(d['seconds'] for d in ds):.2f} s |"
        )
    lines.extend(
        [
            "",
            "## Per-claim results",
            "",
            "T/U = returned/unique; R/E = known relevant/evidence; ? = "
            "unresolved. A failed/unattempted query is unavailable, not zero. "
            "See per_query.csv for API total matches and status.",
            "",
            "| Claim | spaCy T/U; R/E; ? | KeyBERT T/U; R/E; ? | "
            "Union unique | Overlap |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in result["per_claim"]:
        cells = []
        for strategy in ("spacy", "keybert"):
            ds = [
                d
                for d in result["diagnostics"]
                if d["claim_id"] == row["claim_id"]
                and d["strategy"] == strategy
            ]
            s = row[strategy]
            cells.append(
                f"{s['records']}/{s['unique_pairs']}; "
                f"{s['relevant']}/{s['evidence']}; {s['unknown']}"
                if ds
                and ds[0]["status"] in {"success", "zero_results"}
                else "unavailable"
            )
        overlap = (
            row["spacy"]["unique_pairs"]
            + row["keybert"]["unique_pairs"]
            - row["union"]["unique_pairs"]
        )
        lines.append(
            f"| {row['claim_id']} | {cells[0]} | {cells[1]} | "
            f"{row['union']['unique_pairs']} | {overlap} |"
        )
    lines.extend(
        [
            "",
            "## Exact submitted arXiv expressions",
            "",
            "Every expression includes the actual date suffix. Source "
            "phrase lists and client-input text are frozen in run.json.",
            "",
        ]
    )
    lines.extend([
        "Each expression below was sent as `(<expression>) AND "
        "submittedDate:[202001010000 TO 202610302359]`. "
        "Complete wire strings are recorded in per_query.csv and calls.jsonl.",
        "", "| Claim | spaCy expression | KeyBERT expression |",
        "|---|---|---|",
    ])
    for case_id, plans in result["plans"].items():
        lines.append(f"| {case_id} | `{plans['spacy']['rendered']}` | "
                     f"`{plans['keybert']['rendered']}` |")
    lines.extend(
        [
            "## Syntax diagnosis and decision",
            "",
            "Quotes require phrase order; AND requires every quoted "
            "concept to match. KeyBERT commonly adds conditions or "
            "comparators (older adults, 90% capacity, -5 °C, 100 kHz, "
            "coastal waters, random forests, acidic solutions). These "
            "can plausibly exclude otherwise topical papers. A valid "
            "zero is not evidence that the scientific claim is false "
            "or that no relevant article exists outside indexed metadata.",
            "",
            "No phrases, AND operators or dates were relaxed during "
            "the comparison, and no extra backoff search was made. "
            "Thus per-term causality, notation variants, date effects "
            "and corpus coverage cannot be separated. Compare paired "
            "outcomes as observations, not proof that quoting alone "
            "caused a zero. A ten-result list can be capped; inspect "
            "reported_total in per_query.csv.",
            "",
        ]
    )
    complete = len(result["diagnostics"]) == 20 and all(
        d["status"] in {"success", "zero_results"}
        for d in result["diagnostics"]
    )
    if not complete:
        lines.append("**The comparison is incomplete because calls failed or "
                     "were not attempted. Do not infer strategy quality "
                     "from the partial candidate counts.**")
    elif n["keybert_only"] == 0:
        lines.append(
            "**No additional arXiv identities from KeyBERT were observed. "
            "Do not mandate a second arXiv query from this evidence; "
            "retain spaCy there while keeping KeyBERT optional.**"
        )
    else:
        lines.append(
            "**KeyBERT supplies additional arXiv identities, but their "
            "relevance must be assessed before claiming a benefit. "
            "Keep spaCy and use the deduplicated union only if its "
            "additional request/candidate cost is acceptable.**"
        )
    lines.extend(
        [
            "",
            "A uniform two-generator interface is feasible with "
            "provider-specific syntax, but the current default arXiv "
            "concept renderer must not be assumed to preserve KeyBERT "
            "phrases. The explicit literal-mode binding tested here "
            "would need deliberate integration; production is unchanged. "
            "OpenAlex/OpenAIRE strategy findings do not prove equal "
            "benefit on arXiv.",
            "",
            "## Labels, exports and limitations",
            "",
            "Existing bulk250/new120/30-example assessments were frozen "
            "and matched by canonical claim–paper identity. Twelve "
            "cross-source conflicts remain unresolved, consistent with "
            "the previous final comparison. Missing labels are not "
            "irrelevant; all labels are provisional AI-assisted judgments. "
            "No labels are invented for new papers.",
            "",
            "retrieved_papers.csv contains all occurrences, syntax and "
            "reused labels. new_pairs_for_assessment.csv contains only "
            "historically new pairs; unresolved_pairs_for_assessment.csv "
            "also covers earlier unlabelled/conflicting pairs. Both have "
            "blank manual-label columns and all retrieval provenance. "
            "Raw responses, errors and SHA-256 values are in calls.jsonl "
            "and responses/. Metadata links were not downloaded.",
            "",
            "This small capped sample, native index variability, "
            "strict phrases and incomplete labels limit conclusions. "
            "There is no new relevance accuracy, significance, corpus "
            "recall, or full-text claim verification result.",
        ]
    )
    (output / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


async def run(output: Path) -> dict:
    """Use existing translation with frozen explicit phrase boundaries."""
    manifest = json.loads(
        (output / "run.json").read_text(encoding="utf-8")
    )
    if (output / "calls.jsonl").exists():
        raise FileExistsError(
            "Existing calls found; reanalyze without new requests"
        )
    for name, expected in manifest["source_hashes"].items():
        if (
            hashlib.sha256(Path(name).read_bytes()).hexdigest()
            != expected
        ):
            raise ValueError("Frozen source changed")
    options = ExperimentOptions.model_validate(manifest["options"])
    recorder = ResponseRecorder(output, "arxiv")
    recorder.endpoint = ArxivClient.endpoint
    calls = []
    async with httpx.AsyncClient(
        event_hooks={
            "request": [recorder.on_request],
            "response": [recorder.on_response],
        }
    ) as http:
        client = ArxivClient(
            http, max_retries=0, timeout_seconds=30
        )
        client.query_strategy = "literal"
        for index, raw in enumerate(
            manifest["previous"]["baseline"]["cases"]
        ):
            case = ClaimCase.model_validate(raw)
            jobs = [
                ("original", "spacy"),
                ("alternative", "keybert"),
            ]
            for kind, strategy in (
                jobs if index % 2 == 0 else reversed(jobs)
            ):
                if len(calls) >= 20:
                    break
                plan = manifest["plans"][case.id][strategy]
                call = await _search(
                    client,
                    recorder,
                    case,
                    kind,
                    plan["client_input"],
                    options,
                )
                calls.append(call)
                if call.attempts and (
                    call.attempts[0]["params"]["search_query"]
                    != plan["wire_query"]
                ):
                    raise RuntimeError(
                        "Wire expression differs from frozen plan"
                    )
                print(
                    f"{case.id} {strategy}: {call.status}; "
                    f"papers={call.result_count}",
                    flush=True,
                )
                if not call.succeeded:
                    break
            if calls and not calls[-1].succeeded:
                break
    manifest["calls"] = [c.model_dump(mode="json") for c in calls]
    manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
    return analyze(manifest, output)


def main() -> None:
    """Prepare/reanalyze offline; only explicit run can contact arXiv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "run", "report")
    )
    parser.add_argument("--run-live", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "arxiv-keybert-20261101",
    )
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output)
    elif args.action == "run":
        if not args.run_live:
            parser.error("--run-live required; no requests made")
        result = asyncio.run(run(args.output))
        print(json.dumps(result["novelty"]))
    else:
        manifest = json.loads(
            (args.output / "run.json").read_text(encoding="utf-8")
        )
        manifest["calls"] = [
            json.loads(line)
            for line in (args.output / "calls.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        analyze(manifest, args.output)


if __name__ == "__main__":
    main()
