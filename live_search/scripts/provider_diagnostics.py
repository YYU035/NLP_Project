"""Opt-in, bounded provider probes; not new production API adapters."""

import argparse
import asyncio
import csv
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from xml.etree import ElementTree as ET

import httpx

from paper_search.clients.arxiv import ArxivClient, parse_entry
from paper_search.clients.semantic_scholar import (
    SemanticScholarClient,
)
from paper_search.config import Settings
from paper_search.deduplication import normalized_doi
from paper_search.evaluation.models import SearchObservation
from paper_search.evaluation.runner import ResponseRecorder
from paper_search.exceptions import APIClientError
from paper_search.query_rendering import render_query

DATE_RANGE = "submittedDate:[202001010000 TO 202610302359]"


@dataclass(frozen=True)
class Probe:
    """An explicit request and the constraints intentionally changed."""

    id: str
    provider: str
    endpoint: str
    params: dict[str, str]
    variant: str
    claim_id: str | None = None
    dated: bool = False
    relaxation: str = "none"


def arxiv_probes(saved: dict[str, Any]) -> list[Probe]:
    """Compare date/no-date pairs for three saved scientific claims."""
    endpoint = ArxivClient.endpoint
    rows = [
        Probe(
            "arxiv-control-id",
            "arxiv",
            endpoint,
            {"id_list": "1706.03762", "max_results": "3"},
            "known_id",
        ),
    ]
    controls = [
        ("control-title", 'ti:"Attention Is All You Need"', False),
        (
            "control-title-dated",
            'ti:"Attention Is All You Need"',
            True,
        ),
        ("control-electron", "all:electron", True),
    ]
    for label, query, dated in controls:
        expression = (
            f"({query}) AND {DATE_RANGE}" if dated else query
        )
        rows.append(
            Probe(
                "arxiv-" + label,
                "arxiv",
                endpoint,
                {
                    "search_query": expression,
                    "start": "0",
                    "max_results": "3",
                },
                "known_positive_or_date_control",
                dated=dated,
            )
        )
    specifications = {
        "c01": (
            "TNF-α COVID-19",
            '"expression of TNF-α" "COVID-19"',
            "Drops negated risk predicate and older-adult population; "
            "topic discovery only, not an equivalent scientific claim.",
        ),
        "c02": (
            "lithium-ion batteries capacity",
            '"lithium-ion batteries" capacity',
            "Drops retention predicate, 90%, 500 cycles, and -5 °C; "
            "keeps the literal battery terminology.",
        ),
        "c08": (
            "graph neural networks molecular property prediction",
            '"graph neural networks" "molecular property prediction"',
            "Drops improvement assertion and random-forest comparator; "
            "topic discovery only.",
        ),
    }
    by_id = {case["id"]: case for case in saved["cases"]}
    for case_id, (broad, phrases, note) in specifications.items():
        for variant, text, relaxation in (
            ("current", by_id[case_id]["claim"], "none"),
            ("broad", broad, note),
            ("phrases", phrases, note),
        ):
            rendered = render_query(
                text, "arxiv", arxiv_strategy="literal"
            ).text
            for dated in (True, False):
                expression = (
                    f"({rendered}) AND {DATE_RANGE}"
                    if dated
                    else rendered
                )
                rows.append(
                    Probe(
                        f"arxiv-{case_id}-{variant}-"
                        f"{'dated' if dated else 'all-years'}",
                        "arxiv",
                        endpoint,
                        {
                            "search_query": expression,
                            "start": "0",
                            "max_results": "3",
                        },
                        variant,
                        case_id,
                        dated,
                        relaxation,
                    )
                )
    for label, text, dated, note in (
        (
            "covid-topic",
            "COVID-19",
            True,
            "Drops all TNF/predicate/population constraints.",
        ),
        (
            "tnf-alpha",
            "TNF-α",
            False,
            "Single literal target; all other constraints removed.",
        ),
        (
            "tnf-family",
            "TNF",
            False,
            "Drops α specificity; TNF is not an asserted synonym.",
        ),
        (
            "tnf-family-covid",
            "TNF COVID-19",
            True,
            "Drops α specificity and predicate/population; "
            "diagnostic relaxation only.",
        ),
    ):
        query = render_query(text, "arxiv", arxiv_strategy="literal").text
        if dated:
            query = f"({query}) AND {DATE_RANGE}"
        rows.append(
            Probe(
                "arxiv-" + label,
                "arxiv",
                endpoint,
                {
                    "search_query": query,
                    "start": "0",
                    "max_results": "3",
                },
                "literal_sensitivity",
                "c01",
                dated,
                note,
            )
        )
    return rows


def summarize_response(
    provider: str, response: httpx.Response
) -> dict[str, Any]:
    """Inspect research fields without adding production API adapters."""
    if not response.is_success:
        return {
            "status": "http_error",
            "http_status": response.status_code,
            "result_count": None,
        }
    if provider == "arxiv":
        root = ET.fromstring(response.content)
        if root.tag != "{http://www.w3.org/2005/Atom}feed":
            return {
                "status": "unexpected_xml",
                "result_count": None,
            }
        entries = root.findall("{http://www.w3.org/2005/Atom}entry")
        if any(
            "/api/errors"
            in (
                entry.findtext("{http://www.w3.org/2005/Atom}id")
                or ""
            )
            for entry in entries
        ):
            return {"status": "atom_error", "result_count": None}
        papers = [
            parse_entry(entry).model_dump(mode="json")
            for entry in entries
        ]
        return {
            "status": "success" if papers else "zero_results",
            "result_count": len(papers),
            "total_hits": root.findtext(
                "{http://a9.com/-/spec/opensearch/1.1/}totalResults"
            ),
            "records": papers,
        }
    if provider == "europe_pmc_xml":
        root = ET.fromstring(response.content)
        return {
            "status": "success",
            "root_tag": root.tag,
            "bytes": len(response.content),
            "body_present": root.find("body") is not None,
            "article_title": " ".join(
                root.findtext(
                    ".//article-title", default=""
                ).split()
            ),
        }
    payload = response.json()
    if not isinstance(payload, dict):
        return {
            "status": "unexpected_envelope",
            "result_count": None,
        }
    if provider == "semantic_scholar":
        records = payload.get("data")
        if records is None and payload == {"total": 0, "offset": 0}:
            records = []
        if not isinstance(records, list):
            return {
                "status": "unexpected_envelope",
                "result_count": None,
            }
        return {
            "status": "success" if records else "zero_results",
            "result_count": len(records),
            "total_hits": payload.get("total"),
            "records": records,
        }
    if provider == "europe_pmc":
        records = payload.get("resultList", {}).get("result")
        if (
            not isinstance(records, list)
            or "hitCount" not in payload
        ):
            return {
                "status": "unexpected_envelope",
                "result_count": None,
            }
        return {
            "status": "success" if records else "zero_results",
            "total_hits": payload.get("hitCount"),
            "result_count": len(records),
            "records": [
                {
                    key: row.get(key)
                    for key in (
                        "id",
                        "source",
                        "pmcid",
                        "doi",
                        "title",
                        "abstractText",
                        "firstPublicationDate",
                        "authorString",
                        "isOpenAccess",
                        "fullTextUrlList",
                    )
                }
                for row in records
            ],
        }
    if provider == "core":
        records = payload.get("results")
        if not isinstance(records, list):
            return {
                "status": "unexpected_envelope",
                "keys": list(payload),
                "result_count": None,
            }
        return {
            "status": "success" if records else "zero_results",
            "total_hits": payload.get("totalHits"),
            "result_count": len(records),
            "records": [
                {
                    key: row.get(key)
                    for key in (
                        "id",
                        "title",
                        "abstract",
                        "doi",
                        "yearPublished",
                        "fullText",
                        "downloadUrl",
                        "sourceFulltextUrls",
                    )
                }
                for row in records
            ],
        }
    if provider == "openaire":
        records = payload.get("results")
        if not isinstance(records, list):
            return {
                "status": "unexpected_envelope",
                "keys": list(payload),
                "result_count": None,
            }
        return {
            "status": "success" if records else "zero_results",
            "total_hits": payload.get("header", {}).get("numFound"),
            "result_count": len(records),
            "records": [
                {
                    key: row.get(key)
                    for key in (
                        "id",
                        "mainTitle",
                        "descriptions",
                        "publicationDate",
                        "pids",
                        "bestAccessRight",
                        "instances",
                    )
                }
                for row in records
            ],
        }
    if provider == "crossref":
        message = payload.get("message", {})
        records = message.get("items", [message])
        return {
            "status": "success" if records else "zero_results",
            "total_hits": message.get("total-results", 1),
            "result_count": len(records),
            "records": [
                {
                    key: row.get(key)
                    for key in (
                        "DOI",
                        "title",
                        "abstract",
                        "published",
                        "URL",
                        "link",
                        "license",
                    )
                }
                for row in records
            ],
        }
    return {"status": "unparsed", "result_count": None}


def arxiv_followup_probes(saved: dict[str, Any]) -> list[Probe]:
    """Reproduce saved focused queries and isolate a known date exclusion."""
    probes = [
        Probe(
            "arxiv-id-date-exclusion",
            "arxiv",
            ArxivClient.endpoint,
            {
                "id_list": "1706.03762",
                "search_query": DATE_RANGE,
                "max_results": "3",
            },
            "known_id_outside_date_range",
            dated=True,
        )
    ]
    for claim_id in ("c01", "c02", "c08"):
        focused = saved["generation"][claim_id]["queries"][1]
        rendered = render_query(
            focused, "arxiv", arxiv_strategy="literal"
        ).text
        for dated in (True, False):
            expression = (
                f"({rendered}) AND {DATE_RANGE}" if dated else rendered
            )
            probes.append(
                Probe(
                    f"arxiv-{claim_id}-saved-focused-{dated}",
                    "arxiv",
                    ArxivClient.endpoint,
                    {"search_query": expression, "max_results": "3"},
                    "saved_focused",
                    claim_id,
                    dated,
                )
            )
    return probes


async def run(
    output: Path, saved_path: Path, *, followup: bool = False
) -> dict[str, Any]:
    """Execute a finite, paced plan; stop a provider immediately on 429."""
    if (output / "summary.json").exists():
        raise FileExistsError(
            "Use a new directory to preserve evidence"
        )
    output.mkdir(parents=True, exist_ok=True)
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    settings = Settings.from_environment()
    rows: list[dict[str, Any]] = []

    async def group(
        probes: list[Probe],
        *,
        pace: float = 1.5,
        auth: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        observed = []
        recorder = ResponseRecorder(output, probes[0].provider)
        recorder.auth = auth or {}
        async with httpx.AsyncClient(
            timeout=20,
            follow_redirects=False,
            headers={
                "User-Agent": "NTU-NLP-provider-diagnostics/0.1"
            },
            event_hooks={
                "request": [recorder.on_request],
                "response": [recorder.on_response],
            },
        ) as http:
            arxiv = ArxivClient(
                http, max_retries=0, timeout_seconds=20
            )
            for index, probe in enumerate(probes):
                if index and probe.provider != "arxiv":
                    await asyncio.sleep(pace)
                recorder.endpoint = probe.endpoint
                recorder.current = SearchObservation(
                    id=probe.id,
                    provider=probe.provider,
                    claim_id=probe.claim_id or "control",
                    kind="original",
                    query=json.dumps(probe.params),
                    authentication="key_supplied"
                    if auth
                    else "anonymous",
                )
                started = time.perf_counter()
                try:
                    response = (
                        await arxiv._get(probe.params)
                        if probe.provider == "arxiv"
                        else await http.get(
                            probe.endpoint,
                            params=probe.params,
                            headers=auth or {},
                        )
                    )
                    summary = summarize_response(
                        probe.provider, response
                    )
                except APIClientError as exc:
                    summary = {
                        "status": "provider_error",
                        "error": str(exc),
                        "result_count": None,
                    }
                except (
                    httpx.RequestError,
                    ValueError,
                    ET.ParseError,
                ) as exc:
                    summary = {
                        "status": "transport_or_parse_error",
                        "error_type": type(exc).__name__,
                        "result_count": None,
                    }
                row = {
                    **asdict(probe),
                    **summary,
                    "seconds": time.perf_counter() - started,
                    "attempts": recorder.current.attempts,
                }
                rows.append(row)
                observed.append(row)
                with (output / "requests.jsonl").open(
                    "a", encoding="utf-8"
                ) as file:
                    file.write(
                        json.dumps(row, ensure_ascii=False) + "\n"
                    )
                print(
                    f"{probe.id}: {summary['status']} "
                    f"results={summary.get('result_count')} "
                    f"total={summary.get('total_hits')}",
                    flush=True,
                )
                statuses = [
                    attempt["status"]
                    for attempt in recorder.current.attempts
                ]
                if 429 in statuses or any(
                    s in (401, 403) for s in statuses
                ):
                    # Stop rather than retrying access limits.
                    break
        return observed

    async def alternatives() -> None:
        epmc = await group(
            [
                Probe(
                    "epmc-tnf-covid",
                    "europe_pmc",
                    "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                    {
                        "query": '"TNF-α" AND "COVID-19" AND '
                        'FIRST_PDATE:[2020-01-01 TO 2026-10-30]',
                        "format": "json",
                        "resultType": "core",
                        "pageSize": "3",
                    },
                    "keyword_topic",
                ),
                Probe(
                    "epmc-exercise-oa",
                    "europe_pmc",
                    "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                    {
                        "query": "exercise AND memory AND OPEN_ACCESS:Y "
                        "AND FIRST_PDATE:[2020-01-01 TO 2026-10-30]",
                        "format": "json",
                        "resultType": "core",
                        "pageSize": "3",
                    },
                    "keyword_oa_topic",
                ),
            ]
        )
        candidates = [
            record["pmcid"]
            for row in epmc
            for record in row.get("records", [])
            if record.get("pmcid")
            and record.get("isOpenAccess") == "Y"
        ]
        if (
            candidates
            and re.fullmatch(r"PMC\d+", candidates[0])
            and not any(
                a["status"] == 429
                for row in epmc
                for a in row["attempts"]
            )
        ):
            await group(
                [
                    Probe(
                        "epmc-one-xml",
                        "europe_pmc_xml",
                        "https://www.ebi.ac.uk/europepmc/webservices/rest/"
                        + candidates[0]
                        + "/fullTextXML",
                        {},
                        "single_oa_xml_availability",
                    )
                ]
            )
        core_key = os.getenv("CORE_API_KEY") or None
        await group(
            [
                Probe(
                    "core-graph-neural",
                    "core",
                    "https://api.core.ac.uk/v3/search/works/",
                    {"q": '"graph neural networks"', "limit": "3"},
                    "keyword_topic",
                )
            ],
            auth={"Authorization": "Bearer " + core_key}
            if core_key
            else None,
        )
        token = os.getenv("OPENAIRE_TOKEN") or None
        await group(
            [
                Probe(
                    "openaire-graph-neural",
                    "openaire",
                    "https://api.openaire.eu/graph/v3/research-products",
                    {
                        "search": "graph neural networks",
                        "type": "publication",
                        "pageSize": "3",
                        "fromPublicationDate": "2020-01-01",
                        "toPublicationDate": "2026-10-30",
                    },
                    "keyword_topic",
                ),
                Probe(
                    "openaire-tnf-covid",
                    "openaire",
                    "https://api.openaire.eu/graph/v3/research-products",
                    {
                        "search": "TNF-α COVID-19",
                        "type": "publication",
                        "pageSize": "3",
                        "fromPublicationDate": "2020-01-01",
                        "toPublicationDate": "2026-10-30",
                    },
                    "keyword_topic",
                ),
            ],
            pace=2,
            auth={"Authorization": "Bearer " + token}
            if token
            else None,
        )
        doi = next(
            normalized_doi(p["doi"])
            for call in saved["calls"]
            for p in call["papers"]
            if p.get("doi")
        )
        await group(
            [
                Probe(
                    "crossref-battery",
                    "crossref",
                    "https://api.crossref.org/works",
                    {
                        "query.bibliographic": (
                            "lithium-ion batteries capacity"
                        ),
                        "rows": "3",
                        "filter": "from-pub-date:2020-01-01,"
                        "until-pub-date:2026-10-30",
                    },
                    "bibliographic_topic",
                ),
                Probe(
                    "crossref-one-doi",
                    "crossref",
                    "https://api.crossref.org/works/"
                    + quote(doi, safe="/"),
                    {},
                    "doi_enrichment",
                ),
            ],
            pace=2,
        )

    async def semantic_once() -> None:
        # An absent key is disclosed. At most one anonymous control is used;
        # never repeat a 429, regardless of whether Retry-After is present.
        key = settings.semantic_scholar_api_key
        await group(
            [
                Probe(
                    "semantic-single-control",
                    "semantic_scholar",
                    SemanticScholarClient.endpoint,
                    {
                        "query": "graph neural networks",
                        "limit": "3",
                        "fields": "title,abstract,publicationDate,"
                        "openAccessPdf",
                    },
                    "one_no_retry_control",
                )
            ],
            auth={"x-api-key": key.get_secret_value()}
            if key and key.get_secret_value()
            else None,
        )

    if followup:
        await group(arxiv_followup_probes(saved))
    else:
        await asyncio.gather(
            group(arxiv_probes(saved)), alternatives(), semantic_once()
        )
    result = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "previous_experiment": str(saved_path),
        "credentials_present": {
            "semantic_scholar": bool(
                settings.semantic_scholar_api_key
                and settings.semantic_scholar_api_key.get_secret_value()
            ),
            "core": bool(os.getenv("CORE_API_KEY")),
            "openaire": bool(os.getenv("OPENAIRE_TOKEN")),
        },
        "retry_policy": "No retries; stop each group on first 429/401/403",
        "records": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        fields = [
            "id",
            "provider",
            "variant",
            "claim_id",
            "dated",
            "status",
            "result_count",
            "total_hits",
            "seconds",
            "relaxation",
        ]
        writer = csv.DictWriter(
            file, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> None:
    """Require deliberate live opt-in and retain every request/response."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-live", action="store_true")
    parser.add_argument("--arxiv-followup", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/provider-diagnostics-20261030"),
    )
    parser.add_argument(
        "--saved",
        type=Path,
        default=Path(
            "artifacts/query-evaluation-20261030/results.json"
        ),
    )
    args = parser.parse_args()
    if not args.run_live:
        parser.error(
            "--run-live is required; no network requests were made"
        )
    result = asyncio.run(
        run(args.output, args.saved, followup=args.arxiv_followup)
    )
    print(
        f"Saved {len(result['records'])} bounded probes to {args.output}"
    )


if __name__ == "__main__":
    main()
