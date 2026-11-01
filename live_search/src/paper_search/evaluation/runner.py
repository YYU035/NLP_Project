"""Opt-in live experiments using existing clients and persistent pacing."""

import asyncio
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from xml.etree import ElementTree as ET

import httpx

from paper_search.clients.base import PaperSearchClient
from paper_search.clients.http import HTTPProvider
from paper_search.config import Settings
from paper_search.evaluation.models import (
    ClaimCase,
    ExperimentOptions,
    SearchObservation,
)
from paper_search.exceptions import APIClientError
from paper_search.metadata import ScientificPaper
from paper_search.pipeline import build_clients
from paper_search.query_processing import (
    DeterministicQueryGenerator,
)

SAFE_HEADERS = (
    "content-type",
    "retry-after",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
)


def metadata_profile(
    papers: list[ScientificPaper], options: ExperimentOptions
) -> dict[str, Any]:
    """Observe field availability and violations among known dates."""
    dates = [
        p.publication_date for p in papers if p.publication_date
    ]
    return {
        "records": len(papers),
        "present": {
            name: sum(bool(getattr(p, name)) for p in papers)
            for name in (
                "title",
                "authors",
                "doi",
                "abstract",
                "publication_date",
                "year",
                "landing_url",
                "pdf_urls",
            )
        },
        "min_publication_date": min(dates).isoformat()
        if dates
        else None,
        "max_publication_date": max(dates).isoformat()
        if dates
        else None,
        "date_filter_violations": sum(
            bool(
                (
                    options.from_publication_date
                    and day < options.from_publication_date
                )
                or (
                    options.to_publication_date
                    and day > options.to_publication_date
                )
            )
            for day in dates
        ),
    }


def _raw_count(
    response: httpx.Response, provider: str
) -> int | None:
    try:
        if provider == "arxiv":
            return len(
                ET.fromstring(response.content).findall(
                    "{http://www.w3.org/2005/Atom}entry"
                )
            )
        payload = response.json()
        if provider == "semantic_scholar" and payload == {
            "total": 0, "offset": 0
        }:
            return 0
        entries = payload.get(
            "results" if provider in {"openalex", "openaire"} else "data"
        )
        return len(entries) if isinstance(entries, list) else None
    except (ValueError, AttributeError, ET.ParseError):
        return None


class ResponseRecorder:
    """Archive responses and timings, with search-only endpoint enforcement."""

    def __init__(self, output: Path, provider: str) -> None:
        self.output = output
        self.provider = provider
        self.current: SearchObservation | None = None
        self.endpoint = ""
        self.auth: dict[str, str] = {}

    async def on_request(self, request: httpx.Request) -> None:
        """Record wire attempts without writing credential values."""
        if self.current is None:
            raise RuntimeError(
                "An experiment observation must precede HTTP I/O"
            )
        endpoint = httpx.URL(self.endpoint)
        if (
            request.url.scheme,
            request.url.host,
            request.url.path,
        ) != (
            endpoint.scheme,
            endpoint.host,
            endpoint.path,
        ):
            raise RuntimeError(
                "Only provider search endpoints are allowed"
            )
        auth_ok = all(
            request.headers.get(name) == value
            for name, value in self.auth.items()
        )
        if not self.auth:
            auth_ok = not any(
                name in request.headers
                for name in ("Authorization", "x-api-key")
            )
        if not auth_ok:
            raise RuntimeError(
                "Unexpected authentication header configuration"
            )
        attempt: dict[str, Any] = {
            "number": len(self.current.attempts) + 1,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "endpoint": self.endpoint,
            "params": {
                key: value
                for key, value in request.url.params.items()
                if key.casefold() not in {"api_key", "key", "token"}
            },
            "authentication_header_verified": auth_ok,
            "status": None,
        }
        request.extensions["evaluation_start"] = time.perf_counter()
        request.extensions["evaluation_attempt"] = attempt
        self.current.attempts.append(attempt)

    async def on_response(self, response: httpx.Response) -> None:
        """Save all response bodies, including errors, before parsing."""
        if self.current is None:
            raise RuntimeError("Missing active observation")
        await response.aread()
        attempt = response.request.extensions["evaluation_attempt"]
        body = response.content
        redacted = False
        for value in self.auth.values():
            secret = value.removeprefix("Bearer ").encode()
            if secret and secret in body:
                body = body.replace(secret, b"[REDACTED]")
                redacted = True
        relative = Path("responses") / (
            f"{self.current.id}-attempt-{attempt['number']}.body"
        )
        path = self.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        attempt.update(
            {
                "status": response.status_code,
                "response_seconds": time.perf_counter()
                - response.request.extensions["evaluation_start"],
                "headers": {
                    name: response.headers[name]
                    for name in SAFE_HEADERS
                    if name in response.headers
                },
                "response_file": relative.as_posix(),
                "sha256": hashlib.sha256(body).hexdigest(),
                "credential_redaction": redacted,
            }
        )
        if response.is_success:
            self.current.raw_result_count = _raw_count(
                response, self.provider
            )


async def _search(
    client: HTTPProvider,
    recorder: ResponseRecorder,
    case: ClaimCase,
    kind: str,
    query: str,
    options: ExperimentOptions,
) -> SearchObservation:
    observation = SearchObservation(
        id=f"{recorder.provider}-{case.id}-{kind}",
        provider=recorder.provider,
        claim_id=case.id,
        kind=kind,
        query=query,
        authentication="key_supplied"
        if recorder.auth
        else (
            "not_required"
            if recorder.provider == "arxiv"
            else "anonymous"
        ),
    )
    recorder.current = observation
    start = time.perf_counter()
    try:
        # Concrete adapters share this public search signature; HTTPProvider
        # only supplies their transport implementation.
        papers = await cast(PaperSearchClient, client).search(
            query,
            max_results=options.max_results,
            from_publication_date=options.from_publication_date,
            to_publication_date=options.to_publication_date,
        )
        observation.papers = papers
        observation.result_count = len(papers)
        observation.status = "success" if papers else "zero_results"
        observation.metadata = metadata_profile(papers, options)
        if observation.raw_result_count and not papers:
            observation.status = "parsing_failure"
            observation.result_count = None
            observation.error = (
                "All returned records were dropped in parsing"
            )
    except APIClientError as exc:
        statuses = [a["status"] for a in observation.attempts]
        terminal = statuses[-1] if statuses else None
        observation.status = (
            "authentication_failure"
            if terminal in (401, 403)
            else "rate_limited"
            if terminal == 429
            else "parsing_failure"
            if terminal is not None and 200 <= terminal < 300
            else "provider_failure"
        )
        observation.error = str(exc)
        observation.cause_type = (
            type(exc.__cause__).__name__ if exc.__cause__ else None
        )
    finally:
        observation.elapsed_seconds = time.perf_counter() - start
        with (recorder.output / "calls.jsonl").open(
            "a", encoding="utf-8"
        ) as file:
            file.write(observation.model_dump_json() + "\n")
    return observation


async def run_experiment(
    cases: list[ClaimCase],
    settings: Settings,
    options: ExperimentOptions,
    output: Path,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Share originals and balance extra calls across enabled providers."""
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError(
            "Use nonempty claims with unique identifiers"
        )
    enabled = [
        name
        for name in ("openalex", "arxiv", "semantic_scholar")
        if getattr(settings, f"{name}_enabled")
    ]
    if not enabled:
        raise ValueError("No providers enabled")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "calls.jsonl").exists() or (
        output / "run.json"
    ).exists():
        raise FileExistsError(
            "Use a new experiment directory; do not overwrite evidence"
        )
    generator = DeterministicQueryGenerator.from_settings(
        settings.model_copy(
            update={"query_mode": "original_plus_concepts"}
        )
    )
    control = DeterministicQueryGenerator.from_settings(
        settings.model_copy(update={"query_mode": "original_only"})
    )
    plans = {
        case.id: generator.generate_with_details(case.claim)
        for case in cases
    }
    manifest: dict[str, Any] = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "options": options.model_dump(mode="json"),
        "providers": enabled,
        "timeout_seconds": settings.timeout_seconds,
        "cases": [case.model_dump() for case in cases],
        "generation": {
            key: plan.model_dump() for key, plan in plans.items()
        },
        "control_queries": {
            case.id: control.generate(case.claim) for case in cases
        },
        "relevance_labels": None,
        "method": "shared original; fresh original-repeat for matched budget",
    }
    (output / "run.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    async def run_provider(
        provider: str,
    ) -> list[SearchObservation]:
        recorder = ResponseRecorder(output, provider)
        observations = []
        async with httpx.AsyncClient(
            transport=transport,
            event_hooks={
                "request": [recorder.on_request],
                "response": [recorder.on_response],
            },
        ) as http:
            client = cast(
                HTTPProvider,
                build_clients(http, settings)[provider],
            )
            recorder.endpoint = client.endpoint
            recorder.auth = client.auth_headers()
            for index, case in enumerate(cases):
                jobs = [("original", case.claim)]
                extras = [
                    ("alternative", query)
                    for query in plans[case.id].queries[1:]
                ]
                if extras and options.matched_budget:
                    extras.append(("original_repeat", case.claim))
                # Counterbalance order deterministically across claims.
                jobs.extend(
                    extras
                    if index % 2 == 0
                    else list(reversed(extras))
                )
                for kind, query in jobs:
                    result = await _search(
                        client, recorder, case, kind, query, options
                    )
                    observations.append(result)
                    print(
                        f"{result.id}: {result.status}; "
                        f"results={result.result_count}; "
                        f"attempts={len(result.attempts)}; "
                        f"seconds={result.elapsed_seconds:.2f}",
                        flush=True,
                    )
        return observations

    groups = await asyncio.gather(
        *(run_provider(name) for name in enabled)
    )
    manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
    manifest["calls"] = [
        call.model_dump(mode="json")
        for group in groups
        for call in group
    ]
    return manifest
