"""Provenance-rich Person 1 interface; legacy SearchPipeline is unchanged."""

import asyncio
import hashlib
import itertools
import logging
import os
import time
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from datetime import date
from pathlib import Path

import httpx

from paper_search.arxiv_fallback import quote_phrases, relaxation_plan
from paper_search.clients.base import PaperSearchClient
from paper_search.clients.http import HTTPProvider
from paper_search.config import Settings
from paper_search.deduplication import arxiv_ids, group_papers, normalized_doi
from paper_search.exceptions import APIClientError
from paper_search.keybert_queries import (
    DEFAULT_MODEL_PATH,
    KeyBERTQueryGenerator,
)
from paper_search.metadata import ScientificPaper
from paper_search.pipeline import build_clients
from paper_search.query_processing import QueryGenerator, SearchRequest
from paper_search.query_rendering import render_query
from paper_search.retrieval_models import (
    HTTPAttempt,
    PaperOrigin,
    ProviderPlan,
    ProviderRun,
    QueryRun,
    RetrievalResult,
    RetrievedPaper,
    SourceRecord,
)

_TRACE: ContextVar[QueryRun | None] = ContextVar(
    "retrieval_query", default=None
)
logger = logging.getLogger(__name__)
Factory = Callable[
    [httpx.AsyncClient, Settings], Mapping[str, PaperSearchClient]
]
Planner = Callable[[str, str], ProviderPlan]


class RetrievalUnavailable(APIClientError):
    """All providers failed; diagnostics remain accessible on result."""

    def __init__(self, result: RetrievalResult) -> None:
        super().__init__("All enabled retrieval providers failed")
        self.result = result


def merge_records(
    claim: str, records: list[tuple[str, SourceRecord]]
) -> list[RetrievedPaper]:
    """Group conservatively, retaining metadata alternatives and origins."""
    grouped = group_papers([r.paper for _, r in records])
    output = []
    for index, papers in enumerate(grouped.groups):
        members = [
            (provider, record)
            for (provider, record), assigned in zip(
                records, grouped.assignments, strict=True
            )
            if assigned == index
        ]
        aliases = sorted(
            {f"{p.source}:{p.source_id}" for p in papers}
            | {f"doi:{d}" for p in papers if (d := normalized_doi(p.doi))}
            | {f"arxiv:{a}" for p in papers for a in arxiv_ids(p)}
        )
        identity = next(
            (
                a
                for prefix in ("doi:", "arxiv:")
                for a in aliases
                if a.startswith(prefix)
            ),
            aliases[0],
        )
        ordered = sorted(
            papers,
            key=lambda p: (
                -sum(
                    bool(getattr(p, field))
                    for field in (
                        "abstract",
                        "title",
                        "doi",
                        "publication_date",
                        "authors",
                        "pdf_urls",
                    )
                ),
                p.source,
                p.source_id,
            ),
        )
        data = ordered[0].model_dump()
        for p in ordered[1:]:
            for field in (
                "title",
                "abstract",
                "doi",
                "publication_date",
                "year",
                "authors",
                "landing_url",
            ):
                if not data[field] and getattr(p, field):
                    data[field] = getattr(p, field)
        pdfs = sorted(
            {str(link) for p in papers for link in p.pdf_urls}
            | {str(p.pdf_url) for p in papers if p.pdf_url}
        )
        data.update(pdf_urls=pdfs, pdf_url=pdfs[0] if pdfs else None)
        output.append(
            RetrievedPaper(
                identity=identity,
                identity_aliases=aliases,
                original_claim=claim,
                paper=ScientificPaper.model_validate(data),
                source_records=papers,
                provenance=[
                    PaperOrigin(
                        provider=provider,
                        source_id=r.paper.source_id,
                        query_id=r.query_id,
                        rank=r.rank,
                    )
                    for provider, r in members
                ],
            )
        )
    return output


class RetrievalService:
    """Current policy: KeyBERT OpenAlex/OpenAIRE, spaCy arXiv.

    Prefer a persistent async context to retain provider pacing/hourly budgets.
    top_k is per-provider depth, not a fabricated globally ranked limit.
    Optional arXiv relaxation is disabled unless explicitly requested.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        generator: QueryGenerator | None = None,
        planner: Planner | None = None,
        factory: Factory = build_clients,
        transport: httpx.AsyncBaseTransport | None = None,
        providers: tuple[str, ...] = ("openalex", "openaire", "arxiv"),
        arxiv_fallback: bool = False,
        max_fallbacks: int = 2,
        archive: Path | None = None,
    ) -> None:
        if len(set(providers)) != len(providers) or not set(providers) <= {
            "openalex",
            "openaire",
            "arxiv",
        }:
            raise ValueError("Choose unique supported providers")
        if type(max_fallbacks) is not int or not 0 <= max_fallbacks <= 2:
            raise ValueError("max_fallbacks must be zero through two")
        if settings is None:
            settings = Settings.from_environment()
            if "PAPER_SEARCH_OPENAIRE_ENABLED" not in os.environ:
                settings = settings.model_copy(
                    update={"openaire_enabled": True}
                )
        self.settings = settings
        self.providers = tuple(
            p for p in providers if getattr(self.settings, f"{p}_enabled")
        )
        if not self.providers:
            raise ValueError("At least one provider must be enabled")
        self.generator = generator or KeyBERTQueryGenerator(DEFAULT_MODEL_PATH)
        self.planner = planner
        self.factory = factory
        self.transport = transport
        self.arxiv_fallback = arxiv_fallback
        self.max_fallbacks = max_fallbacks
        self.archive = archive
        self.http: httpx.AsyncClient | None = None
        self.clients: Mapping[str, PaperSearchClient] = {}
        self._counter = itertools.count(1)
        self._query_cache: dict[str, str] = {}

    async def __aenter__(self) -> "RetrievalService":
        if self.http is None:
            self.http = httpx.AsyncClient(
                transport=self.transport,
                event_hooks={
                    "request": [self._request],
                    "response": [self._response],
                },
            )
            self.clients = self.factory(
                self.http,
                self.settings.model_copy(
                    update={"semantic_scholar_enabled": False}
                ),
            )
            for name in self.providers:
                if name not in self.clients:
                    await self.aclose()
                    raise ValueError("Factory omitted an enabled provider")
            # New service-owned adapters; legacy shared clients are untouched.
            if "arxiv" in self.providers:
                self.clients["arxiv"].query_strategy = "literal"
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the service HTTP client and its transport."""
        if self.http is not None:
            await self.http.aclose()
            self.http = None
            self.clients = {}

    async def _request(self, request: httpx.Request) -> None:
        trace = _TRACE.get()
        if trace is None:
            raise RuntimeError("HTTP request has no retrieval trace")
        attempt = HTTPAttempt(
            endpoint=str(request.url.copy_with(query=None)),
            parameters={
                k: v
                for k, v in request.url.params.items()
                if k.casefold() not in {"api_key", "key", "token"}
            },
        )
        trace.attempts.append(attempt)
        request.extensions["retrieval_attempt"] = attempt
        request.extensions["retrieval_started"] = time.perf_counter()

    async def _response(self, response: httpx.Response) -> None:
        trace = _TRACE.get()
        attempt = response.request.extensions["retrieval_attempt"]
        attempt.status = response.status_code
        attempt.seconds = (
            time.perf_counter()
            - response.request.extensions["retrieval_started"]
        )
        if self.archive and trace is not None:
            await response.aread()
            content = response.content
            client = self.clients[trace.provider]
            if isinstance(client, HTTPProvider):
                for secret in client.auth_headers().values():
                    secret = secret.removeprefix("Bearer ")
                    if secret:
                        content = content.replace(
                            secret.encode(), b"[REDACTED]"
                        )
            relative = (
                Path("responses") / f"{trace.id}-{len(trace.attempts)}.body"
            )
            path = self.archive / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(content)
            attempt.response_file = relative.as_posix()
            attempt.sha256 = hashlib.sha256(content).hexdigest()

    def _plan(self, provider: str, claim: str) -> ProviderPlan:
        if self.planner:
            result = self.planner(provider, claim)
            if result.provider != provider:
                raise ValueError("Query plan belongs to another provider")
            return result
        if provider == "arxiv":
            rendered = render_query(claim, "arxiv")
            if not rendered.selected_concepts:
                raise ValueError(
                    "spaCy concepts unavailable; cannot plan a safe query"
                )
            return ProviderPlan(
                provider=provider,
                strategy="spacy",
                query=quote_phrases(rendered.selected_concepts),
                required_phrases=rendered.selected_concepts,
                notes=[
                    *rendered.warnings,
                    *[
                        f"generation_omitted:{span}"
                        for span in rendered.omitted_spans
                    ],
                ],
            )
        if claim not in self._query_cache:
            queries = self.generator.generate(claim, max_queries=2)
            if len(queries) < 2 or queries[0] != claim:
                raise ValueError(
                    "KeyBERT produced no alternative; no strategy switch"
                )
            if len(self._query_cache) >= 128:
                self._query_cache.pop(next(iter(self._query_cache)))
            self._query_cache[claim] = queries[1]
        return ProviderPlan(
            provider=provider,
            strategy="keybert",
            query=self._query_cache[claim],
        )

    async def retrieve(
        self,
        claim: str,
        top_k: int = 30,
        *,
        from_publication_date: date | None = None,
        to_publication_date: date | None = None,
    ) -> RetrievalResult:
        """Return an unranked union plus full provider/query diagnostics.

        All provider failure raises RetrievalUnavailable(result); valid empty
        responses return an empty result. Partial provider/fallback failures
        retain all successfully parsed candidates and are explicit in status.
        """
        if self.http is None:
            raise RuntimeError(
                "Use async with RetrievalService() before retrieve"
            )
        SearchRequest(
            query=claim,
            max_results=top_k,
            from_publication_date=from_publication_date,
            to_publication_date=to_publication_date,
        )
        started = time.perf_counter()

        async def provider(name: str) -> ProviderRun:
            outcome = ProviderRun(provider=name)
            begin = time.perf_counter()
            relaxations = []
            try:
                plan = self._plan(name, claim)
                if name == "arxiv" and self.arxiv_fallback:
                    expected = quote_phrases(
                        plan.required_phrases + plan.optional_phrases
                    )
                    if render_query(
                        plan.query, "arxiv", arxiv_strategy="literal"
                    ).text != (
                        render_query(
                            expected, "arxiv", arxiv_strategy="literal"
                        ).text
                    ):
                        raise ValueError(
                            "Fallback annotations must match the primary query"
                        )
                    relaxations = relaxation_plan(
                        plan.required_phrases,
                        plan.optional_phrases,
                        max_attempts=self.max_fallbacks,
                    )
            except (ValueError, OSError, ImportError, RuntimeError) as exc:
                outcome.status = "query_generation_failure"
                outcome.error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "%s query planning failed: %s", name, type(exc).__name__
                )
                outcome.seconds = time.perf_counter() - begin
                return outcome
            jobs = [(plan.query, "primary", plan.notes)]
            if name == "arxiv" and self.arxiv_fallback:
                jobs.extend(
                    (
                        r.query,
                        f"fallback_{i}",
                        [
                            r.reason,
                            *[f"omitted:{p}" for p in r.omitted_phrases],
                        ],
                    )
                    for i, r in enumerate(relaxations, 1)
                )
            for query, stage, notes in jobs:
                if stage != "primary":
                    unique = group_papers([r.paper for r in outcome.records])
                    if len(unique.groups) >= min(5, top_k):
                        break
                trace = QueryRun(
                    id=f"{name}-{next(self._counter):06d}",
                    provider=name,
                    strategy=plan.strategy,
                    stage=stage,
                    query=query,
                    notes=notes,
                )
                outcome.queries.append(trace)
                token = _TRACE.set(trace)
                query_started = time.perf_counter()
                try:
                    papers = await self.clients[name].search(
                        query,
                        max_results=top_k,
                        from_publication_date=from_publication_date,
                        to_publication_date=to_publication_date,
                    )
                    trace.returned = len(papers)
                    trace.status = "success" if papers else "zero_results"
                    outcome.records.extend(
                        SourceRecord(paper=p, query_id=trace.id, rank=i)
                        for i, p in enumerate(papers, 1)
                    )
                except APIClientError as exc:
                    trace.status = "provider_failure"
                    trace.error = str(exc)
                    outcome.error = str(exc)
                    logger.warning("%s retrieval failed: %s", name, exc)
                    outcome.status = (
                        "partial_failure"
                        if len(outcome.queries) > 1
                        else "provider_failure"
                    )
                    break
                finally:
                    trace.seconds = time.perf_counter() - query_started
                    _TRACE.reset(token)
            if outcome.status == "pending":
                outcome.status = (
                    "success" if outcome.records else "zero_results"
                )
            outcome.seconds = time.perf_counter() - begin
            return outcome

        outcomes = await asyncio.gather(
            *(provider(name) for name in self.providers)
        )
        retained = []
        for outcome in outcomes:
            groups = group_papers([r.paper for r in outcome.records])
            keep: set[int] = set()
            for i in groups.assignments:
                if len(keep) < top_k or i in keep:
                    keep.add(i)
            retained.extend(
                (outcome.provider, record)
                for record, i in zip(
                    outcome.records, groups.assignments, strict=True
                )
                if i in keep
            )
        result = RetrievalResult(
            original_claim=claim,
            top_k=top_k,
            from_publication_date=from_publication_date,
            to_publication_date=to_publication_date,
            papers=merge_records(claim, retained),
            providers={o.provider: o for o in outcomes},
            seconds=time.perf_counter() - started,
            http_attempts=sum(o.http_attempts for o in outcomes),
        )
        if all(
            o.status in {"provider_failure", "query_generation_failure"}
            for o in outcomes
        ):
            raise RetrievalUnavailable(result)
        return result


async def retrieve(
    claim: str,
    top_k: int = 30,
    *,
    from_publication_date: date | None = None,
    to_publication_date: date | None = None,
) -> RetrievalResult:
    """One-shot convenience; use a persistent service for repeated claims."""
    async with RetrievalService() as service:
        return await service.retrieve(
            claim,
            top_k,
            from_publication_date=from_publication_date,
            to_publication_date=to_publication_date,
        )
