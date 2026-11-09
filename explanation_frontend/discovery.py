"""Person 1 handoff and session-owned asynchronous discovery; no scientific verdicts."""
from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Annotated, Callable, Literal
import weakref

from pydantic import Field, model_validator

from .models import Model, digest

UPSTREAM_REVISION = "259d2a9e90fb8e9d5988f030a156733f4f3576d1"
PositiveSeconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class ScientificPaper(Model):
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    year: int | None = Field(default=None, ge=1)
    publication_date: date | None = None
    doi: str | None = None
    source: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    landing_url: str | None = None
    pdf_url: str | None = None
    pdf_urls: list[str] = Field(default_factory=list)


class HTTPAttempt(Model):
    endpoint: str = ""
    parameters: dict[str, str] = Field(default_factory=dict)
    status: int | None = None
    seconds: PositiveSeconds = 0
    response_file: str | None = None
    sha256: str | None = None


class QueryRun(Model):
    id: str
    provider: str
    strategy: str
    stage: str
    query: str
    notes: list[str] = Field(default_factory=list)
    status: str = "pending"
    returned: int | None = Field(default=None, ge=0)
    seconds: PositiveSeconds = 0
    error: str | None = None
    attempts: list[HTTPAttempt] = Field(default_factory=list)


class SourceRecord(Model):
    paper: ScientificPaper
    query_id: str
    rank: int = Field(ge=1)


class ProviderRun(Model):
    provider: str
    status: Literal["success", "zero_results", "provider_failure", "query_generation_failure", "partial_failure"]
    seconds: PositiveSeconds = 0
    error: str | None = None
    queries: list[QueryRun] = Field(default_factory=list)
    records: list[SourceRecord] = Field(default_factory=list)


class PaperOrigin(Model):
    provider: str
    source_id: str
    query_id: str
    rank: int = Field(ge=1)


class RetrievedPaper(Model):
    identity: str = Field(min_length=1)
    identity_aliases: list[str]
    original_claim: str
    paper: ScientificPaper
    provenance: list[PaperOrigin]
    source_records: list[ScientificPaper]


class DiscoveryResult(Model):
    original_claim: str = Field(min_length=1, max_length=20_000)
    top_k: int = Field(ge=1, le=100)
    limit_scope: Literal["per_provider"] = "per_provider"
    from_publication_date: date | None = None
    to_publication_date: date | None = None
    papers: list[RetrievedPaper] = Field(default_factory=list, max_length=1000)
    providers: dict[str, ProviderRun] = Field(min_length=1)
    seconds: PositiveSeconds = 0
    http_attempts: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def check_contract(self):
        if not self.original_claim.strip():
            raise ValueError("An exact nonempty claim is required")
        if self.from_publication_date and self.to_publication_date and self.from_publication_date > self.to_publication_date:
            raise ValueError("Dates are reversed")
        if len({p.identity for p in self.papers}) != len(self.papers):
            raise ValueError("Duplicate canonical identities")
        for name, provider in self.providers.items():
            if provider.provider != name:
                raise ValueError("Provider key mismatch")
            if any(q.provider != name for q in provider.queries):
                raise ValueError("Query provider mismatch")
        for paper in self.papers:
            if paper.original_claim != self.original_claim or paper.identity not in paper.identity_aliases:
                raise ValueError("Candidate claim/alias mismatch")
            if not paper.provenance or not paper.source_records:
                raise ValueError("Candidate provenance is missing")
            for origin in paper.provenance:
                if origin.provider not in self.providers:
                    raise ValueError("Origin references unknown provider")
                if origin.query_id not in {q.id for q in self.providers[origin.provider].queries}:
                    raise ValueError("Origin references unknown query")
        return self

    @property
    def status(self) -> str:
        statuses = {provider.status for provider in self.providers.values()}
        failures = {"provider_failure", "query_generation_failure"}
        if statuses <= failures:
            return "failed"
        if statuses & (failures | {"partial_failure"}):
            return "partial"
        return "ok" if self.papers else "empty"


class DiscoverySnapshot(Model):
    schema_version: Literal["discovery-1.0"] = "discovery-1.0"
    mode: Literal["cached", "live"]
    imported_at: datetime
    retrieved_at: datetime | None = None
    source_reference: str
    result: DiscoveryResult
    selected_paper_ids: list[str] = Field(default_factory=list)
    ranking_context: dict | None = None

    @model_validator(mode="after")
    def check_selection(self):
        if self.ranking_context is not None:
            from .ranking import connect_ranking
            connect_ranking(json.dumps(self.ranking_context), self.result)
            ranked = {r["entry"]["identity"] for r in self.ranking_context["papers"]}
            if not set(self.selected_paper_ids) <= ranked:
                raise ValueError("Selection references a paper outside ranking output")
        if not set(self.selected_paper_ids) <= {p.identity for p in self.result.papers}:
            raise ValueError("Selection references unknown candidate")
        for value in [self.imported_at, self.retrieved_at]:
            if value and value.tzinfo is None:
                raise ValueError("Timestamps require timezone")
        return self

    @property
    def fingerprint(self):
        data = self.model_dump(mode="json")
        if self.ranking_context is None:
            data.pop("ranking_context", None)
        return digest(data)


def import_discovery(raw: str | bytes, *, source_reference: str = "Uploaded native RetrievalResult") -> DiscoverySnapshot:
    if len(raw) > 10_000_000:
        raise ValueError("Discovery file exceeds 10 MB")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Expected a discovery object")
    if data.get("export_version") == "discovery-export-1.0":
        if set(data) != {"export_version", "snapshot", "sha256"}:
            raise ValueError("Unexpected export fields")
        if digest(data["snapshot"]) != data["sha256"]:
            raise ValueError("Discovery checksum mismatch")
        snapshot = DiscoverySnapshot.model_validate(data["snapshot"])
        return snapshot.model_copy(update={"mode": "cached"})
    result = DiscoveryResult.model_validate(data)
    return DiscoverySnapshot(mode="cached", imported_at=datetime.now(timezone.utc), source_reference=source_reference,
                             result=result, selected_paper_ids=[p.identity for p in result.papers])


def export_discovery(snapshot: DiscoverySnapshot, paper_ids: list[str]) -> str:
    data = snapshot.model_dump(mode="json")
    data["selected_paper_ids"] = sorted(set(paper_ids))
    checked = DiscoverySnapshot.model_validate(data).model_dump(mode="json")
    return json.dumps({"export_version": "discovery-export-1.0", "snapshot": checked, "sha256": digest(checked)},
                      ensure_ascii=False, indent=2)


def discovery_handoff(snapshot: DiscoverySnapshot, paper_ids: list[str]) -> dict:
    from .ranking import ranked_candidates
    if not set(paper_ids) <= {p.identity for p in snapshot.result.papers}:
        raise ValueError("Unknown selected candidate")
    if snapshot.ranking_context is not None and not set(paper_ids) <= {
        row["entry"]["identity"] for row in snapshot.ranking_context["papers"]
    }:
        raise ValueError("Selection references a paper outside ranking output")
    return {"handoff_version": "discovery-selection-1.0", "original_claim": snapshot.result.original_claim,
            "discovery_hash": snapshot.fingerprint, "verification_status": "not_yet_verified",
            "source_reference": snapshot.source_reference,
            "ranking_context": snapshot.ranking_context,
            "papers": [p.model_dump(mode="json") for p in ranked_candidates(snapshot.result.papers, snapshot.ranking_context) if p.identity in paper_ids],
            "providers": {k: v.model_dump(mode="json") for k, v in snapshot.result.providers.items()},
            "from_publication_date": str(snapshot.result.from_publication_date) if snapshot.result.from_publication_date else None,
            "to_publication_date": str(snapshot.result.to_publication_date) if snapshot.result.to_publication_date else None}


def sample_discovery() -> DiscoverySnapshot:
    path = Path(__file__).parent / "fixtures" / "discovery" / "raghavi-c01.json"
    return import_discovery(path.read_bytes(), source_reference=f"YYU035/NLP_Project@{UPSTREAM_REVISION}:live_search/artifacts/depth-approved-run/c01.json")


def filter_candidates(result: DiscoveryResult, *, text="", providers=None, pdf_only=False) -> list[RetrievedPaper]:
    term = text.casefold().strip()
    return [p for p in result.papers
            if (not term or term in " ".join([p.paper.title or "", p.paper.abstract or "", p.identity, *p.paper.authors]).casefold())
            and (not providers or any(origin.provider in providers for origin in p.provenance))
            and (not pdf_only or p.paper.pdf_url or p.paper.pdf_urls)]


def _service_factory():
    # Dependencies and model downloads are explicit setup steps. No guessing alternative queries/models.
    cache = Path(__file__).parent / ".model-cache"
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from paper_search.retrieval import RetrievalService
    from paper_search.keybert_queries import KeyBERTQueryGenerator
    from paper_search.config import Settings
    model_path = Path(os.environ.get("EXPLANATION_DISCOVERY_MODEL_PATH", str(Path(__file__).parent / ".models" / "all-MiniLM-L6-v2")))
    if not model_path.is_dir():
        raise RuntimeError("Discovery model is not installed; run setup-discovery.ps1")
    settings = Settings.from_environment()
    settings = settings.model_copy(update={
        "openalex_api_key": None, "openaire_token": None, "semantic_scholar_api_key": None,
        "semantic_scholar_enabled": False,
        "openaire_enabled": settings.openaire_enabled if "PAPER_SEARCH_OPENAIRE_ENABLED" in os.environ else True,
    })
    return RetrievalService(settings=settings, generator=KeyBERTQueryGenerator(model_path), arxiv_fallback=False,
                            providers=("openalex", "openaire", "arxiv"))


def _release(loop, resources, thread):
    async def cleanup():
        if resources.get("service") is not None:
            await resources["service"].__aexit__(None, None, None)
            resources["service"] = None
    if loop.is_running():
        try:
            asyncio.run_coroutine_threadsafe(cleanup(), loop).result(timeout=5)
        except Exception:
            pass
        loop.call_soon_threadsafe(loop.stop)
        if threading.current_thread() != thread:
            thread.join(timeout=1)
        if not thread.is_alive():
            loop.close()


class LiveDiscoveryBridge:
    """One persistent event loop + service per session, preserving provider pacing.

    All query work occurs on this loop. UI worker threads only await its future.
    close() explicitly releases HTTP resources; finalization also attempts cleanup.
    """
    def __init__(self, *, service_factory: Callable = _service_factory):
        self._factory = service_factory
        self._resources = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._finalizer = weakref.finalize(self, _release, self._loop, self._resources, self._thread)
        self._closed = False

    async def _retrieve(self, claim, options):
        if self._resources.get("service") is None:
            service = self._factory()
            self._resources["service"] = await service.__aenter__()
        try:
            result = await self._resources["service"].retrieve(claim, **options)
        except Exception as error:
            if getattr(error, "result", None) is None:
                raise RuntimeError("Live discovery unavailable; check the local setup and provider configuration") from None
            result = error.result  # RetrievalUnavailable retains inspectable diagnostics.
        parsed = DiscoveryResult.model_validate(result.model_dump(mode="json"))
        if parsed.original_claim != claim:
            raise ValueError("Live result is attached to a different claim")
        now = datetime.now(timezone.utc)
        return DiscoverySnapshot(mode="live", imported_at=now, retrieved_at=now,
                                 source_reference=f"paper_search RetrievalService; contract {UPSTREAM_REVISION}", result=parsed,
                                 selected_paper_ids=[p.identity for p in parsed.papers])

    def retrieve(self, claim: str, *, top_k=30, from_publication_date=None, to_publication_date=None, timeout=115):
        if self._closed:
            raise RuntimeError("Discovery service was closed")
        if not claim.strip() or len(claim) > 20_000 or type(top_k) is not int or not 1 <= top_k <= 100:
            raise ValueError("Invalid discovery claim or provider limit")
        if from_publication_date and to_publication_date and from_publication_date > to_publication_date:
            raise ValueError("Dates are reversed")
        future = asyncio.run_coroutine_threadsafe(self._retrieve(claim, dict(top_k=top_k,
            from_publication_date=from_publication_date, to_publication_date=to_publication_date)), self._loop)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            future.cancel()
            raise TimeoutError("Discovery request timed out") from None

    def close(self):
        if not self._closed:
            self._closed = True
            self._finalizer()
