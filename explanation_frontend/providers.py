"""Demo test double, cached replay, and injected real pipeline integration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Protocol

from .models import AggregationResponse, RunSnapshot, make_context

FIXTURES = Path(__file__).parent / "fixtures"


class PipelineProvider(Protocol):
    def run(self, claim: str, options: dict) -> RunSnapshot: ...
    def aggregate(self, snapshot: RunSnapshot, paper_ids: list[str]) -> AggregationResponse: ...


def fixture_names() -> list[str]:
    return sorted(p.stem for p in FIXTURES.glob("*.json"))


def replay_aggregate(snapshot: RunSnapshot, paper_ids: list[str]) -> AggregationResponse:
    """Demo/replay policy only. Live mode always uses Person 4's aggregator."""
    selected = set(paper_ids)
    if not selected <= set(snapshot.papers):
        raise ValueError("Unknown selected paper")
    failures = snapshot.verification.processing_failures
    verdicts = [v for pid, v in snapshot.verification.per_paper_verdicts.items() if pid in selected]
    eligible = [p for p in snapshot.verification.evidence_passages if p.paper_id in selected
                and (label := snapshot.verification.per_passage_labels.get(p.passage_id))
                and not label.truncated and label.label != "NEUTRAL"]
    support = any(v.verdict == "SUPPORTED" or v.conflict_flag for v in verdicts)
    contra = any(v.verdict == "CONTRADICTED" or v.conflict_flag for v in verdicts)
    failed_ids = {f.paper_id for f in failures if f.paper_id}
    failed = bool(failures) and (not snapshot.papers or (bool(selected) and
                 (not verdicts or selected <= failed_ids)))
    verdict = None if failed else ("CONFLICTING" if support and contra else "SUPPORTED" if support
                                   else "CONTRADICTED" if contra else "INSUFFICIENT")
    cause = None
    if verdict == "INSUFFICIENT":
        cause = "EMPTY_SELECTION" if not selected and snapshot.papers else next(
            (v.sufficiency_cause for v in verdicts if v.sufficiency_cause), "NO_RETRIEVAL")
    excluded = sorted(set(snapshot.papers) - selected)
    return AggregationResponse(
        claim=snapshot.claim, aggregated_verdict=verdict, primary_evidence=eligible if not failed else [],
        global_confidence=max((v.confidence for v in verdicts if v.confidence is not None), default=None),
        status="failed" if failed else "partial" if failures or not snapshot.search_complete else "ok",
        sufficiency_cause=cause,
        verdict_delta_reason=f"Replay policy; excluded paper IDs: {', '.join(excluded) if excluded else 'none'}."
    )


class FixtureProvider:
    def run_fixture(self, name: str) -> RunSnapshot:
        if name not in fixture_names():
            raise ValueError("Unknown demo scenario")
        return RunSnapshot.model_validate_json((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))

    def run(self, claim: str, options: dict) -> RunSnapshot:
        snapshot = self.run_fixture(options.get("scenario", "conflict"))
        if claim != snapshot.claim:
            raise ValueError("Demo scenarios only verify their exact synthetic claim")
        return snapshot

    def aggregate(self, snapshot: RunSnapshot, paper_ids: list[str]) -> AggregationResponse:
        return replay_aggregate(snapshot, paper_ids)


class IntegratedProvider:
    """Inject teammates' actual callables; no guessed module imports or silent fixture fallback.

    search(claim, options)-> papers; rank(claim,papers)-> ranked papers;
    select(claim,ranked)-> passages_by_paper; verify(claim,ranked,passages)-> response;
    aggregate(response,IDs)-> aggregation; pack(claim,options,ranked,response,aggregation)-> snapshot.
    pack is the team's explicit field mapping into models.py, implemented in the integration factory.
    """
    def __init__(self, *, search: Callable, rank: Callable, select: Callable,
                 verify: Callable, aggregate: Callable, pack: Callable):
        self.search, self.rank, self.select = search, rank, select
        self.verify, self.aggregate_fn, self.pack = verify, aggregate, pack
        self._cached_response = None
        self._cached_fingerprint = None

    def run(self, claim: str, options: dict) -> RunSnapshot:
        if not claim.strip() or len(claim) > 20_000:
            raise ValueError("Provide a nonempty claim of at most 20000 characters")
        papers = self.rank(claim, self.search(claim, options))
        response = self.verify(claim, papers, self.select(claim, papers))
        ids = [p["paper_id"] if isinstance(p, dict) else p.paper_id for p in papers]
        aggregate = self.aggregate_fn(response, ids)
        snapshot = RunSnapshot.model_validate(self.pack(claim, options, papers, response, aggregate))
        if snapshot.claim != claim or snapshot.mode != "live":
            raise ValueError("Live adapter returned a mismatched claim or mode")
        self._cached_response, self._cached_fingerprint = response, snapshot.fingerprint
        return snapshot

    def aggregate(self, snapshot: RunSnapshot, paper_ids: list[str]) -> AggregationResponse:
        if not set(paper_ids) <= set(snapshot.papers):
            raise ValueError("Unknown selected paper")
        if snapshot.fingerprint != self._cached_fingerprint:
            raise ValueError("Original verification response is unavailable; load as cached replay or run the pipeline")
        result = AggregationResponse.model_validate(self.aggregate_fn(self._cached_response, sorted(set(paper_ids))))
        make_context(snapshot, paper_ids, result)
        return result
