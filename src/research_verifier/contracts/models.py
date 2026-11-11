"""Versioned local handoff types. No upstream implementation is assumed."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Verdict = Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT", "CONFLICTING"]
Confidence = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Paper(Model):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    url: str | None = None
    source_version: str | None = None
    source_status: str = "unknown"
    is_open_access: bool | None = None
    has_fulltext: bool = False
    retrieval_scores: dict[str, Annotated[float, Field(allow_inf_nan=False)]] = Field(default_factory=dict)


class EvidencePassage(Model):
    passage_id: str
    paper_id: str
    section: str
    text: str = Field(min_length=1, max_length=100_000)
    score: float | None = Field(default=None, allow_inf_nan=False)
    page: int | None = Field(default=None, ge=1)
    sentence_indices: list[int] = Field(default_factory=list)
    source_version: str | None = None


class PassageVerdict(Model):
    passage_id: str
    paper_id: str
    label: Literal["SUPPORT", "CONTRADICT", "NEUTRAL"]
    confidence: Confidence | None = None
    truncated: bool = False


class PaperVerdict(Model):
    paper_id: str
    title: str
    verdict: Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT"]
    confidence: Confidence | None = None
    evidence_type: Literal["abstract_only", "full_text"]
    sufficiency_cause: str | None = None
    conflict_flag: bool = False
    supporting_passage_ids: list[str] = Field(default_factory=list)
    contradicting_passage_ids: list[str] = Field(default_factory=list)


class ProcessingFailure(Model):
    stage: str
    code: str
    paper_id: str | None = None
    # Intentionally no arbitrary exception text or credentials in this public schema.


class ModelSettings(Model):
    checkpoint: str
    revision: str
    max_length: int = Field(gt=0)
    threshold: Confidence | None = None
    seed: int | None = None


class VerificationResponse(Model):
    per_paper_verdicts: dict[str, PaperVerdict]
    per_passage_labels: dict[str, PassageVerdict]
    processing_failures: list[ProcessingFailure] = Field(default_factory=list)
    evidence_passages: list[EvidencePassage]
    model_settings: ModelSettings


class AggregationResponse(Model):
    claim: str
    aggregated_verdict: Verdict | None
    primary_evidence: list[EvidencePassage] = Field(default_factory=list)
    global_confidence: Confidence | None = None
    verdict_delta_reason: str = ""
    status: Literal["ok", "partial", "failed"] = "ok"
    sufficiency_cause: str | None = None

    @model_validator(mode="after")
    def check_status(self):
        if self.status == "failed" and self.aggregated_verdict is not None:
            raise ValueError("Failed processing is not a scientific verdict")
        if self.status != "failed" and self.aggregated_verdict is None:
            raise ValueError("A successful aggregation requires a verdict")
        if self.aggregated_verdict == "INSUFFICIENT" and not self.sufficiency_cause:
            raise ValueError("Insufficiency requires a cause")
        return self


class RunSnapshot(Model):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    claim: str = Field(min_length=1, max_length=20_000)
    created_at: datetime
    retrieved_at: datetime
    mode: Literal["demo", "live", "cached"]
    source_snapshot_id: str
    filters: dict[str, str] = Field(default_factory=dict)
    search_complete: bool
    papers: dict[str, Paper]
    verification: VerificationResponse
    provisional: AggregationResponse
    stage_statuses: dict[str, str] = Field(default_factory=dict)
    explicit_replacements: dict[str, str] = Field(default_factory=dict)
    discovery_context: dict | None = None

    @model_validator(mode="after")
    def check_links(self):
        if not self.claim.strip():
            raise ValueError("Claim is empty")
        if self.created_at.tzinfo is None or self.retrieved_at.tzinfo is None:
            raise ValueError("Timestamps must include timezone")
        for key, paper in self.papers.items():
            if not key or key != paper.paper_id:
                raise ValueError("Paper key mismatch")
        passages = {p.passage_id: p for p in self.verification.evidence_passages}
        if len(passages) != len(self.verification.evidence_passages):
            raise ValueError("Duplicate passage IDs")
        for p in passages.values():
            if not p.passage_id or p.paper_id not in self.papers:
                raise ValueError("Passage references unknown paper")
        for key, label in self.verification.per_passage_labels.items():
            if key != label.passage_id or key not in passages or passages[key].paper_id != label.paper_id:
                raise ValueError("Passage label reference mismatch")
        for key, verdict in self.verification.per_paper_verdicts.items():
            if key != verdict.paper_id or key not in self.papers:
                raise ValueError("Paper verdict reference mismatch")
            for ids, expected in [(verdict.supporting_passage_ids, "SUPPORT"),
                                  (verdict.contradicting_passage_ids, "CONTRADICT")]:
                for pid in ids:
                    label = self.verification.per_passage_labels.get(pid)
                    if pid not in passages or passages[pid].paper_id != key or not label or label.label != expected:
                        raise ValueError("Verdict evidence reference mismatch")
        validate_aggregation(self, self.provisional, set(self.papers))
        for pid, text in self.explicit_replacements.items():
            if pid not in passages or passages[pid].text != text:
                raise ValueError("Replacement must be an exact supplied passage")
        if self.discovery_context is not None:
            from research_verifier.discovery.service import DiscoverySnapshot
            discovery = DiscoverySnapshot.model_validate(self.discovery_context)
            if discovery.result.original_claim != self.claim or not set(self.papers) <= set(discovery.selected_paper_ids):
                raise ValueError("Verification is outside its original discovery claim/selection")
        return self

    @property
    def fingerprint(self) -> str:
        data = self.model_dump(mode="json")
        if self.discovery_context is None:
            data.pop("discovery_context")  # preserve original 1.0 replay hashes
        return digest(data)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def validate_aggregation(snapshot: RunSnapshot, aggregation: AggregationResponse, selected: set[str]):
    if aggregation.claim != snapshot.claim:
        raise ValueError("Aggregation claim mismatch")
    passages = {p.passage_id: p for p in snapshot.verification.evidence_passages}
    seen = set()
    for p in aggregation.primary_evidence:
        if p.passage_id in seen or p.paper_id not in selected or passages.get(p.passage_id) != p:
            raise ValueError("Aggregation evidence is duplicated, changed or outside selection")
        label = snapshot.verification.per_passage_labels.get(p.passage_id)
        if not label or label.truncated or label.label == "NEUTRAL":
            raise ValueError("Primary evidence must be eligible labelled evidence")
        seen.add(p.passage_id)
    labels = {snapshot.verification.per_passage_labels[pid].label for pid in seen}
    required = {"SUPPORTED": {"SUPPORT"}, "CONTRADICTED": {"CONTRADICT"}, "CONFLICTING": {"SUPPORT", "CONTRADICT"}}
    if not required.get(aggregation.aggregated_verdict, set()) <= labels:
        raise ValueError("Decisive verdict lacks corresponding primary evidence")


class ExplanationContext(Model):
    snapshot: RunSnapshot
    selected_paper_ids: list[str]
    aggregation: AggregationResponse

    @property
    def passages(self) -> dict[str, EvidencePassage]:
        return {p.passage_id: p for p in self.snapshot.verification.evidence_passages
                if p.paper_id in self.selected_paper_ids}

    @property
    def key(self) -> str:
        return digest({"snapshot": self.snapshot.fingerprint, "selection": self.selected_paper_ids,
                       "aggregation": self.aggregation.model_dump(mode="json")})


def make_context(snapshot: RunSnapshot, paper_ids: list[str], aggregation: AggregationResponse) -> ExplanationContext:
    if not set(paper_ids) <= set(snapshot.papers):
        raise ValueError("Unknown selected paper")
    validate_aggregation(snapshot, aggregation, set(paper_ids))
    return ExplanationContext(snapshot=snapshot, selected_paper_ids=sorted(set(paper_ids)), aggregation=aggregation)


class Statement(Model):
    text: str = Field(min_length=1, max_length=100_000)
    citation_ids: list[str] = Field(min_length=1, max_length=20)


class Explanation(Model):
    context_key: str
    verdict: Verdict | None
    statements: list[Statement] = Field(default_factory=list)
    correction: Statement | None = None
    method: Literal["template", "llm"] = "template"
    model: str | None = None
    prompt_version: str | None = None
    limitations: list[str] = Field(default_factory=list)


class ValidationReport(Model):
    references_valid: bool
    extractive_support: bool
    verdict_consistent: bool
    displayable: bool
    issues: list[str]
    semantic_support: str = "Not independently assessed; exact evidence quotations only"
