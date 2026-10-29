"""Shared transport types; IDs and source locations must survive every stage."""
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

Verdict = Literal["Supported", "Contradicted", "Insufficient Evidence"]

@dataclass(frozen=True)
class Claim:
    id: str
    text: str

@dataclass(frozen=True)
class Paper:
    id: str
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str | None = None
    published_on: date | None = None
    doi: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    provider_ids: dict[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class RankedPaper:
    paper_id: str
    rank: int  # One-based; 1 is the highest rank.
    score: float
    method: str

@dataclass(frozen=True)
class EvidencePassage:
    id: str
    paper_id: str
    text: str
    source: Literal["pdf", "abstract", "scifact"]
    location: str  # E.g. PDF page/chunk, abstract sentence, or SciFact sentence IDs.

@dataclass(frozen=True)
class EvidencePrediction:
    evidence_id: str
    entailment: float
    contradiction: float
    neutral: float

@dataclass(frozen=True)
class VerificationResult:
    claim_id: str
    verdict: Verdict
    evidence: list[EvidencePassage]
    predictions: list[EvidencePrediction]

@dataclass(frozen=True)
class PipelineResult:
    claim: Claim
    papers: list[Paper]
    ranking: list[RankedPaper]
    verification: VerificationResult
    explanation: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
