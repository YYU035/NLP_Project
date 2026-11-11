"""Explicit human review records; no inferred or silently filled ratings."""
import csv
import io
import json
from typing import Annotated

from pydantic import Field, model_validator

from research_verifier.contracts.models import Model, Verdict, EvidencePassage, digest
from research_verifier.explanations.review_study import REVIEW_FIELDS, summarize_reviews

Count = Annotated[int, Field(ge=0, le=10000, strict=True)]


class ReviewPacket(Model):
    output_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    case_id: str
    claim: str
    authoritative_verdict: Verdict | None
    evidence: list[EvidencePassage]
    raw_output: str

    @property
    def fingerprint(self):
        return digest(self.model_dump(mode="json"))


class ReviewScore(Model):
    output_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    reviewer: str = Field(pattern=r"^[A-Za-z0-9_-]{1,40}$")
    factual_supported: Count
    factual_total: Count
    citation_supported: Count
    citation_total: Count
    citation_complete: Count
    citation_required: Count
    verdict_consistent: Annotated[int, Field(ge=0, le=1, strict=True)]
    scope_preserved: Annotated[int, Field(ge=0, le=1, strict=True)]
    correction_supported: Count
    correction_total: Count
    usefulness: Annotated[int, Field(ge=1, le=5, strict=True)]
    notes: str = Field(default="", max_length=20000)

    @model_validator(mode="after")
    def check_counts(self):
        for a, b in [(self.factual_supported, self.factual_total), (self.citation_supported, self.citation_total),
                     (self.citation_complete, self.citation_required), (self.correction_supported, self.correction_total)]:
            if a > b:
                raise ValueError("Supported/complete counts exceed the total")
        return self


def load_review_packets(raw: str | bytes) -> list[ReviewPacket]:
    if len(raw) > 10_000_000:
        raise ValueError("Review packets exceed 10 MB")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig")
    packets = [ReviewPacket.model_validate_json(line) for line in raw.splitlines() if line.strip()]
    if not packets or len(packets) > 1000 or len({p.output_id for p in packets}) != len(packets):
        raise ValueError("Missing, duplicate or excessive review packets")
    return packets


def safe_cell(value):
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def export_reviews(scores: list[ReviewScore]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
    writer.writeheader()
    for score in scores:
        writer.writerow({k: safe_cell(v) for k, v in score.model_dump().items()})
    return stream.getvalue()


def summarize_saved_reviews(scores: list[ReviewScore]) -> dict:
    return summarize_reviews([{k: str(v) for k, v in score.model_dump().items()} for score in scores])
