"""Proposed Person 2/3 contract, retaining Person 1's actual paper records."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from paper_search.retrieval_models import RetrievedPaper, RetrievalResult


class RankingInput(BaseModel):
    """Ordered unique paper IDs, most relevant first; scores are not required."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1)
    candidates: RetrievalResult
    bm25_ranking: list[str]
    sbert_ranking: list[str]
    bm25_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    sbert_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        ids = [entry.identity for entry in self.candidates.papers]
        if len(ids) != len(set(ids)):
            raise ValueError("Candidate identities must be unique")
        known = set(ids)
        for entry in self.candidates.papers:
            if entry.original_claim != self.candidates.original_claim:
                raise ValueError("Candidate claim must match original claim")
        for name in ("bm25_ranking", "sbert_ranking"):
            ranking = getattr(self, name)
            if len(ranking) != len(set(ranking)):
                raise ValueError(f"{name} must not contain duplicate IDs")
            if set(ranking) - known:
                raise ValueError(f"{name} contains unknown candidate IDs")
        return self


class RankedPaper(BaseModel):
    """A ranked entry keeps the complete upstream record and source references."""

    model_config = ConfigDict(extra="forbid")

    entry: RetrievedPaper
    rank: int = Field(ge=1)
    rrf_score: float = Field(ge=0, allow_inf_nan=False)
    bm25_rank: int | None = Field(default=None, ge=1)
    sbert_rank: int | None = Field(default=None, ge=1)
    cross_encoder_score: float | None = Field(default=None, allow_inf_nan=False)
