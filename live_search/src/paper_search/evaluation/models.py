"""Serializable experiment inputs and observations without credentials."""

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from paper_search.metadata import ScientificPaper


class ClaimCase(BaseModel):
    """A fixed claim with an explicit domain and stable identifier."""

    id: str
    domain: str
    claim: str


class ExperimentOptions(BaseModel):
    """One set of filters/limits used by every query and comparison arm."""

    max_results: int = Field(default=5, ge=1, le=100)
    from_publication_date: date | None = None
    to_publication_date: date | None = None
    matched_budget: bool = True

    @model_validator(mode="after")
    def check_dates(self) -> "ExperimentOptions":
        """Reject reversed date windows before network requests."""
        start, end = (
            self.from_publication_date,
            self.to_publication_date,
        )
        if start and end and start > end:
            raise ValueError("Publication date window is reversed")
        return self


class SearchObservation(BaseModel):
    """One logical client call, including all its physical HTTP attempts."""

    id: str
    provider: str
    claim_id: str
    kind: Literal["original", "alternative", "original_repeat"]
    query: str
    status: str = "pending"
    result_count: int | None = None
    papers: list[ScientificPaper] = Field(default_factory=list)
    raw_result_count: int | None = None
    elapsed_seconds: float = 0
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None
    error: str | None = None
    cause_type: str | None = None
    authentication: str
    parser_correction: dict[str, Any] | None = None

    @property
    def succeeded(self) -> bool:
        """A valid empty response succeeds; provider failures never do."""
        return self.status in {"success", "zero_results"}
