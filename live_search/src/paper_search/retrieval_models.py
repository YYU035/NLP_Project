"""Person 2 handoff schemas; original metadata and all origins are retained."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from paper_search.metadata import ScientificPaper


class ProviderPlan(BaseModel):
    """Provider-specific input, with explicit arXiv phrase requirements."""

    provider: str
    strategy: str
    query: str
    required_phrases: list[str] = Field(default_factory=list)
    optional_phrases: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class HTTPAttempt(BaseModel):
    """Safe actual HTTP cost; credentials are never exported."""

    endpoint: str = ""
    parameters: dict[str, str] = Field(default_factory=dict)
    status: int | None = None
    seconds: float = 0
    response_file: str | None = None
    sha256: str | None = None


class QueryRun(BaseModel):
    """One logical query, possibly containing shared-transport retries."""

    id: str
    provider: str
    strategy: str
    stage: str
    query: str
    notes: list[str] = Field(default_factory=list)
    status: str = "pending"
    returned: int | None = None
    seconds: float = 0
    error: str | None = None
    attempts: list[HTTPAttempt] = Field(default_factory=list)


class SourceRecord(BaseModel):
    """Source record and position in the normalized provider output."""

    paper: ScientificPaper
    query_id: str
    rank: int


class ProviderRun(BaseModel):
    """Provider outcome, distinguishing valid zeros and partial failures."""

    provider: str
    status: str = "pending"
    seconds: float = 0
    error: str | None = None
    queries: list[QueryRun] = Field(default_factory=list)
    records: list[SourceRecord] = Field(default_factory=list)

    @property
    def http_attempts(self) -> int:
        """Count actual attempts, not inferred query budgets."""
        return sum(len(query.attempts) for query in self.queries)


class PaperOrigin(BaseModel):
    """Every retained provider/query occurrence of a deduplicated paper."""

    provider: str
    source_id: str
    query_id: str
    rank: int


class RetrievedPaper(BaseModel):
    """A merged view plus lossless provider-specific normalized records."""

    identity: str
    identity_aliases: list[str]
    original_claim: str
    paper: ScientificPaper
    provenance: list[PaperOrigin]
    source_records: list[ScientificPaper]


class RetrievalResult(BaseModel):
    """Unranked candidate union; top_k is explicitly per provider."""

    original_claim: str
    top_k: int
    limit_scope: Literal["per_provider"] = "per_provider"
    from_publication_date: date | None = None
    to_publication_date: date | None = None
    papers: list[RetrievedPaper] = Field(default_factory=list)
    providers: dict[str, ProviderRun] = Field(default_factory=dict)
    seconds: float = 0
    http_attempts: int = 0
