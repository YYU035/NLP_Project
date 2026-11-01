"""Shared search validation and interchangeable claim query generation."""

from datetime import date
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from paper_search.config import Settings
from paper_search.linguistic import (
    ExtractionResult,
    SpacyConceptExtractor,
)
from paper_search.query_construction import (
    FocusedQueryResult,
    build_focused_query,
)


class SearchRequest(BaseModel):
    """A claim, per-provider limit, and optional inclusive date bounds."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(min_length=1)
    max_results: int = Field(default=20, ge=1, le=100, strict=True)
    from_publication_date: date | None = None
    to_publication_date: date | None = None

    @model_validator(mode="after")
    def validate_date_range(self) -> Self:
        """Reject reversed ranges before provider requests begin."""
        start, end = self.from_publication_date, self.to_publication_date
        if start and end and start > end:
            raise ValueError("Publication start date must not exceed end date")
        return self


class QueryGenerator(Protocol):
    """Strategy contract usable by deterministic or future LLM generators.

    Implementations return ordered, unique queries with the original claim
    first, respect the limit, and reject blank claims and invalid limits.
    """

    def generate(
        self, claim: str, *, max_queries: int | None = None
    ) -> list[str]:
        """Return queries without executing retrieval or verification."""
        ...


class _GenerationRequest(BaseModel):
    """Validate generation independently of provider result limits."""

    model_config = ConfigDict(extra="forbid")
    claim: str = Field(min_length=1, strict=True)
    max_queries: int = Field(ge=1, strict=True)


QueryMode = Literal["original_only", "original_plus_concepts"]


class _GeneratorOptions(BaseModel):
    """Validated strategy configuration for reproducible ablations."""

    mode: QueryMode = "original_plus_concepts"
    max_queries: int = Field(default=2, ge=1, strict=True)


class QueryGenerationResult(BaseModel):
    """Queries and extraction diagnostics without changing generate's API."""

    queries: list[str]
    mode: QueryMode
    extraction: ExtractionResult
    construction: FocusedQueryResult | None = None
    normalized_claim: str


class DeterministicQueryGenerator:
    """Keep the exact original and add one useful concept-anchored query.

    Fixed spaCy weights produce deterministic annotations; no generative model,
    stemming, unsupported synonyms, or retrieval-quality claim is involved.
    """

    def __init__(
        self,
        *,
        max_queries: int = 2,
        mode: QueryMode = "original_plus_concepts",
        model: str = "en_core_web_sm",
        extractor: SpacyConceptExtractor | None = None,
    ) -> None:
        """Configure ablation and budget; load no model until needed."""
        self._options = _GeneratorOptions(mode=mode, max_queries=max_queries)
        self._extractor = extractor if extractor is not None else (
            SpacyConceptExtractor(model=model)
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Use explicit project configuration without implicit env reads."""
        return cls(
            max_queries=settings.max_queries,
            mode=settings.query_mode,
            model=settings.query_model,
        )

    def generate(
        self, claim: str, *, max_queries: int | None = None
    ) -> list[str]:
        """Keep the existing strategy interface, including per-call budgets."""
        details = self.generate_with_details(claim, max_queries=max_queries)
        return details.queries

    def generate_with_details(
        self, claim: str, *, max_queries: int | None = None
    ) -> QueryGenerationResult:
        """Expose concepts, POS tags, model version, and failures."""
        request = _GenerationRequest(
            claim=claim,
            max_queries=(self._options.max_queries if max_queries is None
                         else max_queries),
        )
        original = request.claim
        normalized = " ".join(original.split())
        if not any(character.isalnum() for character in normalized):
            raise ValueError("Claim must contain searchable words or numbers")
        result = QueryGenerationResult(
            queries=[original], mode=self._options.mode,
            extraction=ExtractionResult(status="not_requested"),
            normalized_claim=normalized,
        )
        if self._options.mode == "original_only" or request.max_queries == 1:
            return result
        result.extraction = self._extractor.extract(normalized)
        result.construction = build_focused_query(
            normalized, result.extraction
        )
        candidate = result.construction.query
        if candidate and candidate != normalized:
            result.queries.append(candidate)
        return result
