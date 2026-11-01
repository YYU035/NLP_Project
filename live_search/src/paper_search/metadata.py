"""Canonical paper metadata shared across providers."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class ScientificPaper(BaseModel):
    """Minimal provider-independent description of a scientific paper."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str | None = Field(default=None, min_length=1)
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    year: int | None = Field(default=None, ge=1)
    publication_date: date | None = None
    doi: str | None = None
    source: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    landing_url: HttpUrl | None = None
    pdf_url: HttpUrl | None = None
    pdf_urls: list[HttpUrl] = Field(default_factory=list)


PaperMetadata = ScientificPaper  # Compatibility with earlier callers.
