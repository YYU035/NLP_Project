"""Semantic Scholar Academic Graph relevance-search adapter."""

from datetime import date
from typing import Any

from paper_search.clients.http import HTTPProvider
from paper_search.clients.normalization import (
    doi,
    mapping,
    publication_date,
    publication_year,
    text,
    url,
)
from paper_search.metadata import ScientificPaper
from paper_search.query_processing import SearchRequest
from paper_search.query_rendering import render_query


def parse_paper(record: dict[str, Any]) -> ScientificPaper:
    """Normalize nullable Graph API fields into shared paper metadata."""
    identifier = text(record.get("paperId"))
    if identifier is None:
        raise ValueError("Semantic Scholar paper has no identifier")
    published = publication_date(record.get("publicationDate"))
    pdf = url(mapping(record.get("openAccessPdf")).get("url"))
    authors = record.get("authors")
    names = [
        name for author in authors
        if (name := text(mapping(author).get("name")))
    ] if isinstance(authors, list) else []
    return ScientificPaper(
        title=text(record.get("title")),
        abstract=text(record.get("abstract")), authors=names,
        doi=doi(mapping(record.get("externalIds")).get("DOI")),
        publication_date=published,
        year=publication_year(record.get("year"), published),
        source="semantic_scholar", source_id=identifier,
        landing_url=url(record.get("url")),
        pdf_url=pdf, pdf_urls=[pdf] if pdf else [],
    )


class SemanticScholarClient(HTTPProvider):
    """Search one page, pacing requests conservatively at one per second."""

    endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"
    provider_name = "Semantic Scholar"
    min_interval = 1.0

    def auth_headers(self) -> dict[str, str]:
        """Use the documented optional x-api-key header."""
        return {"x-api-key": self._api_key} if self._api_key else {}

    async def search(
        self,
        query: str,
        *,
        max_results: int = 20,
        from_publication_date: date | None = None,
        to_publication_date: date | None = None,
    ) -> list[ScientificPaper]:
        """Search a claim with inclusive date bounds and a per-provider cap."""
        request = SearchRequest(
            query=query, max_results=max_results,
            from_publication_date=from_publication_date,
            to_publication_date=to_publication_date,
        )
        params = {
            "query": render_query(request.query, "semantic_scholar").text,
            "limit": str(request.max_results),
            "fields": (
                "title,abstract,authors,externalIds,year,publicationDate,"
                "url,openAccessPdf"
            ),
        }
        start, end = request.from_publication_date, request.to_publication_date
        if start or end:
            params["publicationDateOrYear"] = (
                f"{start.isoformat() if start else ''}:"
                f"{end.isoformat() if end else ''}"
            )
        response = await self._get(params)
        # Observed live: a zero-total response can omit the data array.
        # Accept only the explicit first-page empty envelope, not arbitrary
        # missing/malformed results or a nonzero total without records.
        try:
            payload = response.json()
        except ValueError:
            payload = None  # Shared parsing below raises the domain error.
        if (
            isinstance(payload, dict) and "data" not in payload and
            type(payload.get("total")) is int and payload["total"] == 0 and
            type(payload.get("offset")) is int and payload["offset"] == 0
        ):
            return []
        return self._papers(response, "data", parse_paper, request.max_results)
