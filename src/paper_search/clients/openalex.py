"""Asynchronous OpenAlex works search with bounded transient retries."""

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


def reconstruct_abstract(index: object) -> str | None:
    """Restore token order and repeats without allocating sparse arrays.

    Missing or malformed indexes return None rather than fabricated text.
    """
    if not isinstance(index, dict) or not index:
        return None
    positions: dict[int, str] = {}
    for token, offsets in index.items():
        if not isinstance(token, str) or not isinstance(offsets, list):
            return None
        for offset in offsets:
            if type(offset) is not int or offset < 0 or offset in positions:
                return None
            positions[offset] = token
    if not positions or sorted(positions) != list(range(len(positions))):
        return None
    return " ".join(positions[i] for i in range(len(positions)))


def parse_work(work: dict[str, Any]) -> ScientificPaper:
    """Map a work, retaining unknown fields as None or empty collections.

    Missing identifiers raise ValueError. Malformed optional dates and links
    are omitted; PDF links are metadata only and have not been verified.
    """
    identifier = text(work.get("id"))
    if identifier is None:
        raise ValueError("OpenAlex work has no identifier")
    authors = []
    authorships = work.get("authorships")
    for authorship in authorships if isinstance(authorships, list) else []:
        entry = mapping(authorship)
        name = text(mapping(entry.get("author")).get("display_name"))
        name = name or text(entry.get("raw_author_name"))
        if name:
            authors.append(name)
    locations = [work.get("best_oa_location"), work.get("primary_location")]
    if isinstance(work.get("locations"), list):
        locations.extend(work["locations"])
    pdfs = []
    for location in locations:
        link = url(mapping(location).get("pdf_url"))
        if link and link not in pdfs:
            pdfs.append(link)
    landing = url(
        mapping(work.get("primary_location")).get("landing_page_url")
    )
    if not landing:
        landing = next(
            (link for loc in locations
             if (link := url(mapping(loc).get("landing_page_url")))),
            None,
        )
    published = publication_date(work.get("publication_date"))
    year = publication_year(work.get("publication_year"), published)
    return ScientificPaper(
        title=text(work.get("title")) or text(work.get("display_name")),
        abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
        authors=authors,
        year=year,
        publication_date=published,
        doi=doi(work.get("doi")),
        source="openalex",
        source_id=identifier,
        landing_url=landing or url(work.get("doi")) or url(identifier),
        pdf_url=pdfs[0] if pdfs else None,
        pdf_urls=pdfs,
    )


class OpenAlexClient(HTTPProvider):
    """Search works using a caller-owned httpx client.

    This adapter never closes the supplied client. Each call requests at most
    100 works; no local ranking, PDF retrieval, or verification is performed.
    """

    endpoint = "https://api.openalex.org/works"
    provider_name = "OpenAlex"
    min_interval = 0.1

    def auth_headers(self) -> dict[str, str]:
        """Send an optional OpenAlex key as a bearer header."""
        if self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}
        return {}

    async def search(
        self,
        query: str,
        *,
        max_results: int = 20,
        from_publication_date: date | None = None,
        to_publication_date: date | None = None,
    ) -> list[ScientificPaper]:
        """Search a claim with inclusive date bounds and a 1–100 result cap.

        Invalid input raises ValueError/ValidationError; exhausted requests or
        invalid response envelopes raise APIClientError. Unidentifiable works
        are skipped with a warning, so the list can be shorter than the cap.
        """
        request = SearchRequest(
            query=query,
            max_results=max_results,
            from_publication_date=from_publication_date,
            to_publication_date=to_publication_date,
        )
        start, end = request.from_publication_date, request.to_publication_date
        params = {
            "search": render_query(request.query, "openalex").text,
            "per_page": str(max_results),
        }
        filters = []
        if start:
            filters.append(f"from_publication_date:{start.isoformat()}")
        if end:
            filters.append(f"to_publication_date:{end.isoformat()}")
        if filters:
            params["filter"] = ",".join(filters)
        response = await self._get(params)
        return self._papers(response, "results", parse_work, max_results)
