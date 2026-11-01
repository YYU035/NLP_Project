"""arXiv Atom search adapter; no authentication is required."""

import logging
from datetime import date
from typing import Literal
from xml.etree import ElementTree as ET

from paper_search.clients.http import HTTPProvider
from paper_search.clients.normalization import doi, publication_date, text, url
from paper_search.exceptions import APIClientError
from paper_search.metadata import ScientificPaper
from paper_search.query_processing import SearchRequest
from paper_search.query_rendering import render_query

logger = logging.getLogger(__name__)
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"


def _content(entry: ET.Element, tag: str) -> str | None:
    """Normalize XML whitespace without losing mathematical content."""
    value = text(entry.findtext(tag))
    return " ".join(value.split()) if value else None


def parse_entry(entry: ET.Element) -> ScientificPaper:
    """Normalize an entry; published is the original submission date."""
    identifier = _content(entry, ATOM + "id")
    if identifier is None:
        raise ValueError("arXiv entry has no identifier")
    published = _content(entry, ATOM + "published")
    published_date = publication_date(published[:10] if published else None)
    pdfs = []
    landing = None
    for link in entry.findall(ATOM + "link"):
        href = url(link.get("href"))
        if href and (link.get("title") == "pdf" or
                     link.get("type") == "application/pdf"):
            if href not in pdfs:
                pdfs.append(href)
        elif href and link.get("rel") == "alternate":
            landing = href
    authors = [
        name for author in entry.findall(ATOM + "author")
        if (name := _content(author, ATOM + "name"))
    ]
    return ScientificPaper(
        title=_content(entry, ATOM + "title"),
        abstract=_content(entry, ATOM + "summary"), authors=authors,
        doi=doi(_content(entry, ARXIV + "doi")),
        publication_date=published_date,
        year=published_date.year if published_date else None,
        source="arxiv", source_id=identifier,
        landing_url=landing or url(identifier),
        pdf_url=pdfs[0] if pdfs else None, pdf_urls=pdfs,
    )


class ArxivClient(HTTPProvider):
    """Search Atom feeds with three seconds between request starts."""

    endpoint = "https://export.arxiv.org/api/query"
    provider_name = "arXiv"
    min_interval = 3.0
    query_strategy: Literal["concepts", "literal"] = "concepts"

    async def search(
        self,
        query: str,
        *,
        max_results: int = 20,
        from_publication_date: date | None = None,
        to_publication_date: date | None = None,
    ) -> list[ScientificPaper]:
        """Search scientific concepts; bounds apply to submission dates."""
        request = SearchRequest(
            query=query, max_results=max_results,
            from_publication_date=from_publication_date,
            to_publication_date=to_publication_date,
        )
        expression = render_query(
            request.query, "arxiv", arxiv_strategy=self.query_strategy
        ).text
        start, end = request.from_publication_date, request.to_publication_date
        if start or end:
            lower = start.isoformat().replace("-", "") + "0000" if start else (
                "000101010000"
            )
            upper = end.isoformat().replace("-", "") + "2359" if end else (
                "999912312359"
            )
            expression = (
                f"({expression}) AND submittedDate:[{lower} TO {upper}]"
            )
        response = await self._get({
            "search_query": expression,
            "start": "0", "max_results": str(request.max_results),
        })
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as exc:
            raise APIClientError("arXiv returned invalid XML") from exc
        if root.tag != ATOM + "feed":
            raise APIClientError("arXiv response is not an Atom feed")
        entries = root.findall(ATOM + "entry")
        if any("/api/errors" in (entry.findtext(ATOM + "id") or "")
               for entry in entries):
            raise APIClientError("arXiv returned an Atom error entry")
        papers = []
        for entry in entries[:request.max_results]:
            try:
                papers.append(parse_entry(entry))
            except ValueError:
                logger.warning("Skipping malformed arXiv entry")
        return papers
