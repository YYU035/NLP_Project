"""OpenAIRE Graph v3 publication search using free/public metadata."""

import asyncio
import re
from collections import deque
from datetime import date
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

import httpx

from paper_search.clients.http import HTTPProvider
from paper_search.clients.normalization import (
    doi,
    mapping,
    publication_date,
    text,
    url,
)
from paper_search.exceptions import APIClientError
from paper_search.metadata import ScientificPaper
from paper_search.query_processing import SearchRequest


def _items(value: object) -> list:
    """Treat missing or malformed lists as absent metadata."""
    return value if isinstance(value, list) else []


class _PlainText(HTMLParser):
    """Strip description markup while preserving inline scientific symbols."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in {"p", "div", "br", "jats:p"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "div", "jats:p"}:
            self.parts.append(" ")


def _plain(value: object) -> str | None:
    value = text(value)
    if value is None:
        return None
    parser = _PlainText()
    parser.feed(value)
    return text(" ".join("".join(parser.parts).split()))


def render_openaire_query(query: str) -> str:
    """Use keyword search without converting claim negation into exclusions.

    The experiment's inputs contain no explicit Boolean expressions. For an
    uppercase reserved operator, quote the complete literal input rather than
    interpreting user words as an exclusion or an unsupported expression.
    """
    normalized = " ".join(query.split())
    if re.search(r"\b(?:AND|OR|NOT)\b", normalized):
        return '"' + normalized.replace('"', '\\"') + '"'
    return normalized


def _is_pdf(link: str) -> bool:
    """Count explicit PDF-shaped metadata links only; never fabricate URLs."""
    parts = urlsplit(link)
    return bool(
        re.search(r"\.pdf(?:$|/)", parts.path, re.IGNORECASE)
        or (
            parts.hostname in {"arxiv.org", "www.arxiv.org"}
            and parts.path.startswith("/pdf/")
        )
        or re.search(
            r"(?:^|&)(?:format=pdf|pdf=render)(?:&|$)", parts.query
        )
    )


def parse_work(record: dict[str, Any]) -> ScientificPaper:
    """Map graph descriptions to abstract text, with that caveat documented."""
    identifier = text(record.get("id"))
    if identifier is None:
        raise ValueError("OpenAIRE record has no identifier")
    published = publication_date(record.get("publicationDate"))
    authors = []
    for author in _items(record.get("authors")):
        author = mapping(author)
        name = text(author.get("fullName")) or text(
            " ".join(
                str(author.get(key) or "")
                for key in ("name", "surname")
            )
        )
        if name and name not in authors:
            authors.append(name)
    descriptions = [
        _plain(s) for s in _items(record.get("descriptions"))
    ]
    instances = [
        mapping(i) for i in _items(record.get("instances"))
    ]
    pids = _items(record.get("pids")) + [
        pid
        for instance in instances
        for pid in _items(instance.get("pids"))
    ]
    work_doi = None
    arxiv = None
    for pid in pids:
        pid = mapping(pid)
        if str(pid.get("scheme", "")).casefold() == "doi":
            work_doi = work_doi or doi(pid.get("value"))
        if str(pid.get("scheme", "")).casefold() == "arxiv":
            value = text(pid.get("value"))
            if value and re.fullmatch(
                r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?",
                value,
                re.IGNORECASE,
            ):
                arxiv = arxiv or "https://arxiv.org/abs/" + value
    links = []
    for instance in instances:
        for value in _items(instance.get("urls")):
            link = url(value)
            if link and link not in links:
                links.append(link)
    pdfs = [link for link in links if _is_pdf(link)]
    landing = arxiv or next(
        (link for link in links if link not in pdfs), None
    )
    landing = landing or work_doi
    return ScientificPaper(
        source="openaire",
        source_id=identifier,
        title=_plain(record.get("mainTitle")),
        authors=authors,
        abstract=next((s for s in descriptions if s), None),
        doi=work_doi,
        publication_date=published,
        year=published.year if published else None,
        landing_url=landing,
        pdf_url=pdfs[0] if pdfs else None,
        pdf_urls=pdfs,
    )


class OpenAIREClient(HTTPProvider):
    """A compatible adapter; callers must coordinate the hourly quota.

    Anonymous access is documented as at most 60 requests/hour, authenticated
    access at most 7,200/hour. Per-instance pacing is not an account-wide quota
    manager. The standalone experiment caps all physical attempts at twenty.
    """

    endpoint = "https://api.openaire.eu/graph/v3/research-products"
    provider_name = "OpenAIRE"
    min_interval = 3.0

    def __init__(
        self, client: httpx.AsyncClient, **options: Any
    ) -> None:
        """Retain shared transport options and a conservative hourly budget."""
        super().__init__(client, **options)
        self._quota_lock = asyncio.Lock()
        self._request_times: deque[float] = deque()

    async def _send(
        self, params: dict[str, str], headers: dict[str, str]
    ) -> httpx.Response:
        """At most sixty attempts per rolling hour per reused instance."""
        async with self._quota_lock:
            current = self._clock()
            while (
                self._request_times
                and current - self._request_times[0] >= 3600
            ):
                self._request_times.popleft()
            if len(self._request_times) >= 60:
                await self._sleep(
                    3600 - (current - self._request_times[0])
                )
                self._request_times.popleft()
            try:
                return await super()._send(params, headers)
            finally:
                # Count actual paced starts, including transport failures.
                self._request_times.append(
                    self._last_request if self._last_request is not None
                    else self._clock()
                )

    def auth_headers(self) -> dict[str, str]:
        """Optional free registered-access Bearer token; no paid access."""
        return (
            {"Authorization": "Bearer " + self._api_key}
            if self._api_key
            else {}
        )

    async def search(
        self,
        query: str,
        *,
        max_results: int = 20,
        from_publication_date: date | None = None,
        to_publication_date: date | None = None,
    ) -> list[ScientificPaper]:
        """Search publications with inclusive dates and relevance ordering."""
        request = SearchRequest(
            query=query,
            max_results=max_results,
            from_publication_date=from_publication_date,
            to_publication_date=to_publication_date,
        )
        params = {
            "search": render_openaire_query(request.query),
            "type": "publication",
            "page": "1",
            "pageSize": str(request.max_results),
            "sortBy": "relevance DESC",
        }
        if request.from_publication_date:
            params["fromPublicationDate"] = (
                request.from_publication_date.isoformat()
            )
        if request.to_publication_date:
            params["toPublicationDate"] = (
                request.to_publication_date.isoformat()
            )
        response = await self._get(params)
        try:
            payload = response.json()
            total = payload["header"]["numFound"]
            records = payload["results"]
            if (
                type(total) is not int
                or total < 0
                or not isinstance(records, list)
            ):
                raise ValueError
            if not records and total > 0:
                raise ValueError
        except (ValueError, KeyError, TypeError):
            raise APIClientError(
                "OpenAIRE returned an invalid search envelope"
            ) from None
        papers = self._papers(
            response, "results", parse_work, request.max_results
        )
        if records and not papers:
            raise APIClientError(
                "OpenAIRE records could not be parsed"
            )
        return papers
