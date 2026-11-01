"""Search-only orchestration across enabled scientific paper providers."""

import asyncio
import logging
from collections.abc import Mapping
from datetime import date

import httpx

from paper_search.clients.arxiv import ArxivClient
from paper_search.clients.base import PaperSearchClient
from paper_search.clients.openaire import OpenAIREClient
from paper_search.clients.openalex import OpenAlexClient
from paper_search.clients.semantic_scholar import SemanticScholarClient
from paper_search.config import Settings
from paper_search.exceptions import APIClientError
from paper_search.metadata import ScientificPaper
from paper_search.query_processing import SearchRequest

logger = logging.getLogger(__name__)


def build_clients(
    http: httpx.AsyncClient, settings: Settings
) -> dict[str, PaperSearchClient]:
    """Build only enabled adapters around a caller-owned HTTP client."""
    clients: dict[str, PaperSearchClient] = {}
    if settings.openalex_enabled:
        key = settings.openalex_api_key
        clients["openalex"] = OpenAlexClient(
            http, api_key=key.get_secret_value() if key else None,
            timeout_seconds=settings.timeout_seconds,
        )
    if settings.arxiv_enabled:
        clients["arxiv"] = ArxivClient(
            http, timeout_seconds=settings.timeout_seconds
        )
    if settings.semantic_scholar_enabled:
        key = settings.semantic_scholar_api_key
        clients["semantic_scholar"] = SemanticScholarClient(
            http, api_key=key.get_secret_value() if key else None,
            timeout_seconds=settings.timeout_seconds,
        )
    if settings.openaire_enabled:
        token = settings.openaire_token
        clients["openaire"] = OpenAIREClient(
            http, api_key=token.get_secret_value() if token else None,
            timeout_seconds=settings.timeout_seconds,
        )
    return clients


class SearchPipeline:
    """Aggregate provider results without ranking or PDF verification.

    Supply a persistent caller-owned HTTP client to reuse adapter pacing across
    calls, or let each run create and close its own one-shot HTTP session.
    Injected provider mappings are useful for offline integration tests.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
        clients: Mapping[str, PaperSearchClient] | None = None,
    ) -> None:
        """Configure providers without performing network I/O."""
        if http_client is not None and clients is not None:
            raise ValueError("Supply either http_client or clients")
        self.settings = settings if settings is not None else Settings()
        self._clients = (
            dict(clients) if clients is not None else
            build_clients(http_client, self.settings)
            if http_client is not None else None
        )

    async def run(
        self,
        query: str,
        *,
        max_results: int | None = None,
        from_publication_date: date | None = None,
        to_publication_date: date | None = None,
    ) -> list[ScientificPaper]:
        """Search every enabled provider, capping results per provider.

        Expected provider failures are logged while successful lists are kept.
        If all enabled providers fail, raise APIClientError. No enabled client
        is a configuration error. An empty successful response remains valid.
        """
        request = SearchRequest(
            query=query,
            max_results=(self.settings.max_results if max_results is None
                         else max_results),
            from_publication_date=from_publication_date,
            to_publication_date=to_publication_date,
        )
        enabled = [
            name for name in (
                "openalex", "arxiv", "semantic_scholar", "openaire"
            )
            if getattr(self.settings, f"{name}_enabled")
        ]
        if not enabled:
            raise ValueError("At least one search provider must be enabled")
        if self._clients is not None:
            return await self._search(self._clients, enabled, request)
        async with httpx.AsyncClient() as http:
            return await self._search(
                build_clients(http, self.settings), enabled, request
            )

    async def _search(
        self,
        clients: Mapping[str, PaperSearchClient],
        enabled: list[str],
        request: SearchRequest,
    ) -> list[ScientificPaper]:
        """Isolate expected provider failures and preserve provider order."""
        if any(name not in clients for name in enabled):
            raise ValueError("Missing client for an enabled provider")

        async def search_one(name: str) -> list[ScientificPaper] | None:
            try:
                return await clients[name].search(**request.model_dump())
            except APIClientError:
                logger.warning("Search provider %s failed", name)
                return None

        results = await asyncio.gather(*(search_one(name) for name in enabled))
        if all(result is None for result in results):
            raise APIClientError("All enabled search providers failed")
        return [paper for result in results if result is not None
                for paper in result]
