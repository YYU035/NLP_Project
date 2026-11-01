"""Provider contract for asynchronous paper-search adapters."""

from datetime import date
from typing import Protocol

from paper_search.metadata import ScientificPaper


class PaperSearchClient(Protocol):
    """An asynchronous search provider with caller-owned lifecycle."""

    async def search(
        self,
        query: str,
        *,
        max_results: int = 20,
        from_publication_date: date | None = None,
        to_publication_date: date | None = None,
    ) -> list[ScientificPaper]:
        """Return metadata or raise APIClientError on provider failure."""
        ...
