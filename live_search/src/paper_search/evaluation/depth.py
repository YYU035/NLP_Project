"""Offline response-prefix summaries; missing labels are always explicit."""

from collections.abc import Callable

from paper_search.deduplication import group_papers
from paper_search.metadata import ScientificPaper


def prefix_summary(
    papers: list[ScientificPaper],
    depth: int,
    label_for: Callable[[ScientificPaper], int | None],
) -> dict[str, int]:
    """Take a raw response-order prefix before dedup, without ranking papers.

    A single top-30 fetch supports nested candidate-yield comparisons, not
    estimates of the latency a separate top-10 request would have incurred.
    """
    if depth not in {10, 20, 30}:
        raise ValueError("Supported response prefixes are 10, 20 and 30")
    prefix = papers[:depth]
    groups = group_papers(prefix).groups
    labels = []
    for group in groups:
        known = {
            value for paper in group if (value := label_for(paper)) is not None
        }
        if not known <= {0, 1, 2}:
            raise ValueError("Expected relevance labels 0, 1, 2 or None")
        labels.append(next(iter(known)) if len(known) == 1 else None)
    return {
        "depth": depth,
        "returned": len(prefix),
        "unique": len(groups),
        "relevant_known": sum(value in {1, 2} for value in labels),
        "evidence_known": labels.count(2),
        "unknown": labels.count(None),
        "abstract_unique": sum(any(p.abstract for p in g) for g in groups),
        "pdf_unique": sum(
            any(p.pdf_url or p.pdf_urls for p in g) for g in groups
        ),
    }
