"""Conservative, deterministic paper identity grouping for evaluation."""

import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import unquote

from paper_search.metadata import ScientificPaper

_ARXIV = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:)"
    r"(\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?",
    re.IGNORECASE,
)


def normalized_doi(value: str | None) -> str | None:
    """Normalize resolver prefixes/case without removing DOI punctuation."""
    if not value:
        return None
    result = unquote(value).strip().casefold()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ):
        if result.startswith(prefix):
            result = result[len(prefix) :].strip()
            break
    return (
        result
        if result.startswith("10.") and "/" in result
        else None
    )


def arxiv_ids(paper: ScientificPaper) -> set[str]:
    """Find arXiv IDs in available metadata, ignoring version suffixes."""
    values = [paper.source_id, str(paper.landing_url or "")]
    values.extend(str(url) for url in paper.pdf_urls)
    if paper.pdf_url:
        values.append(str(paper.pdf_url))
    if paper.source == "arxiv" and ":" not in paper.source_id:
        values.append("arxiv:" + paper.source_id)
    return {
        match.group(1).casefold()
        for value in values
        for match in _ARXIV.finditer(value)
    }


def _title(value: str | None) -> str:
    """Keep case-sensitive scientific labels; normalize NFC and whitespace."""
    return " ".join(
        unicodedata.normalize("NFC", value or "").split()
    )


def _conflict(
    left: ScientificPaper, right: ScientificPaper
) -> bool:
    doi_left, doi_right = (
        normalized_doi(left.doi),
        normalized_doi(right.doi),
    )
    if doi_left and doi_right and doi_left != doi_right:
        return True
    a_left, a_right = arxiv_ids(left), arxiv_ids(right)
    return bool(
        a_left and a_right and not a_left.intersection(a_right)
    )


def same_paper(
    left: ScientificPaper, right: ScientificPaper
) -> bool:
    """Match strong IDs or exact long titles with same year/shared full author.

    Conflicting known identifiers never merge. No fuzzy title matching, author
    surname inference, or missing-year title matching is performed.
    """
    if _conflict(left, right):
        return False
    doi = normalized_doi(left.doi)
    if doi and doi == normalized_doi(right.doi):
        return True
    if arxiv_ids(left).intersection(arxiv_ids(right)):
        return True
    if (left.source, left.source_id) == (
        right.source,
        right.source_id,
    ):
        return True
    title = _title(left.title)
    if (
        title != _title(right.title)
        or len(title) < 30
        or len(title.split()) < 6
        or left.year is None
        or left.year != right.year
    ):
        return False
    authors = {
        _title(author).casefold() for author in left.authors if author.strip()
    }
    return bool(
        authors.intersection(_title(a).casefold() for a in right.authors)
    )


@dataclass
class PaperGroups:
    """Retain source records and assignments aligned with input order."""

    groups: list[list[ScientificPaper]]
    assignments: list[int]


def group_papers(papers: list[ScientificPaper]) -> PaperGroups:
    """Group in stable encounter order without conflicting-ID bridge merges."""
    groups: list[list[ScientificPaper]] = []
    assignments = []
    for paper in papers:
        for index, group in enumerate(groups):
            if all(
                not _conflict(paper, member) for member in group
            ) and any(
                same_paper(paper, member) for member in group
            ):
                group.append(paper)
                assignments.append(index)
                break
        else:
            assignments.append(len(groups))
            groups.append([paper])
    return PaperGroups(groups, assignments)
