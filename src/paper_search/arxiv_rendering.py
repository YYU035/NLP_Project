"""Auditable arXiv discovery selection, separate from claim generation."""

import re
from dataclasses import dataclass

from paper_search.linguistic import SpacyConceptExtractor
from paper_search.query_construction import build_focused_query

_REPORTING = re.compile(
    r"^(?:(?:preliminary|experimental|our|this|the)\s+)*"
    r"(?:findings|results|study|experiments)$", re.IGNORECASE,
)
_MEASUREMENT = re.compile(
    r"^(?:presence|expression|risk|efficiency|concentrations?) of "
    r"(?:(?:the|a|an) )?", re.IGNORECASE,
)


@dataclass
class DiscoverySelection:
    """Retained surface phrases and complementary omitted source spans."""

    phrases: list[str]
    omitted: list[str]
    extraction_status: str


def select_discovery_concepts(
    query: str, *, extractor: SpacyConceptExtractor | None = None
) -> DiscoverySelection:
    """Select the first two substantive concepts in source order.

    Omit reporting phrases and standalone numerical conditions. Preserve the
    spelling of retained concepts; do not infer synonyms or claim polarity.
    All omitted text remains available for later relevance assessment.
    """
    extraction = (extractor or SpacyConceptExtractor()).extract(query)
    if extraction.status != "ok":
        return DiscoverySelection([], [], extraction.status)
    concepts = build_focused_query(query, extraction).selected_concepts
    selected: list[tuple[int, int]] = []
    seen: set[str] = set()
    for concept in concepts:
        phrase = concept.text
        if _REPORTING.fullmatch(phrase):
            continue
        quantity = re.match(
            r"^[+−-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?%?\s+", phrase
        )
        if quantity and phrase[quantity.end():].casefold() not in {
            "capacity", "efficiency", "yield", "concentration",
        }:
            continue
        if not quantity and re.fullmatch(
            r"[+−-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?:%|°[CFK])?",
            phrase,
        ):
            continue
        prefix = _MEASUREMENT.match(phrase)
        start = concept.start + (
            quantity.end() if quantity else prefix.end() if prefix else 0
        )
        end = concept.end
        if query[start:end][:1] in {'"', "'", "“"} and (
            query[end - 1:end] in {'"', "'", "”"}
        ):
            start += 1
            end -= 1
        phrase = query[start:end]
        if phrase.casefold() in seen:
            continue
        seen.add(phrase.casefold())
        selected.append((start, end))
        if len(selected) == 2:
            break
    omitted = []
    cursor = 0
    for start, end in selected:
        if query[cursor:start].strip():
            omitted.append(query[cursor:start])
        cursor = end
    if query[cursor:].strip():
        omitted.append(query[cursor:])
    return DiscoverySelection(
        [query[start:end] for start, end in selected], omitted,
        extraction.status,
    )
