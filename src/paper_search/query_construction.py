"""Select meaningful concepts and build conservative, unquoted alternatives.

Extraction describes source spans; this module alone decides whether those
spans support a useful reformulation. No provider syntax is generated here.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field

from paper_search.linguistic import Concept, ExtractionResult

_WORDS = re.compile(r"\w+(?:[-/+.]\w+)*", re.UNICODE)
_REPORTING_FRAME = re.compile(
    r"^(?:"
    r"(?:the results of (?:this|our) study|our (?:results|experiments)|"
    r"experimental results|(?:this|our|the) (?:study|paper|work)) "
    r"(?:show|shows|demonstrate|demonstrates) that\s+|"
    r"in this study,? we (?:show|demonstrate) that\s+|"
    r"it has been (?:shown|demonstrated) that\s+)",
    re.IGNORECASE,
)
# This list selects concepts; it is never a stopword-removal list for queries.
_GENERIC = frozenset({
    "study", "studies", "paper", "work", "results", "result", "finding",
    "findings", "method", "methods", "approach", "system", "model",
    "analysis", "framework", "effect", "performance", "risk", "expression",
    "efficiency", "presence", "response", "treatment", "evidence",
    "cells", "protein", "devices",
})
_MEASUREMENT_HEADS = frozenset({
    "risk", "expression", "efficiency", "concentration", "concentrations",
    "capacity", "performance", "velocity", "presence", "rate", "rates",
    "response",
})
_OF_BRIDGE = re.compile(r"\s+of\s+(?:(?:the|a|an)\s+)?", re.IGNORECASE)


class FocusedQueryResult(BaseModel):
    """Construction decision and auditable concept selection/recall caveats."""

    status: Literal["built", "abstained"]
    query: str | None = None
    reason: str
    selected_concepts: list[Concept] = Field(default_factory=list)
    removed_prefix: str | None = None
    recall_risks: list[str] = Field(default_factory=list)


def _technical(text: str) -> bool:
    """Recognize literal scientific identifiers without inventing synonyms."""
    return bool(re.search(r"[A-Z]{2}|\d|[α-ωΑ-Ω]|\w[-/+]\w|\w\+", text))


def _select_concepts(
    claim: str, concepts: list[Concept], start: int
) -> list[Concept]:
    """Prefer multiword concepts; join measurement heads to their referents."""
    spans = [c.model_copy(deep=True) for c in concepts if c.start >= start]
    combined: list[Concept] = []
    for concept in sorted(spans, key=lambda c: (c.start, c.end)):
        if combined:
            previous = combined[-1]
            if (previous.text.casefold() in _MEASUREMENT_HEADS and
                    _OF_BRIDGE.fullmatch(claim[previous.end:concept.start])):
                previous.end = concept.end
                previous.text = claim[previous.start:previous.end]
                previous.kind = "composed_phrase"
                continue
            if (
                _technical(previous.text) and
                claim[previous.end:concept.start].isspace() and
                concept.text.casefold() in {"cells", "protein", "devices"}
            ):
                previous.end = concept.end
                previous.text = claim[previous.start:previous.end]
                previous.kind = "composed_phrase"
                continue
        combined.append(concept)
    selected = []
    for concept in combined:
        tokens = _WORDS.findall(concept.text)
        substantive = [t for t in tokens if t.casefold() not in {
            "the", "a", "an", "of", "our", "this", "these"
        }]
        if not substantive:
            continue
        if len(substantive) == 1 and substantive[0].casefold() in _GENERIC:
            continue
        if any(t.casefold() not in _GENERIC for t in substantive):
            selected.append(concept)
    return selected


def build_focused_query(
    claim: str, extraction: ExtractionResult
) -> FocusedQueryResult:
    """Build a concept-anchored assertion without speculative semantic edits.

    Only recognized, affirmative reporting frames can be removed. The whole
    scientific assertion linking concepts is retained, so populations, verbs,
    negation, comparisons, and conditions cannot disappear into a noun list.
    Short direct claims intentionally have no manufactured second query.
    """
    if extraction.status != "ok":
        return FocusedQueryResult(
            status="abstained", reason="extraction_unavailable_or_failed"
        )
    ordered = sorted(extraction.concepts, key=lambda c: (c.start, c.end))
    cursor = 0
    for concept in ordered:
        if (concept.start < cursor or concept.end <= concept.start or
                concept.end > len(claim) or
                claim[concept.start:concept.end] != concept.text):
            return FocusedQueryResult(
                status="abstained",
                reason="invalid_or_overlapping_source_spans",
            )
        cursor = concept.end
    match = _REPORTING_FRAME.match(claim)
    start = match.end() if match else 0
    selected = _select_concepts(claim, ordered, start)
    result = FocusedQueryResult(
        status="abstained", reason="no_useful_focused_alternative",
        selected_concepts=selected,
    )
    # A lone generic noun or one phrase is not a complete scientific focus.
    if len(selected) < 2 or not any(
        len(_WORDS.findall(c.text)) >= 2 or _technical(c.text)
        for c in selected
    ):
        result.reason = "insufficient_meaningful_concepts"
        return result
    if not match:
        # Do not delete polarity/uncertainty or rewrite an already direct
        # assertion just to satisfy a requested number of variants.
        return result
    candidate = claim[start:].strip()
    before_words = _WORDS.findall(claim)
    after_words = _WORDS.findall(candidate)
    removed = len(before_words) - len(after_words)
    if removed < 3 or removed / len(before_words) < 0.20:
        result.reason = "alternative_too_similar"
        return result
    result.status = "built"
    result.query = candidate
    result.reason = "concept_anchored_assertion_without_reporting_frame"
    result.removed_prefix = claim[:start]
    result.recall_risks = [
        "retained_populations_and_conditions_may_limit_recall",
        "provider_tokenization_and_default_term_matching_may_limit_recall",
    ]
    return result
