"""Opt-in query relaxation plans; never discard required scientific terms."""

from dataclasses import dataclass

from paper_search.query_rendering import render_query


def quote_phrases(phrases: list[str]) -> str:
    """Preserve phrase boundaries for the existing literal renderer."""
    if not phrases or any(
        not p.strip() or '"' in p or "\\" in p for p in phrases
    ):
        raise ValueError("Expected nonempty, unescaped scientific phrases")
    return " ".join('"' + " ".join(p.split()) + '"' for p in phrases)


@dataclass(frozen=True)
class RelaxedQuery:
    """One exact client input and the constraints intentionally relaxed."""

    query: str
    reason: str
    omitted_phrases: tuple[str, ...] = ()


def relaxation_plan(
    required: list[str], optional: list[str], *, max_attempts: int = 2
) -> list[RelaxedQuery]:
    """Drop explicit optional phrases first, then relax required phrase order.

    No ontology or model guesses what is optional. All spaCy-selected concepts
    are required unless a caller explicitly annotates otherwise. The last
    step retains every required token, acronym, symbol and quantity, using
    the existing literal renderer to quote syntax-sensitive single atoms.
    Dates are not part of these inputs and remain unchanged by the client.
    """
    if type(max_attempts) is not int or not 0 <= max_attempts <= 2:
        raise ValueError("Use zero, one or two fallback attempts")
    if not required:
        return []
    quote_phrases(required + optional)
    if set(required) & set(optional):
        raise ValueError("A required phrase cannot also be optional")
    primary = quote_phrases(required + optional)
    seen = {render_query(primary, "arxiv", arxiv_strategy="literal").text}
    results = []
    remaining = list(optional)
    removed = []
    while remaining and len(results) < max_attempts:
        removed.append(remaining.pop())
        query = quote_phrases(required + remaining)
        wire = render_query(query, "arxiv", arxiv_strategy="literal").text
        if wire not in seen:
            results.append(
                RelaxedQuery(
                    query, "explicit_optional_phrase_removed", tuple(removed)
                )
            )
            seen.add(wire)
    if len(results) < max_attempts and not remaining:
        query = " ".join(required)
        wire = render_query(query, "arxiv", arxiv_strategy="literal").text
        if wire not in seen:
            results.append(
                RelaxedQuery(
                    query,
                    "required_terms_kept_phrase_order_relaxed",
                    tuple(removed),
                )
            )
    return results
