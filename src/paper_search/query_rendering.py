"""Provider-specific query syntax with explicit scientific-text limitations."""

import re
from typing import Literal

from pydantic import BaseModel, Field

from paper_search.arxiv_rendering import select_discovery_concepts

Provider = Literal["openalex", "arxiv", "semantic_scholar"]
_PARTS = re.compile(r'''"([^"\n]*)"|“([^”\n]*)”|'([^'\n]*)'|(\S+)''')
_HYPHEN = re.compile(r"(?<=[^\W\d_])[-‐‑](?=\w)|(?<=\d)[-‐‑](?=[^\W\d_])")
_SCIENTIFIC_NUMBER = re.compile(r"[+-]?\d+(?:\.\d+)?[eE][+-]?\d+%?")


class RenderedQuery(BaseModel):
    """Keep canonical text alongside observable backend limitations."""

    original: str
    text: str
    warnings: list[str] = Field(default_factory=list)
    selected_concepts: list[str] = Field(default_factory=list)
    omitted_spans: list[str] = Field(default_factory=list)
    extraction_status: str | None = None


def _parts(query: str) -> list[tuple[str, bool]]:
    """Recognize phrase delimiters; never split hyphens, units, or decimals."""
    parts = []
    for match in _PARTS.finditer(query):
        groups = match.groups()
        quoted = groups[3] is None
        content = next(value for value in groups if value is not None)
        if content and any(character.isalnum() for character in content):
            parts.append((content, quoted))
        elif content and not all(char in ".,;!?" for char in content):
            parts.append((content, quoted))
    return parts


def _literal(text: str) -> str:
    """Quote a literal value without allowing user text to inject operators."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _arxiv_literal(text: str, explicit_phrase: bool) -> str:
    """Quote phrases/syntax-sensitive values, not ordinary single words."""
    if (explicit_phrase and len(text.split()) > 1) or text in {
        "AND", "OR", "NOT", "ANDNOT", "TO"
    } or re.fullmatch(r"[\w.]+", text, re.UNICODE) is None:
        return _literal(text)
    return text


def render_query(
    query: str, provider: Provider, *,
    arxiv_strategy: Literal["concepts", "literal"] = "concepts",
) -> RenderedQuery:
    """Render phrases where supported; expose unavoidable provider losses.

    Semantic Scholar relevance search documents no query syntax and no matches
    for hyphenated terms. Render internal letter-hyphens as spaces, preserving
    signs, ranges, exponents, case, and the original query in diagnostics.
    No backend can guarantee literal matching of every scientific symbol.
    """
    original = query
    normalized = " ".join(query.split())
    if not any(char.isalnum() for char in normalized):
        raise ValueError("Query must contain searchable words or numbers")
    result = RenderedQuery(original=original, text=original)
    parts = _parts(normalized)
    if provider == "arxiv":
        if arxiv_strategy == "concepts":
            selection = select_discovery_concepts(normalized)
            result.extraction_status = selection.extraction_status
            result.selected_concepts = selection.phrases
            result.omitted_spans = selection.omitted
            if selection.phrases:
                result.text = " AND ".join(
                    f"all:{_arxiv_literal(phrase, True)}"
                    for phrase in selection.phrases
                )
                result.warnings.extend([
                    "discovery_only_not_equivalent_to_claim",
                    "provider_index_controls_symbol_matching",
                ])
                return result
            # Fail conservatively: never fall back to every-word conjunction
            # or an empty/unbounded search when NLP cannot select concepts.
            result.text = f"all:{_literal(normalized)}"
            result.warnings.append("no_concepts_original_phrase_fallback")
            return result
        atoms = [(text.rstrip(".,;!?") if not quoted else text, quoted)
                 for text, quoted in parts]
        result.text = " AND ".join(
            f"all:{_arxiv_literal(atom, quoted)}" for atom, quoted in atoms
        )
        result.warnings.append("provider_index_controls_symbol_matching")
    elif provider == "openalex":
        # Normalize supported phrase delimiters and keep uppercase operator
        # words literal: a claim's NOT must never become Boolean exclusion.
        result.text = " ".join(
            _literal(part) if quoted or part in {"AND", "OR", "NOT"}
            else part for part, quoted in parts
        )
        result.warnings.append("provider_index_controls_symbol_matching")
    elif provider == "semantic_scholar":
        if any(quoted for _, quoted in parts):
            result.warnings.append("phrase_boundaries_not_supported")
        words = []
        for content, _ in parts:
            tokens = []
            for token in content.split():
                replacement = token if _SCIENTIFIC_NUMBER.fullmatch(
                    token.rstrip(".,;!?")
                ) else _HYPHEN.sub(" ", token)
                if replacement != token:
                    warning = "hyphenated_terms_rendered_as_words"
                    if warning not in result.warnings:
                        result.warnings.append(warning)
                tokens.append(replacement)
            words.append(" ".join(tokens))
        result.text = " ".join(words)
        result.warnings.append("provider_index_controls_symbol_matching")
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return result
