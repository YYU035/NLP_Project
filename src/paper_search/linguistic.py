"""POS and dependency-based concept extraction with surface-form recovery."""

import logging
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Callable

    from spacy.language import Language
    from spacy.tokens import Doc

logger = logging.getLogger(__name__)
_ATOM = re.compile(r'[^\s"“”]+')
_QUOTED = re.compile(r'''(?<!\w)(?:"[^"\n]+"|“[^”\n]+”|'[^'\n]+')(?!\w)''')
_SURFACE_EXPRESSION = re.compile(
    r"\b[Vv]itamin\s+[A-Z]\d*\b|"
    r"(?<!\w)[+−-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*"
    r"(?:°[CFK]|(?:[kmunµμ]?(?:Hz|g|m|L|M|V|A|W|Pa))"
    r"(?:/(?:kg|mL|L|s|m))?)(?!\w)"
)


class Concept(BaseModel):
    """An exact source span; dependency labels suggest roles, not ontology."""

    text: str
    start: int
    end: int
    kind: str
    dependency: str | None = None


class ExtractionResult(BaseModel):
    """Auditable extraction outcome, including failures and model identity."""

    status: Literal["ok", "not_requested", "unavailable", "failed"]
    concepts: list[Concept] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    model: str | None = None
    model_version: str | None = None
    pos_tags: list[tuple[str, str]] = Field(default_factory=list)


@lru_cache(maxsize=2)
def _load_model(name: str) -> "Language":
    """Load installed weights lazily; never download at runtime."""
    import spacy

    return spacy.load(name, exclude=["lemmatizer"])


def _expand_atom(text: str, start: int, end: int) -> tuple[int, int]:
    """Recover split scientific atoms such as TNF-α, HER2+, and °C."""
    while end > start and text[end - 1] in ".,;!?":
        end -= 1
    for match in _ATOM.finditer(text):
        left, right = match.span()
        while right > left and text[right - 1] in ".,;!?":
            right -= 1
        if left <= start < right:
            start = left
        if left < end <= right:
            end = right
    return start, end


class SpacyConceptExtractor:
    """Extract noun chunks and entities with deterministic model inference.

    Dependency objects/complements can contain outcomes or populations, but
    no clinical role classification or domain-specific NER is claimed.
    """

    def __init__(
        self, *, model: str = "en_core_web_sm",
        nlp: "Callable[[str], Doc] | None" = None,
    ) -> None:
        """Allow annotated-doc injection for stable offline unit tests."""
        self._model = model
        self._nlp = nlp

    def extract(self, claim: str) -> ExtractionResult:
        """Return explicit diagnostics instead of silently using regex NLP."""
        try:
            nlp = self._nlp if self._nlp is not None else (
                _load_model(self._model)
            )
        except (ImportError, OSError):
            logger.warning("Linguistic query model is unavailable")
            return ExtractionResult(
                status="unavailable", model=self._model,
                warnings=["model_unavailable: install the nlp-model extra"],
            )
        metadata = getattr(nlp, "meta", {})
        result = ExtractionResult(
            status="ok", model=self._model,
            model_version=metadata.get("version"),
        )
        if claim.count('"') % 2 or claim.count("“") != claim.count("”"):
            result.status = "failed"
            result.warnings.append("unbalanced_quotes:original_retained")
            return result
        try:
            doc = nlp(claim)
            if doc.text != claim:
                raise ValueError("Parser changed the source text")
            if doc.lang_ != "en":
                raise ValueError("Only English noun chunks are supported")
            if not doc.has_annotation("POS") or not doc.has_annotation("DEP"):
                raise ValueError("POS and dependency annotations required")
            result.pos_tags = [(token.text, token.pos_) for token in doc]
            spans = []
            quotes = [match.span() for match in _QUOTED.finditer(claim)]
            for chunk in doc.noun_chunks:
                if not any(t.pos_ in {"NOUN", "PROPN"} for t in chunk):
                    continue
                start = chunk.start_char
                # Remove only a syntactically tagged article, not no/every,
                # scientific A labels, or qualifiers inside the phrase.
                first = chunk[0]
                if (
                    len(chunk) > 1 and first.pos_ == "DET" and
                    first.text in {"the", "The", "a", "an", "An"} and
                    chunk[1].text.casefold() not in {"priori", "posteriori"}
                ):
                    start = chunk[1].idx
                start, end = _expand_atom(claim, start, chunk.end_char)
                if any(start < right and end > left for left, right in quotes):
                    continue
                spans.append(Concept(
                    text=claim[start:end], start=start, end=end,
                    kind="noun_phrase", dependency=chunk.root.dep_,
                ))
            for entity in doc.ents:
                start, end = _expand_atom(
                    claim, entity.start_char, entity.end_char
                )
                if any(start < right and end > left for left, right in quotes):
                    continue
                spans.append(Concept(
                    text=claim[start:end], start=start, end=end,
                    kind=f"entity:{entity.label_}",
                ))
            # These literal source patterns repair common scientific boundary
            # errors; they do not translate names or expand abbreviations.
            for match in _SURFACE_EXPRESSION.finditer(claim):
                start, end = match.span()
                if not any(start < right and end > left
                           for left, right in quotes):
                    spans.append(Concept(
                        text=claim[start:end], start=start, end=end,
                        kind="scientific_expression",
                    ))
            for start, end in quotes:
                spans.append(Concept(
                    text=claim[start:end], start=start, end=end,
                    kind="quoted_phrase",
                ))
            # Merge overlaps introduced by NER or scientific token recovery.
            # Adjacent phrases remain separate; source order never changes.
            for span in sorted(spans, key=lambda s: (s.start, -s.end)):
                if result.concepts and span.start < result.concepts[-1].end:
                    previous = result.concepts[-1]
                    previous.end = max(previous.end, span.end)
                    previous.text = claim[previous.start:previous.end]
                else:
                    result.concepts.append(span)
            if not result.concepts:
                result.status = "failed"
                result.warnings.append("no_concepts_extracted")
        except (ValueError, RuntimeError, NotImplementedError) as exc:
            result.status = "failed"
            result.concepts = []
            result.warnings.append(f"extraction_error:{type(exc).__name__}")
            logger.warning("Linguistic concept extraction failed")
        return result

