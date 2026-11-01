"""Optional local embedding keyphrases; no change to the spaCy strategy."""

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "2a7cca6ad7605f3183bce07f677bda658059bbe9"
# Editable-install default, anchored to the checkout rather than the caller.
# Other installations can inject a generator with an explicit model_path.
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts/local-models/all-MiniLM-L6-v2"
)
# Boundary/reporting rules select spans; source text is never stemmed.
_BOUNDARY = frozenset(
    "a an the this our of in at for to under with without "
    "after than that and or by on from compared".split()
)
_PREDICATES = frozenset(
    "show shows shown demonstrate demonstrates demonstrated results study "
    "experiments findings preliminary suggest suggests "
    "does do is are be been has have may might can not "
    "increase increases improve improves reduce reduces "
    "retain retains accelerate accelerates decreasing "
    "compared lower elevated".split()
)
_CONTEXT = frozenset(
    "at in under after without with by than".split()
)
_MEASUREMENT = re.compile(
    r"[+−-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s+"
    r"(?:°[CFK]|kHz|MHz|Hz|mg/kg|cycles)\b|\S+"
)


def candidate_phrases(claim: str) -> list[str]:
    """Enumerate exact 1–4-token surface spans, keeping scientific atoms.

    Unlike default CountVectorizer tokenization, whitespace atoms preserve
    TNF-α, SiC, CO2, signed numbers, units and vitamin labels. This is a small
    fixed rule-based candidate filter, not scientific entity recognition.
    """
    atoms = [
        value
        for m in _MEASUREMENT.finditer(claim)
        if (value := m.group().rstrip(".,;!?"))
    ]
    candidates = []
    for start in range(len(atoms)):
        for length in range(1, 5):
            words = atoms[start : start + length]
            if len(words) != length:
                continue
            lower = [word.casefold() for word in words]
            if lower[0] in _BOUNDARY or lower[-1] in _BOUNDARY:
                continue
            if any(word in _PREDICATES for word in lower):
                continue
            if any(word in _CONTEXT for word in lower):
                continue
            phrase = " ".join(words)
            if not any(c.isalpha() for c in phrase):
                continue
            if len(words) == 1 and (
                len(words[0]) == 1
                or words[0] in {"°C", "kHz", "cycles"}
                or lower[0] in {"risk", "expression", "presence"}
            ):
                continue
            if phrase not in candidates:
                candidates.append(phrase)
    return candidates


class KeyBERTResult(BaseModel):
    """Query plus extraction diagnostics; scores are not paper relevance."""

    queries: list[str]
    selected_phrases: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    candidate_count: int = 0
    omitted_spans: list[str] = Field(default_factory=list)
    status: str = "abstained"
    model: str = MODEL_ID
    revision: str = MODEL_REVISION


class KeyBERTQueryGenerator:
    """Keep the original and optionally add a 2–4-phrase embedding query.

    Compatible with QueryGenerator.generate(). Loading is lazy, local-only,
    CPU-only and opt-in. No spaCy candidate extraction or paid backend is used.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_queries: int = 2,
        phrase_limit: int = 3,
        extractor: Any = None,
    ) -> None:
        if type(max_queries) is not int or max_queries < 1:
            raise ValueError(
                "max_queries must be a positive integer"
            )
        if (
            type(phrase_limit) is not int
            or not 2 <= phrase_limit <= 4
        ):
            raise ValueError(
                "phrase_limit must be between two and four"
            )
        self.model_path = Path(model_path)
        self.max_queries = max_queries
        self.phrase_limit = phrase_limit
        self._extractor = extractor

    def _load(self) -> Any:
        """Never download weights or call an embedding service implicitly."""
        if self._extractor is None:
            if not self.model_path.is_dir():
                raise FileNotFoundError(
                    "Download the pinned local model first"
                )
            revision_file = self.model_path / "pinned_revision.txt"
            if (not revision_file.exists() or
                    revision_file.read_text().strip() != MODEL_REVISION):
                raise ValueError("Local model revision is not verified")
            try:
                from keybert import KeyBERT
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Install the optional keybert extra"
                ) from exc
            model = SentenceTransformer(
                str(self.model_path),
                device="cpu",
                local_files_only=True,
                trust_remote_code=False,
            )
            self._extractor = KeyBERT(model=model)
        return self._extractor

    def generate(
        self, claim: str, *, max_queries: int | None = None
    ) -> list[str]:
        """Retain the exact input first and obey the unchanged strategy API."""
        return self.generate_with_details(
            claim, max_queries=max_queries
        ).queries

    def generate_with_details(
        self, claim: str, *, max_queries: int | None = None
    ) -> KeyBERTResult:
        """Use cosine/MMR extraction and nonoverlapping source phrases."""
        limit = (
            self.max_queries if max_queries is None else max_queries
        )
        if type(limit) is not int or limit < 1:
            raise ValueError(
                "max_queries must be a positive integer"
            )
        if not isinstance(claim, str) or not any(
            c.isalnum() for c in claim
        ):
            raise ValueError("Claim must contain searchable text")
        result = KeyBERTResult(queries=[claim])
        if limit == 1:
            result.status = "original_only"
            return result
        from sklearn.feature_extraction.text import CountVectorizer

        normalized = " ".join(claim.split())
        candidates = candidate_phrases(normalized)
        result.candidate_count = len(candidates)
        if not candidates:
            return result
        vectorizer = CountVectorizer(
            analyzer=candidate_phrases, lowercase=False
        )
        ranked = self._load().extract_keywords(
            normalized,
            vectorizer=vectorizer,
            stop_words=None,
            top_n=min(20, len(candidates)),
            use_mmr=True,
            diversity=0.3,
        )
        spans: list[tuple[int, int]] = []
        # KeyBERT returns MMR-selected phrases sorted by cosine similarity.
        for phrase, score in ranked:
            if phrase not in candidates:
                continue
            start = normalized.find(phrase)
            if start < 0:
                continue
            end = start + len(phrase)
            if any(
                start < right and end > left
                for left, right in spans
            ):
                continue
            spans.append((start, end))
            result.selected_phrases.append(phrase)
            result.scores.append(float(score))
            if len(spans) == self.phrase_limit:
                break
        if len(spans) < 2:
            return result
        ordered = sorted(
            zip(
                spans,
                result.selected_phrases,
                result.scores,
                strict=True,
            )
        )
        result.selected_phrases = [
            phrase for _, phrase, _ in ordered
        ]
        result.scores = [score for _, _, score in ordered]
        cursor = 0
        for (start, end), _, _ in ordered:
            if normalized[cursor:start].strip():
                result.omitted_spans.append(
                    normalized[cursor:start]
                )
            cursor = end
        if normalized[cursor:].strip():
            result.omitted_spans.append(normalized[cursor:])
        query = " ".join(result.selected_phrases)
        if query != normalized:
            result.queries.append(query)
            result.status = "built"
        return result
