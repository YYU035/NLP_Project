"""Data contracts for the retrieval module (Person 2).

Input:  Paper records (from the SciFact corpus or from Person 1's live search).
Output: RankedList objects, one per claim per method, consumed by Person 3.

Keep `abstract` as a LIST OF SENTENCES. SciFact rationale annotations are
sentence indices into that list, so Person 3's evidence F1 only works if the
sentence boundaries survive retrieval unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: list[str]                    # sentence list; [] if unavailable
    source: str = "scifact"                # scifact | arxiv | semantic_scholar | openalex
    text_status: str = "abstract_only"     # full_text | abstract_only | unavailable
    year: int | None = None
    url: str | None = None
    full_text: str | None = None           # not indexed; passed through for Person 3
    extra: dict[str, Any] = field(default_factory=dict)

    def index_text(self) -> str:
        """Text that BM25/SBERT see: title + abstract (per the agreed spec)."""
        return (self.title + " " + " ".join(self.abstract)).strip()


@dataclass
class RetrievalResult:
    paper_id: str
    rank: int                              # 1-based, within this method
    score: float                           # the method's own score
    method: str                            # bm25 | sbert | rrf
    component_ranks: dict[str, int | None] = field(default_factory=dict)
    component_scores: dict[str, float | None] = field(default_factory=dict)


@dataclass
class RankedList:
    claim_id: str
    claim: str                             # original claim, unchanged
    method: str
    results: list[RetrievalResult]
    latency_ms: float                      # query time only, excludes indexing
    corpus: str = "scifact"                # scifact | live
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, papers: dict[str, Paper] | None = None) -> dict:
        """Serialise for Person 3. If `papers` is given, attach the paper text
        so the handoff file is self-contained."""
        d = asdict(self)
        if papers is not None:
            for r in d["results"]:
                p = papers[r["paper_id"]]
                r["paper"] = {
                    "paper_id": p.paper_id, "title": p.title,
                    "abstract": p.abstract, "full_text": p.full_text,
                    "text_status": p.text_status, "source": p.source,
                    "year": p.year, "url": p.url,
                }
        return d
