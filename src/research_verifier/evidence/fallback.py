"""Temporary abstract-only selector: exact adjacent sentence spans, no PDF claims."""
import re

from research_verifier.contracts.models import digest
from research_verifier.retrieval.schemas import Paper
from research_verifier.retrieval.bm25 import BM25Retriever
from research_verifier.verification.schemas import EvidencePassage


def select_abstract(claim, identity, abstract, *, top_k=3):
    if not abstract or not abstract.strip():
        return []
    # Keep offsets into original text; no rewriting of numbers or qualifiers.
    boundaries = [0] + [m.end() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9])", abstract)] + [len(abstract)]
    candidates = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[min(i + 2, len(boundaries) - 1)]
        raw = abstract[start:end]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        text = raw[left:right]
        if text:
            start, end = start + left, start + right
            pid = f"{identity}:abstract:{start}:{end}:{digest(text)[:12]}"
            candidates.append((pid, start, end, text))
    # A very long sentence remains whole; NLI marks overflow and excludes it.
    unique = list({row[0]: row for row in candidates}.values())
    if not unique:
        return []
    index = BM25Retriever().index([Paper(paper_id=p[0], title="", abstract=[p[3]]) for p in unique])
    scores = index.score_all(claim)
    order = sorted(range(len(unique)), key=lambda i: (-float(scores[i]), unique[i][1]))[:top_k]
    return [EvidencePassage(passage_id=unique[i][0], paper_id=identity, section="abstract",
                            text=unique[i][3], score=float(scores[i])) for i in order]
