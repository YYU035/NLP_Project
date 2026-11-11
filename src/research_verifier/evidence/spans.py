"""Deterministic lexical evidence baselines over original source sentences.

No training, label access, implicit model downloads, or sentence re-splitting.
This is a new baseline, not the frontend's provisional regex selector.
"""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from math import log
import re
from typing import Sequence

_TOKEN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class EvidenceSpan:
    paper_id: str
    sentence_indices: tuple[int, ...]
    start: int
    end: int
    text: str
    score: float
    passage_id: str


def select_evidence(
    claim: str, paper_id: str, sentences: Sequence[str], *,
    window_size: int = 1, top_k: int = 3,
) -> list[EvidenceSpan]:
    """BM25-score single sentences or adjacent original-sentence windows.

    Parameters are fixed at k1=0.9, b=0.4; Unicode word tokens are casefolded,
    with no stemming/stopword removal. Query terms count once. Ties use start
    sentence index. Windows may overlap; evaluation deduplicates sentence IDs.
    Offsets reference exactly ' '.join(sentences), preserving source text.
    """
    if not claim.strip() or not paper_id:
        raise ValueError('Claim and paper identity must be nonempty')
    if type(window_size) is not int or window_size < 1:
        raise ValueError('window_size must be a positive integer')
    if type(top_k) is not int or top_k < 0:
        raise ValueError('top_k must be a nonnegative integer')
    if isinstance(sentences, str) or any(not isinstance(s, str) for s in sentences):
        raise ValueError('Original sentences must be a sequence of strings')
    body = ' '.join(sentences)
    starts, offset = [], 0
    for sentence in sentences:
        starts.append(offset)
        offset += len(sentence) + 1
    candidates = []
    for index in range(len(sentences)):
        indices = tuple(range(index, min(index + window_size, len(sentences))))
        start = starts[index]
        last = indices[-1]
        end = starts[last] + len(sentences[last])
        text = body[start:end]
        if text.strip():
            candidates.append((indices, start, end, text))
    if not candidates or top_k == 0:
        return []
    counts = [Counter(_TOKEN.findall(row[3].casefold())) for row in candidates]
    lengths = [sum(row.values()) for row in counts]
    average_length = sum(lengths) / len(lengths) or 1
    df = Counter(term for row in counts for term in row)
    query = set(_TOKEN.findall(claim.casefold()))
    scores = []
    for terms, length in zip(counts, lengths):
        score = 0.0
        for term in sorted(query):
            tf = terms.get(term, 0)
            if tf:
                idf = log(1 + (len(counts) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf * 1.9 / (tf + 0.9 * (0.6 + 0.4 * length / average_length))
        scores.append(score)
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i][0][0]))[:top_k]
    output = []
    for index in order:
        indices, start, end, text = candidates[index]
        pid = f'{paper_id}:abstract:{start}:{end}:{sha256(text.encode()).hexdigest()[:12]}'
        output.append(EvidenceSpan(paper_id, indices, start, end, text, scores[index], pid))
    return output
