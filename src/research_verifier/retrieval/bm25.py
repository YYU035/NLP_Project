"""Okapi BM25 over title + abstract.

Implemented directly (sparse, numpy) instead of via rank_bm25 so that:
  - scoring is vectorised over the 5k-doc SciFact corpus (fast per query),
  - the exact formula and parameters are visible for the report.

Defaults k1=0.9, b=0.4 follow the Anserini/BEIR BM25 baseline, so numbers
are comparable to published SciFact BEIR results.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from research_verifier.retrieval.schemas import Paper

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

# Compact English stopword list (no NLTK dependency).
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by can could did do does doing down during
each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no
nor not of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where
which while who whom why will with would you your yours yourself yourselves
""".split())


def _porter_stem(word: str) -> str:
    """Tiny suffix stripper. Install `nltk` and pass stemmer='porter' for the
    real Porter stemmer; this fallback just keeps tests dependency-free."""
    for suf in ("ational", "ization", "fulness", "ousness", "iveness", "ations",
                "ments", "ness", "ment", "ings", "ing", "edly", "ed", "ies", "es", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)] + ("y" if suf == "ies" else "")
    return word


def make_tokenizer(lowercase=True, remove_stopwords=True, stemmer: str | None = "porter"):
    stem = None
    if stemmer == "porter":
        try:
            from nltk.stem import PorterStemmer
            stem = PorterStemmer().stem
        except ImportError:
            stem = _porter_stem
    elif stemmer == "simple":
        stem = _porter_stem

    def tokenize(text: str) -> list[str]:
        toks = _TOKEN.findall(text.lower() if lowercase else text)
        if remove_stopwords:
            toks = [t for t in toks if t not in STOPWORDS]
        if stem:
            toks = [stem(t) for t in toks]
        return toks

    return tokenize


class BM25Retriever:
    name = "bm25"

    def __init__(self, k1: float = 0.9, b: float = 0.4, stemmer: str | None = "porter",
                 remove_stopwords: bool = True):
        self.k1, self.b = k1, b
        self.stemmer, self.remove_stopwords = stemmer, remove_stopwords
        self.tokenize = make_tokenizer(remove_stopwords=remove_stopwords, stemmer=stemmer)
        self.paper_ids: list[str] = []

    def settings(self) -> dict:
        return {"method": "bm25", "k1": self.k1, "b": self.b, "stemmer": self.stemmer,
                "remove_stopwords": self.remove_stopwords, "fields": "title+abstract"}

    def index(self, papers: list[Paper]) -> "BM25Retriever":
        self.paper_ids = [p.paper_id for p in papers]
        docs = [Counter(self.tokenize(p.index_text())) for p in papers]
        n = len(docs)
        lengths = np.array([sum(d.values()) for d in docs], dtype=np.float64)
        avgdl = lengths.mean() if n else 0.0

        df: Counter = Counter()
        for d in docs:
            df.update(d.keys())
        self.vocab = {t: i for i, t in enumerate(df)}
        # Lucene-style IDF: always positive, even for very common terms.
        self.idf = np.array([math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in self.vocab])

        # Precompute per-(term, doc) saturated TF weight; store postings per term.
        norm = self.k1 * (1 - self.b + self.b * lengths / (avgdl or 1.0))
        self.postings: list[tuple[np.ndarray, np.ndarray]] = [None] * len(self.vocab)  # type: ignore
        rows: dict[int, tuple[list, list]] = {}
        for j, d in enumerate(docs):
            for t, tf in d.items():
                w = tf * (self.k1 + 1) / (tf + norm[j])
                r = rows.setdefault(self.vocab[t], ([], []))
                r[0].append(j)
                r[1].append(w)
        for i, (js, ws) in rows.items():
            self.postings[i] = (np.array(js, dtype=np.int32), np.array(ws, dtype=np.float64))
        return self

    def score_all(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.paper_ids))
        # Each distinct query term counts once (query-side TF ignored, as in Anserini).
        for t in set(self.tokenize(query)):
            i = self.vocab.get(t)
            if i is None:
                continue
            js, ws = self.postings[i]
            scores[js] += self.idf[i] * ws
        return scores
