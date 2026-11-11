"""One interface for all three methods.

    r = Retriever(method="rrf").index(papers)
    ranked = r.search(claim, k=10, claim_id="13")   # -> RankedList

BM25, SBERT and RRF all return RankedList objects with the same fields, so
Person 3 and the evaluation code never need to know which method produced them.
"""
from __future__ import annotations

import time

import numpy as np

from research_verifier.retrieval.bm25 import BM25Retriever
from research_verifier.retrieval.embeddings import SBERTRetriever
from research_verifier.retrieval.schemas import Paper, RankedList, RetrievalResult

METHODS = ("bm25", "sbert", "rrf")


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, best first; ties broken by index for determinism."""
    k = min(k, len(scores))
    if k <= 0:
        return np.array([], dtype=int)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError('Ranking scores must be a finite one-dimensional array')
    cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
    higher = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)
    idx = np.concatenate((higher, tied[:k - len(higher)]))
    return idx[np.lexsort((idx, -scores[idx]))]


def rrf_fuse(rankings: dict[str, list[str]], k_rrf: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum_m 1 / (k_rrf + rank_m(d)).
    A document missing from a method's list contributes 0 for that method."""
    fused: dict[str, float] = {}
    for ranked_ids in rankings.values():
        for rank, pid in enumerate(ranked_ids, start=1):
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (k_rrf + rank)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


class Retriever:
    def __init__(self, method: str = "rrf", bm25: BM25Retriever | None = None,
                 sbert: SBERTRetriever | None = None, k_rrf: int = 60, fusion_depth: int = 100,
                 corpus: str = "scifact"):
        """fusion_depth = how many results each branch contributes to RRF."""
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method, self.k_rrf, self.fusion_depth, self.corpus = method, k_rrf, fusion_depth, corpus
        self.bm25 = bm25 if bm25 is not None else (BM25Retriever() if method in ("bm25", "rrf") else None)
        self.sbert = sbert if sbert is not None else (SBERTRetriever() if method in ("sbert", "rrf") else None)
        self.index_seconds: dict[str, float] = {}

    def settings(self) -> dict:
        s: dict = {"method": self.method, "corpus": self.corpus}
        if self.bm25:
            s["bm25"] = self.bm25.settings()
        if self.sbert:
            s["sbert"] = self.sbert.settings()
        if self.method == "rrf":
            s["rrf"] = {"k_rrf": self.k_rrf, "fusion_depth": self.fusion_depth}
        return s

    def index(self, papers: list[Paper], use_cache: bool = True) -> "Retriever":
        if not papers:
            raise ValueError("Cannot index an empty paper list")
        self.papers = {p.paper_id: p for p in papers}
        if len(self.papers) != len(papers):
            raise ValueError("Duplicate paper_ids in input; dedupe before indexing")
        if self.bm25:
            t0 = time.perf_counter()
            self.bm25.index(papers)
            self.index_seconds["bm25"] = time.perf_counter() - t0
        if self.sbert:
            self.sbert.index(papers, use_cache=use_cache)
            self.index_seconds["sbert"] = self.sbert.index_seconds
            self.index_seconds["sbert_from_cache"] = self.sbert.loaded_from_cache
        return self

    def _branch(self, retr, query: str, depth: int) -> tuple[list[str], dict[str, float]]:
        scores = retr.score_all(query)
        idx = _top_k(scores, depth)
        ids = [retr.paper_ids[i] for i in idx]
        return ids, {retr.paper_ids[i]: float(scores[i]) for i in idx}

    def search(self, claim: str, k: int = 10, claim_id: str = "q") -> RankedList:
        if not claim or not claim.strip():
            raise ValueError("Empty claim")
        t0 = time.perf_counter()
        if self.method in ("bm25", "sbert"):
            retr = self.bm25 if self.method == "bm25" else self.sbert
            ids, scores = self._branch(retr, claim, k)
            results = [RetrievalResult(pid, r, scores[pid], self.method,
                                       {self.method: r}, {self.method: scores[pid]})
                       for r, pid in enumerate(ids, start=1)]
        else:
            b_ids, b_sc = self._branch(self.bm25, claim, self.fusion_depth)
            s_ids, s_sc = self._branch(self.sbert, claim, self.fusion_depth)
            b_rank = {pid: r for r, pid in enumerate(b_ids, 1)}
            s_rank = {pid: r for r, pid in enumerate(s_ids, 1)}
            fused = rrf_fuse({"bm25": b_ids, "sbert": s_ids}, self.k_rrf)[:k]
            results = [RetrievalResult(
                pid, r, sc, "rrf",
                {"bm25": b_rank.get(pid), "sbert": s_rank.get(pid)},
                {"bm25": b_sc.get(pid), "sbert": s_sc.get(pid)})
                for r, (pid, sc) in enumerate(fused, start=1)]
        latency = (time.perf_counter() - t0) * 1000
        return RankedList(claim_id, claim, self.method, results, latency, self.corpus,
                          self.settings())


def build_all(papers: list[Paper], sbert_model: str | None = None, cache_dir="retrieval/cache",
              encoder=None, corpus="scifact", **kw) -> dict[str, Retriever]:
    """Index BM25 and SBERT once and share them across the three methods,
    so the comparison uses identical indexes and SBERT encodes only once."""
    from research_verifier.retrieval.embeddings import DEFAULT_MODEL
    bm25 = BM25Retriever(**{k: v for k, v in kw.items() if k in ("k1", "b", "stemmer")})
    sbert = SBERTRetriever(sbert_model or DEFAULT_MODEL, cache_dir=cache_dir, encoder=encoder)
    rrf = Retriever("rrf", bm25=bm25, sbert=sbert, corpus=corpus,
                    **{k: v for k, v in kw.items() if k in ("k_rrf", "fusion_depth")}).index(papers)
    out = {}
    for m in ("bm25", "sbert"):
        r = Retriever(m, bm25=bm25 if m == "bm25" else None, sbert=sbert if m == "sbert" else None,
                      corpus=corpus)
        r.papers = rrf.papers
        r.index_seconds = {key: v for key, v in rrf.index_seconds.items() if key.startswith(m)}
        out[m] = r
    out["rrf"] = rrf
    return out
