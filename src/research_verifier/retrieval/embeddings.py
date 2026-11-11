"""SBERT dense retrieval: cosine similarity over normalised embeddings.

Corpus embeddings are cached to disk keyed by (model, corpus content hash), so
the 5k SciFact abstracts are encoded once and reused across experiments.

Model suggestions (set via --sbert-model):
  sentence-transformers/all-MiniLM-L6-v2        fast default, CPU-friendly
  sentence-transformers/multi-qa-mpnet-base-cos-v1  trained for query->passage
  sentence-transformers/allenai-specter         scientific papers (title+abstract)
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from research_verifier.retrieval.schemas import Paper

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _corpus_hash(papers: list[Paper]) -> str:
    h = hashlib.sha256()
    for p in papers:
        h.update(p.paper_id.encode())
        h.update(b"\0")
        h.update(p.index_text().encode())
        h.update(b"\1")
    return h.hexdigest()[:16]


class SBERTRetriever:
    name = "sbert"

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 64,
                 cache_dir: str | Path | None = "retrieval/cache", device: str | None = None,
                 encoder=None):
        """`encoder` lets tests inject a stand-in with .encode(list[str]) -> ndarray."""
        self.model_name, self.batch_size, self.device = model_name, batch_size, device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._encoder = encoder
        self.paper_ids: list[str] = []
        self.index_seconds: float | None = None
        self.loaded_from_cache = False

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_name, device=self.device)
        return self._encoder

    def settings(self) -> dict:
        s = {"method": "sbert", "model": self.model_name, "similarity": "cosine",
             "fields": "title+abstract", "batch_size": self.batch_size}
        try:
            import sentence_transformers
            s["sentence_transformers_version"] = sentence_transformers.__version__
        except ImportError:
            pass
        return s

    def _encode(self, texts: list[str]) -> np.ndarray:
        emb = self.encoder.encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=len(texts) > 500)
        emb = np.asarray(emb, dtype=np.float32)
        # Normalise again in case an injected encoder ignores the flag.
        return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)

    def index(self, papers: list[Paper], use_cache: bool = True) -> "SBERTRetriever":
        self.paper_ids = [p.paper_id for p in papers]
        key = f"{self.model_name.replace('/', '__')}_{_corpus_hash(papers)}"
        npy = self.cache_dir / f"{key}.npy" if self.cache_dir else None
        if use_cache and npy and npy.exists():
            self.embeddings = np.load(npy)
            self.loaded_from_cache = True
            self.index_seconds = json.loads(npy.with_suffix(".json").read_text())["encode_seconds"]
            return self
        t0 = time.perf_counter()
        self.embeddings = self._encode([p.index_text() for p in papers])
        self.index_seconds = time.perf_counter() - t0
        if npy and use_cache:
            npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy, self.embeddings)
            npy.with_suffix(".json").write_text(json.dumps(
                {**self.settings(), "n_papers": len(papers), "dim": int(self.embeddings.shape[1]),
                 "encode_seconds": self.index_seconds, "paper_ids": self.paper_ids}, indent=1))
        return self

    def score_all(self, query: str) -> np.ndarray:
        q = self._encode([query])[0]
        return self.embeddings @ q
