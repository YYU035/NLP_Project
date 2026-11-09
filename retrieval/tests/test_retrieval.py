"""Run: python -m pytest retrieval/tests -q   (from the repo root)

Uses a tiny SciFact-format corpus and a deterministic hashing encoder, so the
tests need no downloads and no GPU. Real SBERT is exercised by run_scifact.
"""
import hashlib
import json
import math

import numpy as np
import pytest

from retrieval.bm25 import BM25Retriever
from retrieval.data import load_live_papers, load_scifact_claims, load_scifact_corpus
from retrieval.evaluate import evaluate_method, mrr, ndcg_at_k, recall_at_k
from retrieval.retriever import Retriever, build_all, rrf_fuse
from retrieval.run_scifact import main as run_scifact
from retrieval.schema import Paper
from retrieval.search_live import rank_candidates


class HashEncoder:
    """Bag-of-hashed-words embedding: a stand-in for SBERT in tests."""
    def __init__(self, dim=256):
        self.dim = dim
        self.calls = 0

    def encode(self, texts, **kw):
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, int(hashlib.md5(w.strip(".,").encode()).hexdigest(), 16) % self.dim] += 1
        return out


CORPUS = [
    {"doc_id": 1, "title": "Vitamin D and fracture risk",
     "abstract": ["Vitamin D supplementation reduced hip fractures in elderly adults.",
                  "The effect was dose dependent."]},
    {"doc_id": 2, "title": "Statins and cardiovascular outcomes",
     "abstract": ["Statin therapy lowered LDL cholesterol.", "Cardiovascular events decreased."]},
    {"doc_id": 3, "title": "Coffee consumption and heart disease",
     "abstract": ["Moderate coffee intake was not associated with coronary heart disease."]},
    {"doc_id": 4, "title": "Exercise and depression",
     "abstract": ["Aerobic exercise improved depressive symptoms in adults."]},
    {"doc_id": 5, "title": "Calcium intake in older women",
     "abstract": ["Calcium did not reduce fracture incidence in older women."]},
]
CLAIMS = [
    {"id": 10, "claim": "Vitamin D supplements prevent hip fractures.",
     "evidence": {"1": [{"sentences": [0], "label": "SUPPORT"}]}, "cited_doc_ids": [1]},
    {"id": 11, "claim": "Drinking coffee causes heart disease.",
     "evidence": {"3": [{"sentences": [0], "label": "CONTRADICT"}]}, "cited_doc_ids": [3]},
    {"id": 12, "claim": "Fracture rates in older women fall with calcium or vitamin D.",
     "evidence": {"1": [], "5": []}, "cited_doc_ids": [1, 5]},
    {"id": 13, "claim": "Unrelated NEI claim.", "evidence": {}, "cited_doc_ids": [2]},
]


@pytest.fixture
def scifact_dir(tmp_path):
    d = tmp_path / "scifact"
    d.mkdir()
    (d / "corpus.jsonl").write_text("\n".join(json.dumps(r) for r in CORPUS))
    (d / "claims_dev.jsonl").write_text("\n".join(json.dumps(r) for r in CLAIMS))
    return d


@pytest.fixture
def papers(scifact_dir):
    return load_scifact_corpus(scifact_dir / "corpus.jsonl")


# ---------------------------------------------------------------- data
def test_loaders_keep_sentences_and_drop_nei(scifact_dir, papers):
    assert papers[0].paper_id == "1" and len(papers[0].abstract) == 2
    claims = load_scifact_claims(scifact_dir / "claims_dev.jsonl")
    assert [c["claim_id"] for c in claims] == ["10", "11", "12"]
    assert claims[2]["relevant"] == {"1", "5"}


def test_live_adapter_handles_varied_fields_and_dupes():
    recs = [
        {"id": "arxiv:1", "title": "A", "abstract": "First sentence. Second one here."},
        {"id": "arxiv:1", "title": "A dup", "abstract": "x"},
        {"doi": "10.1/x", "title": "B", "summary": "Only abstract.", "publication_date": "2021-05-01"},
        {"paperId": "s2:9", "title": "C"},
    ]
    ps = load_live_papers(recs)
    assert [p.paper_id for p in ps] == ["arxiv:1", "10.1/x", "s2:9"]
    assert ps[0].abstract == ["First sentence.", "Second one here."]
    assert ps[1].year == 2021 and ps[2].text_status == "unavailable"


# ---------------------------------------------------------------- metrics
def test_metrics_hand_computed():
    ranked, rel = ["a", "b", "c", "d"], {"b", "d"}
    assert recall_at_k(ranked, rel, 2) == 0.5
    assert mrr(ranked, rel) == 0.5
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(ranked, rel, 4) == pytest.approx(expected)
    assert ndcg_at_k(["b", "d"], rel, 2) == pytest.approx(1.0)
    assert mrr(["x"], rel) == 0.0


def test_rrf_formula_and_missing_docs():
    fused = dict(rrf_fuse({"bm25": ["a", "b"], "sbert": ["b", "c"]}, k_rrf=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert max(fused, key=fused.get) == "b"


# ---------------------------------------------------------------- retrieval
def test_bm25_ranks_lexical_match_first(papers):
    r = Retriever("bm25").index(papers)
    out = r.search("vitamin D hip fractures", k=3)
    assert out.results[0].paper_id == "1"
    assert [x.rank for x in out.results] == [1, 2, 3]
    assert out.results[0].score >= out.results[1].score


def test_bm25_unknown_terms_give_zero_scores(papers):
    s = BM25Retriever().index(papers).score_all("zzzz qqqq")
    assert np.all(s == 0)


def test_all_methods_share_interface(papers):
    rs = build_all(papers, encoder=HashEncoder(), cache_dir=None)
    for m, r in rs.items():
        out = r.search("coffee heart disease", k=5, claim_id="c1")
        assert out.method == m and out.claim_id == "c1" and len(out.results) == 5
        assert len({x.paper_id for x in out.results}) == 5
        assert out.latency_ms >= 0
    rrf = rs["rrf"].search("coffee heart disease", k=5)
    assert set(rrf.results[0].component_ranks) == {"bm25", "sbert"}


def test_sbert_cache_reused(papers, tmp_path):
    enc = HashEncoder()
    from retrieval.sbert import SBERTRetriever
    SBERTRetriever("test-model", cache_dir=tmp_path, encoder=enc).index(papers)
    first_calls = enc.calls
    s2 = SBERTRetriever("test-model", cache_dir=tmp_path, encoder=enc).index(papers)
    assert s2.loaded_from_cache and enc.calls == first_calls
    assert (tmp_path / next(p.name for p in tmp_path.glob("*.json"))).exists()


def test_rejects_bad_input(papers):
    with pytest.raises(ValueError):
        Retriever("bm25").index(papers + [papers[0]])
    with pytest.raises(ValueError):
        Retriever("bm25").index(papers).search("   ")


def test_k_larger_than_corpus(papers):
    out = Retriever("bm25").index(papers).search("fracture", k=50)
    assert len(out.results) == len(papers)


# ---------------------------------------------------------------- end to end
def test_run_scifact_end_to_end(scifact_dir, tmp_path):
    out = tmp_path / "out"
    res = run_scifact(["--data-dir", str(scifact_dir), "--split", "dev", "--k", "5",
                       "--cache-dir", str(tmp_path / "cache"), "--out-dir", str(out),
                       "--sbert-model", "test-model"], encoder=HashEncoder())
    assert set(res) == {"bm25", "sbert", "rrf"}
    assert res["bm25"]["summary"]["n_claims"] == 3
    for f in ("results.md", "metrics.json", "failures.json", "config.json",
              "runs_bm25.jsonl", "runs_sbert.jsonl", "runs_rrf.jsonl"):
        assert (out / f).exists(), f
    handoff = [json.loads(l) for l in (out / "runs_rrf.jsonl").read_text().splitlines()]
    first = handoff[0]["results"][0]
    assert first["paper"]["abstract"] and "component_ranks" in first


def test_rank_live_candidates():
    recs = [{"id": f"p{i}", "title": c["title"], "abstract": " ".join(c["abstract"])}
            for i, c in enumerate(CORPUS)] + [{"id": "empty", "title": ""}]
    ranked, papers = rank_candidates("vitamin D fractures", recs, "rrf", k=3,
                                     encoder=HashEncoder())
    assert ranked.results[0].paper_id == "p0"
    assert ranked.settings["skipped_no_text"] == ["empty"]
    d = ranked.to_dict(papers)
    assert d["corpus"] == "live" and d["results"][0]["paper"]["title"]


def test_evaluate_method_summary(papers):
    r = Retriever("bm25").index(papers)
    runs = [r.search("vitamin D fractures", 5, "a")]
    res = evaluate_method(runs, {"a": {"1"}})
    assert res["summary"]["mrr"] == 1.0 and res["summary"]["recall@1"] == 1.0
