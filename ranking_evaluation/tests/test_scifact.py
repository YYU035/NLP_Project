"""Constructed SciFact-format fixtures; not sampled scientific claims."""

import json

import pytest
from pydantic import ValidationError

from ranking_evaluation.scifact import (
    RankingExport, convert_scifact, load_scifact, sha256,
)
from ranking_evaluation.evaluate import run_benchmark


@pytest.fixture
def files(tmp_path):
    corpus, claims = tmp_path / "corpus.jsonl", tmp_path / "claims_dev.jsonl"
    corpus.write_text("\n".join(json.dumps({
        "doc_id": index, "title": f"Fictional {index}",
        "abstract": ["Invented sentence."], "structured": False,
    }) for index in (1, 2, 3)))
    claims.write_text("\n".join(json.dumps(row) for row in [
        {"id": 10, "claim": "Fictional claim", "cited_doc_ids": [1, 2, 3],
         "evidence": {"1": [{"label": "SUPPORT", "sentences": [0]}],
                      "2": [{"label": "CONTRADICT", "sentences": [0]}]}},
        {"id": 20, "claim": "Fictional no evidence", "cited_doc_ids": [3], "evidence": {}},
    ]))
    return corpus, claims


def export(files, **changes):
    corpus, claims = files
    data = dict(status="complete", corpus_sha256=sha256(corpus),
                claims_sha256=sha256(claims), retrieval_config={"fixture": "manual"},
                records=[{"claim_id": "10", "bm25_ranking": ["3", "1"],
                          "sbert_ranking": ["1"]},
                         {"claim_id": "20", "bm25_ranking": [], "sbert_ranking": []}])
    data.update(changes)
    return RankingExport.model_validate(data)


def test_evidence_mapping_keeps_both_labels_and_unretrieved_gold(files):
    corpus, claims = files
    bundle = convert_scifact(corpus, claims, export(files), split="dev")
    assert bundle.qrels == {"10": ["1", "2"], "20": []}
    assert [row.identity for row in bundle.claims[0].candidates.papers] == ["3", "1"]
    assert bundle.claims[0].candidates.papers[1].paper.abstract == "Invented sentence."
    assert bundle.claims[0].candidates.papers[1].provenance == []
    report = run_benchmark(bundle, ks=(2,))
    assert report["metrics"]["rrf"]["macro"]["Recall@2"] == 0.5
    assert report["metrics"]["rrf"]["excluded_no_relevant"] == 1


@pytest.mark.parametrize("changes,match", [
    ({"status": "pending"}, "Pending"),
    ({"corpus_sha256": "wrong"}, "hashes"),
    ({"claims_sha256": "wrong"}, "hashes"),
    ({"records": []}, "each split claim"),
])
def test_invalid_export(files, changes, match):
    with pytest.raises(ValueError, match=match):
        convert_scifact(*files, export(files, **changes), split="dev")


def test_missing_labels_and_test_split_rejected(files):
    corpus, claims = files
    with pytest.raises(ValueError, match="train/dev"):
        load_scifact(corpus, claims, split="test")
    claims.write_text(json.dumps({"id": 10, "claim": "unlabeled", "cited_doc_ids": []}))
    with pytest.raises(ValidationError):
        load_scifact(corpus, claims, split="dev")


def test_invalid_evidence_and_duplicate_ids(files):
    corpus, claims = files
    rows = [json.loads(line) for line in claims.read_text().splitlines()]
    rows[0]["evidence"]["1"][0]["sentences"] = [5]
    claims.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="sentence index"):
        load_scifact(corpus, claims, split="dev")
    corpus.write_text(corpus.read_text() + "\n" + corpus.read_text().splitlines()[0])
    with pytest.raises(ValueError, match="Duplicate corpus"):
        load_scifact(corpus, claims, split="dev")


def test_unknown_and_duplicate_ranked_ids_rejected(files):
    for ids in (["unknown"], ["1", "1"]):
        rankings = export(files)
        rankings.records[0].bm25_ranking = ids
        with pytest.raises(ValueError):
            convert_scifact(*files, rankings, split="dev")
