"""Verify corpus/gold alignment, runner comparison, and CLI provenance."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from ranking_evaluation.benchmark import BenchmarkBundle
from ranking_evaluation.evaluate import run_benchmark
from ranking_evaluation.examples.write_mock_benchmark import build_mock_benchmark


def test_runner_uses_full_gold_denominator_and_common_claims():
    report = run_benchmark(build_mock_benchmark(), ks=(1, 3))
    assert set(report["metrics"]) == {"bm25", "sbert", "rrf"}
    for method in report["metrics"].values():
        assert method["macro"]["Recall@3"] == 0.5
        assert method["total_claims"] == 1
    assert report["metrics"]["bm25"]["macro"]["MRR"] == pytest.approx(1 / 3)
    assert report["metrics"]["sbert"]["macro"]["MRR"] == 1
    assert report["metrics"]["rrf"]["macro"]["MRR"] == 0.5
    assert report["latency"]["mock-claim-1"]["bm25_seconds"] is None
    assert report["latency"]["mock-claim-1"]["end_to_end_seconds"] is None


def test_optional_reranking_comparison():
    class Scorer:
        def score_pairs(self, pairs):
            return [0, 1, -1]
    report = run_benchmark(build_mock_benchmark(), scorer=Scorer())
    assert report["metrics"]["rrf_cross_encoder"]["macro"]["MRR"] == 1
    assert report["latency"]["mock-claim-1"]["rerank_seconds"] >= 0


@pytest.mark.parametrize("change", ["duplicate_claim", "missing_qrel", "unknown_gold",
                                     "unknown_candidate", "duplicate_corpus", "duplicate_gold"])
def test_bad_bundle_ids_rejected(change):
    data = build_mock_benchmark().model_dump()
    if change == "duplicate_claim":
        data["claims"].append(data["claims"][0])
    elif change == "missing_qrel":
        data["qrels"] = {}
    elif change == "unknown_gold":
        data["qrels"]["mock-claim-1"] = ["unknown"]
    elif change == "unknown_candidate":
        data["corpus_doc_ids"].remove("synthetic:A")
    elif change == "duplicate_corpus":
        data["corpus_doc_ids"].append("synthetic:A")
    else:
        data["qrels"]["mock-claim-1"] = ["synthetic:B", "synthetic:B"]
    with pytest.raises(ValidationError):
        BenchmarkBundle.model_validate(data)


def test_empty_branch_is_kept_in_comparison():
    bundle = build_mock_benchmark()
    bundle.claims[0].bm25_ranking = []
    report = run_benchmark(bundle)
    assert report["metrics"]["bm25"]["macro"]["MRR"] == 0
    assert report["metrics"]["sbert"]["macro"]["MRR"] == 1


def test_cli_writes_provenance_and_refuses_overwrite(tmp_path):
    source, output = tmp_path / "input.json", tmp_path / "report.json"
    source.write_text(build_mock_benchmark().model_dump_json())
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root / "live_search/src"))
    cmd = [sys.executable, "-m", "ranking_evaluation.evaluate",
           "--input", str(source), "--output", str(output), "--ks", "1", "3"]
    first = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    report = json.loads(output.read_text())
    assert len(report["provenance"]["input_sha256"]) == 64
    assert "ranking_evaluation/rrf.py" in report["provenance"]["source_sha256"]
    assert report["dataset"]["synthetic"] is True
    before = output.read_bytes()
    second = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
    assert second.returncode != 0
    assert "already exists" in second.stderr
    assert output.read_bytes() == before
