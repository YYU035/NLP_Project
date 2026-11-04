"""Batch replay, baseline ablations and preserved run artifacts."""

import json

import pytest
from pydantic import ValidationError

from ranking_evaluation.examples.write_mock_benchmark import build_mock_benchmark
from ranking_evaluation.experiment_suite import SuiteConfig, run_suite, summarize_report
from ranking_evaluation.evaluate import run_benchmark


@pytest.mark.parametrize("data", [
    {"ks": []}, {"ks": [1, 1]}, {"rrf_constants": [-1]},
    {"rrf_constants": [True]}, {"rerank_depths": [0]}, {"repeats": 0},
])
def test_config_rejects_invalid_grids(data):
    with pytest.raises(ValidationError):
        SuiteConfig.model_validate(data)


def test_suite_replays_same_input_and_deduplicates_baselines(tmp_path):
    source = tmp_path / "input.json"
    source.write_text(build_mock_benchmark().model_dump_json())
    output = tmp_path / "runs"
    config = SuiteConfig(ks=[1, 3], rrf_constants=[0, 60], repeats=2)
    manifest = run_suite(source, config, output)
    assert manifest["status"] == "complete"
    assert manifest["synthetic"] is True
    assert len(manifest["runs"]) == 4
    rows = json.loads((output / "summary.json").read_text())
    assert len(rows) == 8  # 2 baselines + 2 RRF settings, for each repeat.
    assert sum(row["method"] == "bm25" for row in rows) == 2
    assert all(row["latency_observations"] == 0 for row in rows if row["method"] == "bm25")
    hashes = {json.loads((output / run["report"]).read_text())["provenance"]["input_sha256"]
              for run in manifest["runs"]}
    assert hashes == {manifest["input_sha256"]}
    with pytest.raises(FileExistsError):
        run_suite(source, config, output)


def test_rerank_grid_needs_model_before_writing(tmp_path):
    source = tmp_path / "input.json"
    source.write_text(build_mock_benchmark().model_dump_json())
    with pytest.raises(ValueError, match="model options"):
        run_suite(source, SuiteConfig(rerank_depths=[10]), tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_summary_measures_only_requested_stage():
    report = run_benchmark(build_mock_benchmark())
    report["latency"] = {
        "a": {"rrf_seconds": 1}, "b": {"rrf_seconds": 3},
    }
    row = summarize_report(report, ["rrf"], run="test", repeat=1)[0]
    assert row["latency_mean_seconds"] == 2
    assert row["latency_median_seconds"] == 2
    assert row["latency_p95_seconds"] == 3


def test_failure_retains_explicit_manifest_and_log(tmp_path, monkeypatch):
    from types import SimpleNamespace

    source = tmp_path / "input.json"
    source.write_text(build_mock_benchmark().model_dump_json())
    monkeypatch.setattr("ranking_evaluation.experiment_suite.subprocess.run",
                        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="failed"))
    output = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="failed"):
        run_suite(source, SuiteConfig(), output)
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    assert list(output.glob("*.log"))[0].read_text() == "failed"
