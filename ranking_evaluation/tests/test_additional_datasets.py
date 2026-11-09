"""Additional dataset boundaries, reference arithmetic and leakage controls."""

import copy
import json
import os
import subprocess
import sys
from io import BytesIO
from math import log2
from pathlib import Path

import pytest

from shared.datasets import csfcube, matter_of_fact
from shared.datasets.setup import SOURCES, download, file_hash, verify_download
from ranking_evaluation.evaluate_csfcube import evaluate as evaluate_cs, graded_ndcg
from ranking_evaluation.evaluate_matter_of_fact import evaluate as evaluate_mof


def write_release(directory, dataset, contents):
    directory.mkdir()
    for name in SOURCES[dataset]["files"]:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents.get(name, "fixture"))
    source = SOURCES[dataset]
    manifest = {"dataset": dataset, "repository": source["repository"],
                "revision": source["revision"], "files_sha256": {
                    name: file_hash(directory / name) for name in source["files"]}}
    (directory / "download_manifest.json").write_text(json.dumps(manifest))


@pytest.fixture
def cs_release(tmp_path):
    directory = tmp_path / "csfcube"
    papers = [{"paper_id": pid, "title": f"Research {pid}", "abstract": [f"Text {pid}."]}
              for pid in ("q", "a", "b", "c")]
    content = {"abstracts-csfcube-preds.jsonl": "\n".join(json.dumps(p) for p in papers)}
    splits = {}
    for facet in csfcube.FACETS:
        content[f"test-pid2anns-csfcube-{facet}.json"] = json.dumps({
            "q": {"cands": ["q", "a", "b", "c"], "relevance_adju": [3, 0, 1, 3]}})
        splits[facet] = {split: [f"q_{facet}"] for split in csfcube.SPLITS if split != "all"}
    splits["all"] = {split: [f"q_{facet}" for facet in csfcube.FACETS]
                     for split in csfcube.SPLITS if split != "all"}
    content["evaluation_splits.json"] = json.dumps(splits)
    write_release(directory, "csfcube", content)
    return directory


@pytest.fixture
def mof_release(tmp_path):
    directory = tmp_path / "matter_of_fact"
    rows = [{"claim_id": f"paper_{i}_{'T' if label else 'F'}", "claim_text": f"Claim {i}",
             "gold_label": label, "metadata": {"paper_id": f"paper{i}",
             "topics": [theme], "exclude_date": {"year": 2023, "month": 2},
             "explanation": "Secret gold explanation", "supporting_facts_from_paper": ["Gold facts"]}}
            for i, (label, theme) in enumerate([(True, "theme_batteries"),
                (False, "theme_semi-conductors"), (True, "theme_aerospace materials")])]
    content = {f"benchmark/{name}": json.dumps(rows) for name in matter_of_fact.FILES.values()}
    write_release(directory, "matter_of_fact", content)
    return directory


def test_download_uses_pinned_urls_and_reuses_verified_files(tmp_path, monkeypatch):
    urls = []
    def fetch(request, timeout):
        urls.append(request.full_url)
        return BytesIO(b"fixture content")
    monkeypatch.setattr("urllib.request.urlopen", fetch)
    directory = tmp_path / "dataset"
    first = download("csfcube", directory)
    assert len(urls) == len(SOURCES["csfcube"]["files"])
    assert all(SOURCES["csfcube"]["revision"] in url for url in urls)
    assert download("csfcube", directory) == first
    assert len(urls) == len(SOURCES["csfcube"]["files"])
    (directory / "README.md").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_download(directory, "csfcube")


def test_failed_download_leaves_no_partial_dataset(tmp_path, monkeypatch):
    def fail(request, timeout):
        raise OSError("synthetic transport failure")
    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(OSError):
        download("csfcube", tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()
    assert list(tmp_path.iterdir()) == []


def test_cs_preserves_grades_and_excludes_self_match(cs_release, tmp_path):
    dataset = csfcube.load(cs_release)
    assert len(dataset["queries"]) == 3
    assert dataset["queries"]["q_method"]["candidate_ids"] == ["a", "b", "c"]
    assert dataset["grades"]["q_method"] == {"a": 0, "b": 1, "c": 3}
    output = tmp_path / "prepared"
    csfcube.prepare(dataset, output)
    inputs = json.loads((output / "retrieval-input.json").read_text())
    assert "grades" not in inputs
    assert all("relevance_adju" not in row for row in inputs["queries"].values())
    assert all("relevance_adju" not in row for row in inputs["documents"].values())
    with pytest.raises(FileExistsError):
        csfcube.prepare(dataset, output)


def test_cs_evaluation_retains_full_ideal_and_threshold(cs_release):
    dataset = csfcube.load(cs_release)
    export = {"status": "complete", "dataset_version": dataset["version"],
              "method": "fixture", "rankings": {q: ["b", "c"] for q in dataset["queries"]}}
    report = evaluate_cs(dataset, export, ks=(1, 2))
    assert report["binary_metrics"]["macro"]["Recall@1"] == 0
    assert report["binary_metrics"]["macro"]["MRR"] == 0.5
    assert report["graded_macro"]["graded_nDCG@1"] == pytest.approx(1 / 3)
    assert report["graded_macro"]["graded_nDCG@2"] == pytest.approx(
        (1 + 3 / log2(3)) / (3 + 1 / log2(3)))


def test_graded_ndcg_against_independent_library():
    sklearn = pytest.importorskip("sklearn.metrics")
    expected = sklearn.ndcg_score([[0, 1, 3]], [[3, 2, 1]], k=2)
    assert graded_ndcg(["a", "b", "c"], {"a": 0, "b": 1, "c": 3}, 2) == pytest.approx(expected)
    assert graded_ndcg([], {"a": 3}, 1) == 0
    assert graded_ndcg([], {"a": 0}, 1) is None


@pytest.mark.parametrize("bad", [["a", "a"], ["unknown"], ["q"]])
def test_cs_invalid_pool_rankings_rejected(cs_release, bad):
    dataset = csfcube.load(cs_release)
    export = {"status": "complete", "dataset_version": dataset["version"],
              "method": "fixture", "rankings": {q: bad for q in dataset["queries"]}}
    with pytest.raises(ValueError):
        evaluate_cs(dataset, export)


def test_cs_pending_and_wrong_versions_rejected(cs_release):
    dataset = csfcube.load(cs_release)
    for export in ({"status": "pending"}, {"status": "complete", "dataset_version": "wrong"}):
        with pytest.raises(ValueError, match="completed export"):
            evaluate_cs(dataset, export)


def test_real_bm25_adapter_runs_without_judgments(cs_release):
    pytest.importorskip("numpy")
    dataset = csfcube.load(cs_release)
    dataset.pop("grades")
    result = csfcube.bm25_rankings(dataset)
    assert result["status"] == "complete"
    assert all(set(ranking) == {"a", "b", "c"} for ranking in result["rankings"].values())


def test_mof_eee_filter_and_no_gold_in_inference(mof_release, tmp_path):
    dataset = matter_of_fact.load(mof_release)
    assert dataset["source_split_count"] == 3
    assert len(dataset["inputs"]) == 2
    assert all(set(row) == {"case_id", "claim_text", "evidence_not_after"} for row in dataset["inputs"])
    assert all(row["evidence_not_after"] == "2023-02-28" for row in dataset["inputs"])
    assert all(not row["case_id"].endswith(("_T", "_F")) for row in dataset["inputs"])
    output = tmp_path / "prepared"
    matter_of_fact.prepare(dataset, output)
    text = (output / "inference-input.json").read_text()
    assert "Secret gold" not in text and "paper_0_T" not in text and "Gold facts" not in text
    assert len(matter_of_fact.load(mof_release, scope="all")["inputs"]) == 3


def predictions(dataset, values):
    return {"status": "complete", "dataset_version": dataset["version"], "model": "fixture model",
            "predictions": [{"case_id": row["case_id"], "prediction": value, "evidence": []}
                            for row, value in zip(dataset["inputs"], values)]}


def test_mof_metrics_and_abstention_accounting(mof_release):
    dataset = matter_of_fact.load(mof_release)
    report = evaluate_mof(dataset, predictions(dataset, [True, False]))
    assert report["accuracy"] == report["macro_f1"] == 1
    assert report["cases_without_evidence"] == 2
    assert report["formal_temporal_protocol_verified"] is False
    report = evaluate_mof(dataset, predictions(dataset, [None, False]))
    assert report["accuracy"] == report["macro_f1"] == 0.5


@pytest.mark.parametrize("bad", [1, "SUPPORTED", "infeasible"])
def test_mof_rejects_nonboolean_task_labels(mof_release, bad):
    dataset = matter_of_fact.load(mof_release)
    with pytest.raises(ValueError, match="boolean"):
        evaluate_mof(dataset, predictions(dataset, [bad, False]))


def test_mof_date_leakage_and_duplicate_predictions_rejected(mof_release):
    dataset = matter_of_fact.load(mof_release)
    export = predictions(dataset, [True, False])
    export["predictions"][0]["evidence"] = [{"reference": "paper", "published_date": "2023-03-01"}]
    with pytest.raises(ValueError, match="postdates"):
        evaluate_mof(dataset, export)
    export["predictions"][0]["evidence"][0]["published_date"] = "2023-02-28"
    assert evaluate_mof(dataset, export)["accuracy"] == 1
    export["predictions"].append(copy.deepcopy(export["predictions"][0]))
    with pytest.raises(ValueError, match="exactly once"):
        evaluate_mof(dataset, export)


def test_mof_pending_and_incomplete_predictions_rejected(mof_release):
    dataset = matter_of_fact.load(mof_release)
    export = predictions(dataset, [True])
    with pytest.raises(ValueError, match="all selected"):
        evaluate_mof(dataset, export)
    export["status"] = "pending"
    with pytest.raises(ValueError, match="completed"):
        evaluate_mof(dataset, export)


@pytest.mark.parametrize("dataset_name", ["csfcube", "matter_of_fact"])
def test_evaluator_cli_and_overwrite_guard(dataset_name, cs_release, mof_release, tmp_path):
    root = Path(__file__).resolve().parents[2]
    output, source = tmp_path / "report.json", tmp_path / "predictions.json"
    if dataset_name == "csfcube":
        dataset = csfcube.load(cs_release)
        export = {"status": "complete", "dataset_version": dataset["version"], "method": "fixture",
                  "rankings": {q: ["c", "b", "a"] for q in dataset["queries"]}}
        command = [sys.executable, "-m", "ranking_evaluation.evaluate_csfcube",
                   "--data", str(cs_release), "--rankings", str(source)]
    else:
        dataset = matter_of_fact.load(mof_release)
        export = predictions(dataset, [True, False])
        command = [sys.executable, "-m", "ranking_evaluation.evaluate_matter_of_fact",
                   "--data", str(mof_release), "--predictions", str(source)]
    source.write_text(json.dumps(export))
    command += ["--output", str(output)]
    environment = dict(os.environ, PYTHONPATH=str(root / "live_search/src"))
    result = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert "provenance" in report
    before = output.read_bytes()
    result = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True)
    assert result.returncode != 0 and "Output exists" in result.stderr
    assert output.read_bytes() == before


def test_cs_full_comparison_preserves_pools_and_uses_actual_module_boundaries(cs_release):
    np = pytest.importorskip("numpy")
    from ranking_evaluation.run_csfcube import run
    from ranking_evaluation.pipeline import ModelDescription

    class Encoder:
        def encode(self, texts, **kwargs):
            return np.asarray([[1, len(t), sum(ord(c) for c in t) % 17] for t in texts], dtype=float)
    class Scorer:
        def score_pairs(self, pairs):
            return [float(i) for i in range(len(pairs))]
    dataset = csfcube.load(cs_release)
    description = ModelDescription(model_id="fixture", revision="fixture", device="cpu",
                                    dtype="float32", max_length=512, batch_size=2)
    report = run(dataset, encoder=Encoder(), model_name="fixture", scorer=Scorer(),
                 model_description=description, top_n=2)
    assert set(report["methods"]) == {"bm25", "sbert", "rrf", "rrf_cross_encoder"}
    for rankings in report["rankings"].values():
        assert all(set(ids) == {"a", "b", "c"} for ids in rankings.values())
    assert all(row["rerank_scored_count"] == 2 for row in report["per_query"].values())
    assert report["query_count"] == 3


def test_cs_comparison_rejects_nonfinite_model_scores():
    pytest.importorskip("numpy")
    from ranking_evaluation.run_csfcube import ordered_ids
    with pytest.raises(ValueError, match="finite"):
        ordered_ids([float("nan")], ["a"])


def test_matter_discovery_passes_cutoff_and_excludes_unknown_future_metadata():
    import asyncio
    from datetime import date
    from paper_search.metadata import ScientificPaper
    from paper_search.retrieval_models import RetrievedPaper, RetrievalResult
    from shared.datasets.matter_discovery import discover_case

    claim = "Original research claim"
    entries = []
    for identity, published in (("old", date(2020, 1, 1)),
                                ("future", date(2024, 1, 1)), ("unknown", None)):
        paper = ScientificPaper(title=identity, source="synthetic", source_id=identity,
                                publication_date=published)
        entries.append(RetrievedPaper(identity=identity, identity_aliases=[identity],
            original_claim=claim, paper=paper, source_records=[paper], provenance=[]))
    upstream = RetrievalResult(original_claim=claim, top_k=30, papers=entries)
    calls = []
    class Service:
        async def retrieve(self, text, **kwargs):
            calls.append((text, kwargs))
            return upstream
    result = asyncio.run(discover_case({"case_id": "opaque", "claim_text": claim,
                                        "evidence_not_after": "2022-12-31"}, Service()))
    assert calls == [(claim, {"top_k": 30, "to_publication_date": date(2022, 12, 31)})]
    assert [entry["identity"] for entry in result["candidates"]["papers"]] == ["old"]
    assert {row["reason"] for row in result["audit"]["rejected"]} == {
        "post_cutoff_record", "unknown_publication_date"}
    assert len(upstream.papers) == 3
    with pytest.raises(ValueError, match="leakage-safe"):
        asyncio.run(discover_case({"gold_label": True}, Service()))
