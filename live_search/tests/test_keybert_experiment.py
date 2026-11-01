"""Offline report tests: reuse only known labels, never score new papers."""

import csv

import pytest

from paper_search.metadata import ScientificPaper
from scripts import experiment_keybert as experiment


def test_report_reuses_labels_by_identity_and_separates_new_pairs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(experiment, "OLD", tmp_path)
    with (tmp_path / "manual_assessment_provisional_250.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["claim_id", "identity", "manual_relevance"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "claim_id": "c01",
                "identity": "0",
                "manual_relevance": "2.0",
            }
        )
    old = ScientificPaper(
        source="openalex", source_id="old", doi="10.123/old"
    )
    spacy_new = ScientificPaper(
        source="openalex", source_id="new-spacy"
    )
    kb_new = ScientificPaper(source="openalex", source_id="new-kb")

    def call(kind, papers):
        return {
            "id": kind,
            "provider": "openalex",
            "claim_id": "c01",
            "kind": kind,
            "query": "query",
            "status": "success",
            "authentication": "anonymous",
            "result_count": len(papers),
            "attempts": [
                {"status": 200, "response_file": "missing.body"}
            ],
            "papers": [p.model_dump(mode="json") for p in papers],
        }

    manifest = {
        "baseline": {
            "cases": [{"id": "c01", "claim": "original claim"}],
            "calls": [call("original", [old])],
        },
        "prior_openaire_calls": [],
        "calls": [
            call("original", [old, spacy_new]),
            call("alternative", [old, kb_new]),
        ],
    }
    result = experiment.report(manifest, tmp_path)
    assert result["historical_pairs"] == 1
    assert result["new_pairs"] == 2
    assert result["new_keybert_pairs"] == 1
    summary = next(
        s
        for s in result["summaries"]
        if s["claim_id"] == "ALL"
        and s["provider"] == "openalex"
        and s["strategy"] == "keybert"
    )
    assert summary["known_labels"] == summary["known_evidence"] == 1
    assert summary["unlabelled"] == 1
    comparison = next(
        c
        for c in result["comparisons"]
        if c["claim_id"] == "ALL" and c["provider"] == "openalex"
    )
    assert comparison["overlap"] == 1
    with (tmp_path / "new_pairs_for_assessment.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert all(
        row["reused_label"] == row["manual_relevance"] == ""
        for row in rows
    )
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "not attempted" in text


def test_report_refuses_changed_historical_label_source(tmp_path):
    source = tmp_path / "labels.csv"
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(
        ValueError, match="Historical source changed"
    ):
        experiment.report(
            {"historical_sources": {str(source): "wrong-hash"}},
            tmp_path,
        )
