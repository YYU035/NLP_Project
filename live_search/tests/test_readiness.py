"""Offline depth semantics and reproducible manual-review preparation."""

import csv
import hashlib
import importlib.util
from pathlib import Path

import pytest

from paper_search.evaluation.depth import prefix_summary
from paper_search.metadata import ScientificPaper


def test_prefixes_deduplicate_without_counting_unknown_as_irrelevant():
    papers = [
        ScientificPaper(source="openalex", source_id=str(i)) for i in range(29)
    ]
    papers.insert(1, papers[0])

    def label(paper):
        return {"0": 2, "1": 1, "2": 0}.get(paper.source_id)

    ten = prefix_summary(papers, 10, label)
    assert ten["returned"] == 10
    assert ten["unique"] == 9
    assert ten["relevant_known"] == 2
    assert ten["evidence_known"] == 1
    assert ten["unknown"] == 6
    assert prefix_summary(papers, 30, label)["unique"] == 29
    with pytest.raises(ValueError):
        prefix_summary(papers, 15, label)


def test_saved_review_exports_are_blind_stratified_and_immutable(
    tmp_path: Path,
):
    spec = importlib.util.spec_from_file_location(
        "readiness_prep", "scripts/prepare_integration_readiness.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "review"
    module.prepare(root)
    with (root / "manual_validation_47.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        reviews = list(csv.DictReader(stream))
    with (root / "manual_validation_audit.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        audit = list(csv.DictReader(stream))
    assert len(reviews) == len(audit) == 47
    assert len({r["review_id"] for r in reviews}) == 47
    assert len({r["claim_id"] for r in audit}) == 10
    assert not {
        "provider",
        "providers",
        "labels",
        "label",
        "stratum",
        "rank",
    } & set(reviews[0])
    assert all(
        not r[field]
        for r in reviews
        for field in (
            "reviewer_1_label",
            "reviewer_2_label",
            "adjudicated_label",
        )
    )
    # An accidental rerun cannot replace a completed review.
    before = hashlib.sha256(
        (root / "manual_validation_47.csv").read_bytes()
    ).hexdigest()
    with pytest.raises(FileExistsError):
        module.prepare(root)
    assert (
        hashlib.sha256(
            (root / "manual_validation_47.csv").read_bytes()
        ).hexdigest()
        == before
    )
