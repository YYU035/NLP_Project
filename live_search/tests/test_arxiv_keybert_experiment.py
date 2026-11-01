"""Offline arXiv experiment regressions; no HTTP or model downloads."""

import json

import pytest

from paper_search.query_rendering import render_query
from scripts.experiment_arxiv_keybert import (
    PREVIOUS,
    analyze,
    phrase_input,
    prepare,
)


def test_explicit_phrases_preserve_keybert_choices() -> None:
    phrases = ["lithium-ion batteries", "90% capacity", "-5 °C"]
    query = phrase_input(phrases)
    assert render_query(
        query, "arxiv", arxiv_strategy="literal"
    ).text == (
        'all:"lithium-ion batteries" AND all:"90% capacity" AND all:"-5 °C"'
    )


@pytest.mark.parametrize("phrases", [[], [""], ['a"b'], ["a\\b"]])
def test_ambiguous_phrase_input_is_rejected(phrases) -> None:
    with pytest.raises(ValueError):
        phrase_input(phrases)


def test_frozen_plan_matches_all_saved_spacy_expressions(
    tmp_path,
) -> None:
    if not (PREVIOUS / "results.json").exists():
        pytest.skip("Local experiment artifacts not distributed")
    manifest = prepare(tmp_path / "run")
    assert len(manifest["plans"]) == 10
    old = manifest["previous"]["baseline"]["rendering"]
    for case_id, plans in manifest["plans"].items():
        assert plans["spacy"]["rendered"] == old[case_id]["text"]
        for plan in plans.values():
            assert plan["wire_query"].endswith(
                "AND submittedDate:[202001010000 TO 202610302359]"
            )
    assert len(manifest["conflicting_label_keys"]) == 12


def test_empty_results_are_not_failures_or_extra_union_requests(
    tmp_path,
):
    plan = {
        "rendered": 'all:"test topic"',
        "wire_query": '(all:"test topic") AND '
        'submittedDate:[202001010000 TO 202610302359]'
    }
    previous = {
        "baseline": {
            "calls": [],
            "cases": [{"id": "c01", "claim": "original claim"}],
        },
        "prior_openaire_calls": [],
        "calls": [],
    }
    manifest = {
        "previous": previous,
        "labels": {},
        "conflicting_label_keys": [],
        "plans": {"c01": {"spacy": plan, "keybert": plan}},
        "calls": [],
    }
    for kind, status in [
        ("original", "zero_results"),
        ("alternative", "rate_limited"),
    ]:
        manifest["calls"].append(
            {
                "id": kind,
                "provider": "arxiv",
                "claim_id": "c01",
                "kind": kind,
                "query": '"test topic"',
                "authentication": "not_required",
                "status": status,
                "result_count": 0
                if status == "zero_results"
                else None,
                "papers": [],
                "attempts": [{"status": 429}],
            }
        )
    result = analyze(manifest, tmp_path)
    assert result["totals"]["union"]["unique_pairs"] == 0
    assert len(result["calls"]) == 2
    assert result["diagnostics"][0]["returned"] == 0
    assert result["diagnostics"][1]["returned"] is None
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "unavailable" in text
    assert not json.loads((tmp_path / "results.json").read_text())[
        "novelty"
    ]["new_pairs"]
