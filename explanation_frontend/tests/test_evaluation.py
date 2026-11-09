import csv
import json
from pathlib import Path

import pytest

from explanation_frontend.evaluation.run import run_study, summarize_reviews


def test_paired_fixture_study_blinds_outputs_and_counts_invalid_generations(tmp_path):
    manifest = [{"case_id": "stress-1", "fixture": "qualified", "dataset": "synthetic", "fault": "numbers"},
                {"case_id": "stress-2", "fixture": "conflict", "dataset": "synthetic"}]
    result = run_study(manifest, mode="fixture", seed=7)
    assert result["summary"]["cases"] == 2
    assert result["summary"]["llm_statuses"] == {"rejected": 1, "accepted": 1}
    assert len(result["review_packets"]) == 4
    assert all("method" not in packet for packet in result["review_packets"])
    assert any("20%" in packet["raw_output"] for packet in result["review_packets"])
    assert len({packet["output_id"] for packet in result["review_packets"]}) == 4
    assert all(packet["claim"] for packet in result["review_packets"])


def test_human_scores_are_not_invented_and_denominators_are_explicit():
    assert summarize_reviews([])["completed_reviews"] == 0
    row = dict(output_id="o1", reviewer="r1", factual_supported="1", factual_total="2",
               citation_supported="1", citation_total="2", citation_complete="1", citation_required="2",
               verdict_consistent="1", scope_preserved="0", correction_supported="0", correction_total="0", usefulness="3")
    summary = summarize_reviews([row])
    assert summary["factual_support"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert summary["usefulness_mean"] == 3
    bad = dict(row, factual_supported="3")
    with pytest.raises(ValueError):
        summarize_reviews([bad])


def test_unknown_manifest_cases_fail_instead_of_becoming_demo_success():
    with pytest.raises(ValueError):
        run_study([{"case_id": "bad", "fixture": "not-found", "dataset": "synthetic"}], mode="fixture", seed=7)


def test_failed_raw_generation_is_not_reported_as_valid_baseline():
    result = run_study([{"case_id": "bad-json", "fixture": "supported", "dataset": "synthetic", "fault": "invalid"}], mode="fixture")
    generated = next(r for r in result["records"] if r["method"] == "llm")
    assert generated["generation_validation"] is None
    assert generated["display_validation"]["displayable"]


def test_review_summary_compares_methods_without_inventing_missing_reviews():
    row = dict(output_id="o1", reviewer="r1", factual_supported="1", factual_total="2",
               citation_supported="1", citation_total="2", citation_complete="1", citation_required="2",
               verdict_consistent="1", scope_preserved="0", correction_supported="0", correction_total="0", usefulness="3")
    key = [{"output_id": "o1", "method": "template"}, {"output_id": "o2", "method": "llm"}]
    summary = summarize_reviews([row], method_key=key)
    assert summary["by_method"]["template"]["completed_reviews"] == 1
    assert summary["by_method"]["llm"]["completed_reviews"] == 0


def test_snapshot_paths_resolve_relative_to_manifest_directory(tmp_path):
    from explanation_frontend.providers import FixtureProvider
    from explanation_frontend.models import make_context
    from explanation_frontend.templates import build_template
    from explanation_frontend.export import export_snapshot
    directory = tmp_path / 'snapshots'
    directory.mkdir()
    snapshot = FixtureProvider().run_fixture('supported')
    context = make_context(snapshot, list(snapshot.papers), snapshot.provisional)
    (directory / 'run.json').write_text(export_snapshot(context, build_template(context)), encoding='utf-8')
    result = run_study([{'case_id': 'relative', 'snapshot_file': 'snapshots/run.json'}],
                       mode='fixture', manifest_base=tmp_path)
    assert result['summary']['cases'] == 1
    assert result['review_packets'][0]['claim'] == snapshot.claim
