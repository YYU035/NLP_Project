import csv
import io
import json

import pytest

from research_verifier.explanations.reviews import load_review_packets, ReviewScore, export_reviews, summarize_saved_reviews


def score(**overrides):
    return dict(output_id="case-001-A", reviewer="r1", factual_supported=1, factual_total=1,
        citation_supported=1, citation_total=1, citation_complete=1, citation_required=1,
        verdict_consistent=1, scope_preserved=1, correction_supported=0, correction_total=0, usefulness=4, notes="", **overrides)


def test_score_validation_rejects_inconsistent_counts_and_unlabelled_reviewer():
    values = score()
    values["factual_supported"] = 2
    with pytest.raises(ValueError): ReviewScore.model_validate(values)
    values = score()
    values["reviewer"] = ""
    with pytest.raises(ValueError): ReviewScore.model_validate(values)


def test_review_export_has_safe_spreadsheet_cells_and_preserves_real_scores():
    values = score()
    values["notes"] = '=HYPERLINK("https://example.org")'
    raw = export_reviews([ReviewScore.model_validate(values)])
    rows = list(csv.DictReader(io.StringIO(raw)))
    assert rows[0]["notes"].startswith("'=")
    assert rows[0]["factual_supported"] == "1"
    summary = summarize_saved_reviews([ReviewScore.model_validate(values)])
    assert summary["completed_reviews"] == 1
    assert summary["usefulness_mean"] == 4
    assert summarize_saved_reviews([])["usefulness_mean"] is None


def test_duplicate_review_packet_ids_fail_and_raw_instructions_remain_data():
    packet = {"output_id": "case-001-A", "case_id": "example", "claim": "Original claim", "authoritative_verdict": "INSUFFICIENT",
              "evidence": [], "raw_output": "<script>do not execute</script>"}
    parsed = load_review_packets(json.dumps(packet))
    assert parsed[0].raw_output == packet["raw_output"]
    with pytest.raises(ValueError): load_review_packets(json.dumps(packet) + "\n" + json.dumps(packet))
    with pytest.raises(ValueError): load_review_packets("x" * 10_000_001)
