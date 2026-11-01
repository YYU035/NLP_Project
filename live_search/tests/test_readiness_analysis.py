"""Offline relevance reconciliation; no judgments or requests are generated."""

import importlib.util

from paper_search.metadata import ScientificPaper


def module():
    """Import the offline command without executing its CLI."""
    spec = importlib.util.spec_from_file_location(
        "readiness_analysis", "scripts/analyze_readiness_runs.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_conflicts_and_blanks_are_never_resolved_to_irrelevant():
    labels, conflicts = module().resolve_labels(
        {
            "c01:1": [2, 2],
            "c01:2": [1, 2],
            "c02:1": [],
            "c02:2": [0],
        }
    )
    assert labels == {"c01:1": 2, "c02:2": 0}
    assert conflicts == {"c01:2"}


def test_selected_metadata_and_claim_scoped_counts():
    paper = ScientificPaper(source="arxiv", source_id="one", abstract="text")
    records = [
        {"key": "c01:1", "paper": paper},
        {"key": "c01:1", "paper": paper},
        {"key": "c02:1", "paper": paper},
    ]
    result = module().stats(records, {"c01:1": 2})
    assert result == {
        "records": 3,
        "unique": 2,
        "relevant_known": 1,
        "evidence_known": 1,
        "unknown": 1,
        "abstract_unique": 2,
        "pdf_unique": 0,
    }
