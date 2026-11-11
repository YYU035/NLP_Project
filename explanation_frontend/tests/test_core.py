import copy
import json

import pytest

from explanation_frontend.models import RunSnapshot, make_context
from explanation_frontend.providers import FixtureProvider, fixture_names
from explanation_frontend.templates import build_template
from explanation_frontend.validation import validate_explanation


def context(name="conflict", selected=None):
    provider = FixtureProvider()
    snapshot = provider.run_fixture(name)
    ids = list(snapshot.papers) if selected is None else selected
    return make_context(snapshot, ids, provider.aggregate(snapshot, ids))


def test_fixture_journey_distinguishes_empty_and_failed():
    assert len(fixture_names()) >= 10
    assert context("supported").aggregation.aggregated_verdict == "SUPPORTED"
    assert context("contradicted").aggregation.aggregated_verdict == "CONTRADICTED"
    assert context("empty").aggregation.aggregated_verdict == "INSUFFICIENT"
    assert context("failure").aggregation.aggregated_verdict is None
    assert context("failure").aggregation.status == "failed"


def test_subset_keeps_conflict_and_drops_excluded_citations():
    ctx = context()
    assert ctx.aggregation.aggregated_verdict == "CONFLICTING"
    scoped = context(selected=["demo:alpha"])
    assert scoped.aggregation.aggregated_verdict == "SUPPORTED"
    assert set(scoped.passages) == {"demo:alpha#p0"}
    explanation = build_template(scoped)
    assert validate_explanation(explanation, scoped).displayable
    assert all(c.startswith("demo:alpha") for s in explanation.statements for c in s.citation_ids)
    assert context(selected=[]).aggregation.aggregated_verdict == "INSUFFICIENT"


@pytest.mark.parametrize("mutation", ["version", "duplicate", "dangling", "nan", "paper_reference", "primary"])
def test_corrupt_snapshot_is_rejected(mutation):
    data = json.loads(FixtureProvider().run_fixture("supported").model_dump_json())
    if mutation == "version":
        data["schema_version"] = "999"
    elif mutation == "duplicate":
        data["verification"]["evidence_passages"] *= 2
    elif mutation == "dangling":
        data["verification"]["evidence_passages"][0]["paper_id"] = "unknown"
    elif mutation == "nan":
        data["verification"]["per_paper_verdicts"]["demo:alpha"]["confidence"] = float("nan")
    elif mutation == "paper_reference":
        data["verification"]["per_paper_verdicts"]["demo:alpha"]["supporting_passage_ids"] = ["missing"]
    else:
        data["provisional"]["primary_evidence"][0]["text"] = "Substituted text"
    with pytest.raises(ValueError):
        RunSnapshot.model_validate(data)


def test_context_rejects_unselected_or_unknown_evidence():
    ctx = context()
    with pytest.raises(ValueError):
        make_context(ctx.snapshot, ["unknown"], ctx.aggregation)
    with pytest.raises(ValueError):
        make_context(ctx.snapshot, ["demo:alpha"], ctx.aggregation)


def test_template_preserves_whole_passages_and_limitations():
    ctx = context("qualified")
    explanation = build_template(ctx)
    assert explanation.statements[0].text == "In this simulated benchmark, latency decreased by 12% (95% CI 8–16%) at 20 ms; this does not establish performance on other systems."
    assert explanation.correction is None
    assert validate_explanation(explanation, ctx).displayable


def test_existing_but_irrelevant_reference_does_not_validate_prose():
    ctx = context("supported")
    explanation = build_template(ctx)
    data = explanation.model_dump()
    data["statements"][0]["text"] = "This proves all systems are safe."
    from explanation_frontend.models import Explanation
    report = validate_explanation(Explanation.model_validate(data), ctx)
    assert report.references_valid
    assert not report.displayable
    assert not report.extractive_support


def test_neutral_and_truncated_passages_cannot_become_explanations():
    assert build_template(context("neutral")).statements == []
    assert build_template(context("truncated")).statements == []
    assert "TRUNCATED" in " ".join(build_template(context("truncated")).limitations)


def test_snapshots_are_detached_between_sessions():
    a = FixtureProvider().run_fixture("supported")
    b = FixtureProvider().run_fixture("supported")
    changed = copy.deepcopy(a.model_dump())
    changed["papers"]["demo:alpha"]["title"] = "Changed"
    assert b.papers["demo:alpha"].title != changed["papers"]["demo:alpha"]["title"]
