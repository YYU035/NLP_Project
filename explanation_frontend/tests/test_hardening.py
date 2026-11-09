import json

import pytest

from explanation_frontend.models import RunSnapshot, make_context
from explanation_frontend.providers import FixtureProvider
from explanation_frontend.templates import build_template
from explanation_frontend.validation import validate_explanation
from explanation_frontend.providers import IntegratedProvider
from explanation_frontend.export import import_snapshot, export_snapshot
from explanation_frontend.models import digest


def test_unvalidated_limitations_cannot_smuggle_a_scientific_claim():
    provider = FixtureProvider()
    snapshot = provider.run_fixture("supported")
    context = make_context(snapshot, list(snapshot.papers), snapshot.provisional)
    explanation = build_template(context).model_copy(update={"limitations": ["All systems are safe."]})
    assert not validate_explanation(explanation, context).displayable


def test_nonfinite_retrieval_scores_rejected_before_hashing():
    data = json.loads(FixtureProvider().run_fixture("supported").model_dump_json())
    data["papers"]["demo:alpha"]["retrieval_scores"]["bm25"] = float("inf")
    with pytest.raises(ValueError):
        RunSnapshot.model_validate(data)


def test_decisive_verdict_without_decisive_evidence_is_rejected():
    data = json.loads(FixtureProvider().run_fixture("supported").model_dump_json())
    data["provisional"]["primary_evidence"] = []
    with pytest.raises(ValueError):
        RunSnapshot.model_validate(data)


def test_template_supports_large_valid_evidence_set():
    data = json.loads(FixtureProvider().run_fixture("supported").model_dump_json())
    original = data["verification"]["evidence_passages"][0]
    original_label = data["verification"]["per_passage_labels"][original["passage_id"]]
    for index in range(1, 32):
        passage = dict(original, passage_id=f"demo:alpha#p{index}")
        data["verification"]["evidence_passages"].append(passage)
        data["provisional"]["primary_evidence"].append(passage)
        data["verification"]["per_passage_labels"][passage["passage_id"]] = dict(original_label, passage_id=passage["passage_id"])
    snapshot = RunSnapshot.model_validate(data)
    context = make_context(snapshot, list(snapshot.papers), snapshot.provisional)
    assert len(build_template(context).statements) == 32


def test_typed_verifier_response_preserved_for_subset_aggregation():
    snapshot = FixtureProvider().run_fixture("supported")
    class TypedResponse:
        custom = "needed by upstream"
    original_response = TypedResponse()
    def aggregate(response, ids):
        assert isinstance(response, TypedResponse)
        assert response.custom == "needed by upstream"
        return FixtureProvider().aggregate(snapshot, ids)
    live = snapshot.model_dump()
    live["mode"] = "live"
    provider = IntegratedProvider(search=lambda *a: [], rank=lambda *a: [p.model_dump() for p in snapshot.papers.values()],
        select=lambda *a: {}, verify=lambda *a: original_response, aggregate=aggregate, pack=lambda *a: live)
    result = provider.run(snapshot.claim, {})
    assert provider.aggregate(result, []).aggregated_verdict == "INSUFFICIENT"


def test_original_v1_exports_replay_after_optional_discovery_field_added():
    snapshot = FixtureProvider().run_fixture("supported")
    context = make_context(snapshot, list(snapshot.papers), snapshot.provisional)
    data = json.loads(export_snapshot(context, build_template(context)))
    data["snapshot"].pop("discovery_context", None)
    legacy_hash = digest(data["snapshot"])
    data["snapshot_hash"] = legacy_hash
    data["explanation"]["context_key"] = digest({"snapshot": legacy_hash,
        "selection": context.selected_paper_ids, "aggregation": context.aggregation.model_dump(mode="json")})
    assert import_snapshot(json.dumps(data)).snapshot.claim == snapshot.claim


def test_legacy_integrated_replay_restores_only_the_known_safety_note():
    from explanation_frontend.templates import AUTOMATIC_SELECTION_LIMITATION
    source = FixtureProvider().run_fixture('supported').model_dump(mode='json')
    source['stage_statuses']['passages'] = 'frontend_abstract_windows_v1'
    snapshot = RunSnapshot.model_validate(source)
    context = make_context(snapshot, list(snapshot.papers), snapshot.provisional)
    data = json.loads(export_snapshot(context, build_template(context)))
    data['explanation']['limitations'].remove(AUTOMATIC_SELECTION_LIMITATION)
    migrated = import_snapshot(json.dumps(data))
    assert AUTOMATIC_SELECTION_LIMITATION in migrated.explanation.limitations
    assert migrated.snapshot_hash == snapshot.fingerprint
    assert migrated.explanation.statements == build_template(context).statements
    data['explanation']['limitations'] = ['All systems are safe.']
    with pytest.raises(ValueError):
        import_snapshot(json.dumps(data))
