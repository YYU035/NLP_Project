import json
import numpy as np
import pytest

pytest.importorskip("paper_search")
from explanation_frontend.discovery import sample_discovery, DiscoverySnapshot
from explanation_frontend.models import make_context
from explanation_frontend.templates import build_template
from explanation_frontend.validation import validate_explanation
from explanation_frontend.export import export_snapshot, import_snapshot
from explanation_frontend.pipeline import LocalPipeline, safe_aggregate
from explanation_frontend.passage_selection import select_abstract


class Encoder:
    def encode(self, texts, **kwargs):
        return np.array([[1., float(len(t) % 5)] for t in texts])


def source():
    original = sample_discovery().model_dump(mode="json")
    original['result']['papers'] = original['result']['papers'][:2]
    claim = 'The synthetic intervention reduces recovery time.'
    original['result']['original_claim'] = claim
    for i, p in enumerate(original['result']['papers']):
        p['original_claim'] = claim
        p['paper']['abstract'] = ('The synthetic intervention reduces recovery time.' if i == 0 else
                                  'The synthetic intervention does not reduce recovery time.')
    original['selected_paper_ids'] = [p['identity'] for p in original['result']['papers']]
    return DiscoverySnapshot.model_validate(original)


def verifier():
    pytest.importorskip('torch')
    from explanation_frontend._upstream.claim_verifier.verifier import ClaimVerifier
    from explanation_frontend._upstream.claim_verifier.schemas import PassageVerdict
    engine = object.__new__(ClaimVerifier)
    engine.checkpoint, engine.revision = 'synthetic-test', 'test'
    engine.max_length, engine.threshold, engine.seed = 512, .50, 42
    def classify(claim, p):
        return PassageVerdict(passage_id=p.passage_id, paper_id=p.paper_id,
            label='CONTRADICT' if 'does not' in p.text else 'SUPPORT', confidence=.95)
    engine._classify_passage = classify
    return engine


def test_whole_chain_preserves_claim_rank_passages_subset_and_export():
    discovery = source()
    pipeline = LocalPipeline(encoder=Encoder(), verifier=verifier(), selector_mode="legacy")
    ids = [p.identity for p in discovery.result.papers]
    snapshot = pipeline.run_from_discovery(discovery, ids)
    assert snapshot.claim == discovery.result.original_claim
    assert all(p.source_version is None for p in snapshot.papers.values())
    assert snapshot.provisional.aggregated_verdict == 'CONFLICTING'
    assert snapshot.discovery_context['ranking_context']['papers']
    assert list(snapshot.papers) == [r['entry']['identity'] for r in
        snapshot.discovery_context['ranking_context']['papers']]
    for p in snapshot.verification.evidence_passages:
        original = next(x for x in discovery.result.papers if x.identity == p.paper_id).paper.abstract
        assert p.text in original
    result = pipeline.aggregate(snapshot, ids[:1])
    assert result.aggregated_verdict == 'SUPPORTED'
    context = make_context(snapshot, ids[:1], result)
    explanation = build_template(context)
    assert validate_explanation(explanation, context).displayable
    restored = import_snapshot(export_snapshot(context, explanation))
    assert restored.snapshot.fingerprint == snapshot.fingerprint
    assert pipeline.aggregate(snapshot, []).sufficiency_cause == 'EMPTY_SELECTION'


def test_exact_span_selection_keeps_negation_qualifiers_and_offsets():
    text = '  Adults improved. However, children did not improve. Results were uncertain.  '
    passages = select_abstract('children improve', 'p', text)
    assert any('did not improve' in p.text for p in passages)
    for p in passages:
        _, start, end, _ = p.passage_id.rsplit(':', 3)
        assert text[int(start):int(end)] == p.text


def test_no_abstract_never_runs_inference_and_discovery_failure_is_not_science():
    discovery = source().model_dump(mode='json')
    for entry in discovery['result']['papers']:
        entry['paper'].update(title=None, abstract=None)
    source_ = DiscoverySnapshot.model_validate(discovery)
    pipeline = LocalPipeline(selector_mode="legacy")
    snapshot = pipeline.run_from_discovery(source_, source_.selected_paper_ids)
    assert snapshot.provisional.status == 'failed'
    assert snapshot.provisional.aggregated_verdict is None
    discovery['result']['papers'] = []
    discovery['selected_paper_ids'] = []
    for provider in discovery['result']['providers'].values():
        provider['status'] = 'provider_failure'
    source_ = DiscoverySnapshot.model_validate(discovery)
    snapshot = pipeline.run_from_discovery(source_, [])
    assert snapshot.provisional.aggregated_verdict is None
    assert pipeline.aggregate(snapshot, []).status == 'failed'


def test_internal_conflict_keeps_both_sides_and_does_not_repeat_inference():
    discovery = source().model_dump(mode='json')
    discovery['result']['papers'] = discovery['result']['papers'][:1]
    discovery['selected_paper_ids'] = [discovery['result']['papers'][0]['identity']]
    discovery['result']['papers'][0]['paper']['abstract'] = (
        'The intervention reduces recovery time. The intervention does not reduce recovery time. '
        'In adults the intervention reduces recovery time.')
    source_ = DiscoverySnapshot.model_validate(discovery)
    pipeline = LocalPipeline(encoder=Encoder(), verifier=verifier(), selector_mode="legacy")
    snapshot = pipeline.run_from_discovery(source_, source_.selected_paper_ids)
    assert snapshot.provisional.aggregated_verdict == 'CONFLICTING'
    assert len({snapshot.verification.per_passage_labels[p.passage_id].label
                for p in snapshot.provisional.primary_evidence}) == 2
    pipeline.verifier._classify_passage = lambda *a: pytest.fail('Repeated neural inference')
    assert pipeline.aggregate(snapshot, source_.selected_paper_ids).aggregated_verdict == 'CONFLICTING'


def test_builtin_streamlit_selection_to_verdict_and_cached_checkbox_change(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    import explanation_frontend.pipeline as module
    discovery = source()
    ids = discovery.selected_paper_ids
    monkeypatch.delenv('EXPLANATION_VERIFY_DISCOVERY_FACTORY', raising=False)
    monkeypatch.setattr(module, 'create_provider', lambda: LocalPipeline(encoder=Encoder(), verifier=verifier(), selector_mode="legacy"))
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py'))
    app.session_state['workspace'] = 'Verify evidence'
    app.session_state['mode'] = 'Selected search'
    app.session_state['claim'] = discovery.result.original_claim
    app.session_state['pending_discovery'] = {'snapshot': discovery, 'ids': ids}
    app.run()
    next(b for b in app.button if b.label == 'Verify claim').click().run()
    if 'pipeline_job' in app.session_state:
        app.session_state['pipeline_job'].future.result(timeout=20)
        app.run()
    assert not app.exception
    assert any(m.value == 'CONFLICTING' for m in app.metric)
    current = app.session_state['snapshot']
    app.checkbox(key=f'paper:{current.fingerprint}:{ids[0]}').uncheck().run()
    if 'aggregation_job' in app.session_state:
        app.session_state['aggregation_job'].future.result(timeout=20)
        app.run()
    assert not app.exception
    assert any(m.value == 'CONTRADICTED' for m in app.metric)


def test_conflict_cap_reserves_a_slot_for_each_side():
    from explanation_frontend._upstream.claim_verifier.schemas import EvidencePassage
    engine = verifier()
    papers = [{'paper_id': f'p{i}', 'title': f'Synthetic {i}'} for i in range(4)]
    passages = {p['paper_id']: [EvidencePassage(passage_id=f"{p['paper_id']}:{j}",
        paper_id=p['paper_id'], section='abstract', text='does not reduce' if i == 3 else 'reduces')
        for j in range(3)] for i, p in enumerate(papers)}
    response = engine.verify('synthetic claim', papers, passages)
    original = engine.aggregate(response)
    labels = {p.passage_id: p.label.value for p in response.per_passage_labels}
    assert {labels[p.passage_id] for p in original.primary_evidence} == {'SUPPORT'}
    safe = safe_aggregate(engine, response)
    assert len(safe.primary_evidence) == 5
    assert {labels[p.passage_id] for p in safe.primary_evidence} == {'SUPPORT', 'CONTRADICT'}


def test_partial_discovery_stays_partial_on_changed_subsets():
    data = source().model_dump(mode='json')
    next(iter(data['result']['providers'].values()))['status'] = 'partial_failure'
    discovery = DiscoverySnapshot.model_validate(data)
    pipeline = LocalPipeline(encoder=Encoder(), verifier=verifier(), selector_mode="legacy")
    snapshot = pipeline.run_from_discovery(discovery, discovery.selected_paper_ids)
    assert snapshot.provisional.status == 'partial'
    assert not snapshot.search_complete
    assert pipeline.aggregate(snapshot, discovery.selected_paper_ids[:1]).status == 'partial'


def test_default_factory_requires_explicit_local_setup(monkeypatch, tmp_path):
    import explanation_frontend.pipeline as module
    monkeypatch.setattr(module, 'NLI_PATH', tmp_path / 'missing')
    with pytest.raises(ValueError, match='setup-pipeline'):
        module.create_provider()
