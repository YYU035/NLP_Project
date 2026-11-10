import json
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from explanation_frontend.overview_view import load_showcase
from explanation_frontend.inspection import evidence_counts

ROOT=Path(__file__).resolve().parents[1]
APP=str(ROOT/'app.py')


def test_all_actual_showcase_replays_validate_and_do_not_allow_path_escape():
    manifest=json.loads((ROOT/'showcase/manifest.json').read_text(encoding='utf-8'))
    assert len(manifest['sha256'])==8
    for name in manifest['sha256']:
        bundle=load_showcase(name)
        assert bundle.snapshot.mode=='cached'
        assert bundle.snapshot.verification.model_settings.revision
    with pytest.raises(ValueError,match='escape'):load_showcase('../outside')


def test_overview_replay_and_fresh_candidate_handoff():
    app=AppTest.from_file(APP);app.session_state['workspace']='Project overview';app.run()
    assert not app.exception
    assert any(m.value=='95.9%' for m in app.metric)
    replay=next(b for b in app.button if b.key and b.key.startswith('showcase:'))
    replay.click().run()
    assert not app.exception and app.radio(key='workspace').value=='Verify evidence'
    assert app.session_state['mode']=='Cached replay'
    app.radio(key='workspace').set_value('Project overview').run()
    next(b for b in app.button if b.key and b.key.startswith('run:')).click().run()
    assert not app.exception and app.radio(key='workspace').value=='Find papers'
    assert app.session_state['discovery_snapshot'].ranking_context is None
    assert not any(m.value in ['SUPPORTED','CONTRADICTED','CONFLICTING'] for m in app.metric)


def test_actual_review_set_stays_unscored_until_a_human_reviews():
    app=AppTest.from_file(APP);app.session_state['workspace']='Evaluate explanations';app.run()
    next(b for b in app.button if b.label=='Load actual-model review set').click().run()
    assert not app.exception
    assert any(m.label=='Saved human reviews' and m.value=='0' for m in app.metric)
    assert len(app.session_state['review_packets'])==8


def test_scoped_counts_refuse_unknown_or_duplicate_papers():
    manifest=json.loads((ROOT/'showcase/manifest.json').read_text(encoding='utf-8'))
    snapshot=load_showcase(next(iter(manifest['sha256']))).snapshot
    counts=evidence_counts(snapshot,list(snapshot.papers))
    assert sum(counts.values())<=len(snapshot.verification.evidence_passages)
    assert sum(evidence_counts(snapshot,[]).values())==0
    pid=next(iter(snapshot.papers))
    for ids in [[pid,pid],['unknown']]:
        with pytest.raises(ValueError):evidence_counts(snapshot,ids)


def test_updated_csfcube_test_and_comprehensive_assets_are_present():
    from explanation_frontend.benchmark_view import load_results
    data=load_results();cs=data[1]['metadata']
    assert cs['total_queries']==26 and cs['development_summary_path']
    assert data[0]['source']['revision']=='90aaca109585c12c877367b1a08fe4dfee4a4de3'

def test_recorded_attribution_is_exact_input_bound_and_checksum_protected(tmp_path,monkeypatch):
    import shutil
    from explanation_frontend import attribution_cache as cache
    bundle=load_showcase('scifact-230.json');snapshot=bundle.snapshot
    pid=snapshot.provisional.primary_evidence[0].passage_id
    result=cache.load_recorded(snapshot,pid)
    assert result['recorded'] and result['kind']=='word_occlusion'
    changed=snapshot.model_copy(update={'claim':snapshot.claim+' modified'})
    assert cache.load_recorded(changed,pid) is None
    copy=tmp_path/'cache';shutil.copytree(cache.ROOT,copy);monkeypatch.setattr(cache,'ROOT',copy)
    item=next(iter(json.loads((copy/'manifest.json').read_text()).values()))
    (copy/item['file']).write_text('{}')
    with pytest.raises(ValueError,match='checksum'):cache.load_recorded(snapshot,pid)


def test_saved_word_sensitivity_displays_without_loading_model(monkeypatch):
    import explanation_frontend.attribution_view as view
    def forbidden():raise AssertionError('Saved attribution must not load the model')
    monkeypatch.setattr(view,'local_verifier',forbidden)
    app=AppTest.from_file(APP)
    app.session_state['workspace']='Model attribution'
    app.session_state['snapshot']=load_showcase('scifact-230.json').snapshot
    app.run()
    next(b for b in app.button if b.label=='Load recorded word sensitivity').click().run()
    assert not app.exception and app.dataframe
    assert any('no new inference' in info.value for info in app.info)

def test_top_k_stably_selects_cutoff_ties_and_rejects_nonfinite_scores():
    import numpy as np
    from explanation_frontend._upstream.retrieval.retriever import _top_k
    scores=np.array([0.,0.,0.,0.,1.,0.,0.,0.,0.,0.])
    assert _top_k(scores,3).tolist()==[4,0,1]
    for seed in range(10):
        scores=np.random.default_rng(seed).integers(0,4,100).astype(float)
        expected=np.lexsort((np.arange(len(scores)),-scores))[:7]
        assert _top_k(scores,7).tolist()==expected.tolist()
    with pytest.raises(ValueError):_top_k(np.array([float('nan'),1.]),1)
