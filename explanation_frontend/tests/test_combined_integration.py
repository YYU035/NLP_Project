import hashlib
import json
from pathlib import Path
import shutil

import pytest
from streamlit.testing.v1 import AppTest

from explanation_frontend.benchmark_view import ROOT, load_results, checked_path
from explanation_frontend.semantic_integration import select_bounded
from explanation_frontend._upstream.person3.semantic_evidence import select_scored_sentences


class Tokenizer:
    def encode(self, premise, hypothesis, **kwargs):
        assert kwargs == {'add_special_tokens': True, 'truncation': False}
        return list(range(len(premise.split()) + len(hypothesis.split()) + 3))


class Selector:
    def __init__(self, spans):
        self.spans = spans
    def select_abstract(self, *args):
        return self.spans


def test_three_dataset_report_preserves_configs_denominators_and_missing_gold():
    results = load_results()
    assert [d['metadata']['id'] for d in results] == ['scifact','csfcube','reciper']
    assert len(results[0]['rows']) == 6
    assert results[0]['metadata']['ranking_eligible_queries'] == 188
    assert results[0]['metadata']['ranking_excluded_no_positive'] == 112
    assert results[1]['evidence'] is None and results[2]['evidence'] is None
    assert results[2]['metadata']['total_queries'] == 1024
    assert isinstance(results[2]['rows'][0]['Recall@10'],float)


def test_benchmark_tampering_and_path_escape_are_rejected(tmp_path):
    copy = tmp_path/'assets'
    shutil.copytree(ROOT,copy)
    path = copy/'shared/datasets/evaluation_catalog.json'
    path.write_text('{}')
    with pytest.raises(ValueError,match='checksum'):
        load_results(copy)
    with pytest.raises(ValueError,match='escapes'):
        checked_path('../outside',copy)


def test_benchmark_ui_and_new_empty_workspaces_do_not_start_models():
    app = AppTest.from_file(str(Path(__file__).parents[1]/'app.py')).run()
    app.radio(key='workspace').set_value('Benchmark results').run()
    assert not app.exception
    app.selectbox(key='benchmark_dataset').set_value('CSFCube').run()
    assert not app.exception
    assert any('No expert sentence' in w.value for w in app.warning)
    app.selectbox(key='benchmark_dataset').set_value('RECIPER').run()
    assert any('1024' in t.value for t in app.text)
    app.radio(key='workspace').set_value('Local PDF').run()
    assert not app.exception and not app.metric
    app.radio(key='workspace').set_value('Model attribution').run()
    assert not app.exception and any('Verify a claim' in t.value for t in app.info)


def test_person3_budget_rejects_whole_span_without_trimming():
    text = 'word '*510
    spans = select_scored_sentences('p',[text],[1.0])
    decision = select_bounded('original claim','p',text,Tokenizer(),selector=Selector(spans))
    assert decision['status']=='budget_rejected' and decision['passages']==[]
    assert decision['rejected_passages'][0]['pair_tokens'] > 512
    assert not decision['rejected_passages'][0]['text_modified']
    short = select_scored_sentences('p',['Original exact text.'],[1.0])
    decision = select_bounded('claim','p','Original exact text.',Tokenizer(),selector=Selector(short))
    assert decision['passages'][0]['text']=='Original exact text.'
    assert decision['source_spans'][0]['start']==0


def test_successful_semantic_abstention_is_not_processing_failure():
    pytest.importorskip('paper_search')
    from explanation_frontend.tests.test_pipeline import source, verifier, Encoder
    from explanation_frontend.pipeline import LocalPipeline
    discovery = source()
    pipeline = LocalPipeline(encoder=Encoder(),verifier=verifier(),selector=Selector([]),tokenizer=Tokenizer())
    snapshot = pipeline.run_from_discovery(discovery,discovery.selected_paper_ids)
    assert snapshot.provisional.aggregated_verdict=='INSUFFICIENT'
    assert snapshot.provisional.status=='ok'
    assert not snapshot.verification.processing_failures
    assert all(v.sufficiency_cause=='NO_PASSAGE_ABOVE_THRESHOLD' for v in snapshot.verification.per_paper_verdicts.values())
    assert pipeline.aggregate(snapshot,discovery.selected_paper_ids[:1]).status=='ok'


def test_unsafe_pdf_inputs_rejected_without_neural_work():
    from explanation_frontend.pdf_view import verify_pdf
    with pytest.raises(ValueError):
        verify_pdf(b'not a PDF','claim')
    with pytest.raises(ValueError):
        verify_pdf(b'%PDF-1.7',' ')


def test_all_budget_rejections_have_no_scientific_verdict():
    pytest.importorskip('paper_search')
    from explanation_frontend.tests.test_pipeline import source, verifier, Encoder
    from explanation_frontend.pipeline import LocalPipeline
    discovery = source()
    spans = select_scored_sentences(discovery.selected_paper_ids[0],['word '*510],[1.0])
    pipeline = LocalPipeline(encoder=Encoder(),verifier=verifier(),selector=Selector(spans),tokenizer=Tokenizer())
    snapshot = pipeline.run_from_discovery(discovery,discovery.selected_paper_ids[:1])
    assert snapshot.provisional.status=='failed'
    assert snapshot.provisional.aggregated_verdict is None
    assert any(f.code=='VERIFIER_INPUT_LIMIT' for f in snapshot.verification.processing_failures)


def test_unstable_attribution_is_not_presented_as_reliable_word_scores():
    from explanation_frontend.providers import FixtureProvider
    from explanation_frontend.models import RunSnapshot
    from explanation_frontend.pipeline_setup import NLI_ID,NLI_REVISION
    data = FixtureProvider().run_fixture('supported').model_dump(mode='json')
    data['verification']['model_settings'].update(checkpoint=NLI_ID,revision=NLI_REVISION)
    snapshot = RunSnapshot.model_validate(data)
    pid = snapshot.verification.evidence_passages[0].passage_id
    app = AppTest.from_file(str(Path(__file__).parents[1]/'app.py'))
    app.session_state['workspace']='Model attribution'
    app.session_state['attribution_method']='Integrated Gradients (experimental)'
    app.session_state['snapshot']=snapshot
    app.session_state['attribution_result']=((snapshot.fingerprint,pid,'mask','Integrated Gradients (experimental)'),{
        'target_label':'SUPPORT','target_class':'entailment','convergence_delta':.8,
        'settings':{},'words':[{'text':'unreliable','attribution':1.0}], 'numerical_status':'unstable'})
    app.run()
    assert not app.exception
    assert any('did not pass' in w.value for w in app.warning)
    assert not app.dataframe
