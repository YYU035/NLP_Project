import math
import pytest
pytest.importorskip("torch")
pytest.importorskip("captum")
pytest.importorskip("transformers")
from research_verifier.interpretability.service import reliability, explain_checked
from evaluation.xai_qc import summarise, sample_examples
from tests.test_integrated_gradients import model_dir, verifier, CLAIM_TXT, PASSAGE


def result(delta=0.0001,p=.8,b=.3):
    return dict(target_probability=p,baseline_probability=b,convergence_delta=delta,
                truncated=False,words=[dict(attribution=.2,normalized=1.)])


def test_gate_rejects_large_absolute_error_even_with_good_relative_error():
    assert reliability(result(.02,p=.9,b=.1))['numerical_status']=='unstable'
    assert reliability(result(.0001))['numerical_status']=='passed'


def test_gate_rejects_small_gap_nonfinite_and_truncated():
    assert reliability(result(.0001,p=.50001,b=.5))['numerical_status']=='unstable'
    assert not reliability(result(float('nan')))['finite']
    r=result();r['truncated']=True
    assert reliability(r)['numerical_status']=='unstable'


def test_checked_actual_small_model_preserves_verifier_and_matches_target(verifier):
    from research_verifier.verification.schemas import PassageLabel
    pred=verifier._classify_passage(CLAIM_TXT,PASSAGE)
    r=explain_checked(verifier,CLAIM_TXT,PASSAGE,steps=(16,64),target=PassageLabel.SUPPORT)
    assert r['target_probability']==pytest.approx(pred.raw_probabilities['entailment'],abs=1e-6)
    assert r['settings']['baseline_name']=='mask'
    assert r['numerical_status']=='passed'
    assert not verifier.model.training
    assert verifier._classify_passage(CLAIM_TXT,PASSAGE)==pred


def test_budget_refused_before_attribution(verifier,monkeypatch):
    monkeypatch.setattr(verifier,'max_length',2)
    with pytest.raises(ValueError,match='budget'):
        explain_checked(verifier,CLAIM_TXT,PASSAGE)


def test_summary_never_counts_unstable_or_failed_metrics():
    s=summarise([dict(numerical_status='unstable',ap=1),dict(error='failed')])
    assert s['stable']==0 and s['unstable']==1 and s['failed']==1
    assert s['all_stable']['ap'] is None


def test_balanced_selection_reproducible_and_length_bounded():
    class T:
        def encode(self,text,claim,**kw):return text.split()
    examples=[dict(claim_id=i,doc_id=str(i),claim='claim',sentences=['short'],gold_label=label) for i,label in enumerate(['SUPPORT']*5+['CONTRADICT']*5)]
    a,n=sample_examples(examples,T(),4,10,42)
    assert a==sample_examples(examples,T(),4,10,42)[0]
    assert [e['gold_label'] for e in a]==['SUPPORT','CONTRADICT']*2
    assert n==10


def test_summary_excludes_rows_whose_downstream_evaluation_failed():
    s=summarise([dict(numerical_status='passed',error='perturbation failed')])
    assert s['stable']==0 and s['failed']==1 and s['all_stable']['ap'] is None

def test_stable_word_table_and_baseline_changes_hide_stale_results():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from research_verifier.verification.providers import FixtureProvider
    from research_verifier.contracts.models import RunSnapshot
    from research_verifier.runtime.setup_verifier import NLI_ID, NLI_REVISION
    data=FixtureProvider().run_fixture('supported').model_dump(mode='json')
    data['verification']['model_settings'].update(checkpoint=NLI_ID,revision=NLI_REVISION)
    snapshot=RunSnapshot.model_validate(data)
    pid=snapshot.verification.evidence_passages[0].passage_id
    app=AppTest.from_file(str(Path(__file__).parents[1]/'src/research_verifier/ui/app.py'))
    app.session_state['workspace']='Model attribution'
    app.session_state['attribution_method']='Integrated Gradients (experimental)'
    app.session_state['snapshot']=snapshot
    app.session_state['attribution_result']=((snapshot.fingerprint,pid,'mask','Integrated Gradients (experimental)'),{
        'target_label':'SUPPORT','target_class':'entailment','convergence_delta':.00001,
        'settings':{'baseline':'mask'},'words':[{'text':'<script>unsafe</script>',
        'attribution':.1,'normalized':1.,'segment':'evidence','start':0,'end':6}],
        'numerical_status':'passed'})
    app.run()
    assert not app.exception and len(app.dataframe)==1
    app.selectbox(key='attribution_baseline').select('pad').run()
    assert not app.exception and not app.dataframe

def test_evaluation_driver_computes_gated_metrics_with_local_tiny_model(verifier,tmp_path,monkeypatch):
    import json
    from evaluation import xai_qc
    import research_verifier.pipeline as pipeline; import research_verifier.evidence.semantic as semantic_integration
    class Selector:
        def select(self,*args):return []
    monkeypatch.setattr(pipeline,'local_verifier',lambda:verifier)
    monkeypatch.setattr(semantic_integration,'local_selector',lambda:Selector())
    monkeypatch.setattr(xai_qc,'ROOT',tmp_path)
    data=tmp_path/'data';data.mkdir()
    (data/'corpus.jsonl').write_text(json.dumps({'doc_id':1,'abstract':['Vitamin D reduced hip fractures.','The effect was dose dependent.']})+'\n')
    (data/'claims_dev.jsonl').write_text(json.dumps({'id':1,'claim':CLAIM_TXT,'evidence':{'1':[{'label':'SUPPORT','sentences':[0]}]}})+'\n')
    summary=xai_qc.main(['--data-dir',str(data),'--out-dir',str(tmp_path/'out'),'--limit','1','--steps','16','--random-trials','2'])
    assert summary['attempted']==summary['stable']==1
    assert summary['all_stable']['ap'] is not None
    assert math.isfinite(summary['all_stable']['faithfulness']['aopc_comprehensiveness_random'])
