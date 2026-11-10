from types import SimpleNamespace
import math
import pytest
from explanation_frontend.evaluation.validation.metrics import ranking_metrics,classification,evidence_score,bootstrap,reliability
from explanation_frontend.evaluation.validation.scifact import sentence_recovery,gold_label


def test_ranking_metrics_hand_computed_and_duplicate_guard():
    r=ranking_metrics(['a','x','b'],{'a','b'})
    assert r['Recall@5']==1 and r['MRR@10']==1
    assert r['nDCG@10']==pytest.approx((1+1/math.log2(4))/(1+1/math.log2(3)))
    assert ranking_metrics(['a'],set()) is None
    with pytest.raises(ValueError):ranking_metrics(['a','a'],{'a'})


def test_conflicts_and_operational_failures_are_errors_and_classes_are_explicit():
    rows=[{'gold':'SUPPORTED','prediction':'SUPPORTED'},{'gold':'CONTRADICTED','prediction':'CONFLICTING'}]
    r=classification(rows)
    assert r['accuracy']==.5 and r['macro_f1']==.5
    assert r['macro_classes']==['SUPPORTED','CONTRADICTED']
    rows.append({'gold':'INSUFFICIENT','prediction':None})
    assert classification(rows)['macro_f1']==pytest.approx(1/3)
    with pytest.raises(ValueError):classification([{'gold':'wrong','prediction':'SUPPORTED'}])


def test_evidence_micro_counts():
    r=evidence_score(3,1,2)
    assert r['precision']==.75 and r['recall']==.6 and r['f1']==pytest.approx(2/3)


def test_production_fragments_do_not_receive_whole_sentence_credit():
    sentences=['First sentence.','An abbreviation e.g. this remains one source sentence.']
    full,partial=sentence_recovery(sentences,[SimpleNamespace(start=0,end=15),SimpleNamespace(start=16,end=36)])
    assert full=={0} and partial=={1}
    full,partial=sentence_recovery(['abcdef'],[SimpleNamespace(start=0,end=3),SimpleNamespace(start=3,end=6)])
    assert full=={0} and not partial


def test_reliability_bin_boundaries_count_each_case_once():
    rows=[{'confidence':x,'correct':True} for x in [.1,.2,.3,.7,.9,1.]]
    r=reliability(rows)
    assert sum(b['n'] for b in r['bins'])==len(rows)
    assert r['selective_accuracy'][-1]['coverage']==pytest.approx(1/6)
    with pytest.raises(ValueError):reliability([{'confidence':1.2,'correct':True}])


def test_bootstrap_is_seeded_clustered_and_checks_alignment():
    a=bootstrap([0.,1.,1.],['a','a','b'],repeats=100)
    assert a==bootstrap([0.,1.,1.],['a','a','b'],repeats=100)
    assert a['estimate']==pytest.approx(2/3) and a['clusters']==2
    with pytest.raises(ValueError):bootstrap([0.,1.],['a'],repeats=100)


def test_claim_labels_do_not_invent_neutral_gold_passages():
    assert gold_label({'evidence':{}})=='INSUFFICIENT'
    assert gold_label({'evidence':{'1':[{'label':'SUPPORT'}]}})=='SUPPORTED'
