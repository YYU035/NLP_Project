import json
from types import SimpleNamespace
import pytest
from research_verifier.verification.primary_policy import PrimaryCandidatePipeline
from tests.test_pipeline import source, Encoder, verifier
from research_verifier.evidence.spans import EvidenceSpan
from research_verifier.explanations.replay import export_snapshot, import_snapshot
from research_verifier.contracts.models import make_context
from research_verifier.explanations.templates import build_template

class ExactSelector:
    def select_abstract(self,claim,pid,text):
        return [EvidenceSpan(pid,(0,),0,len(text),text,.8,pid+':abstract:0:'+str(len(text))+':unit')]
class Tokenizer:
    def encode(self,text,claim,**kwargs):return list(range(len(text.split())+len(claim.split())+3))

def test_candidate_provenance_keeps_subset_cache_and_replay_consistent():
    discovery=source();ids=[p.identity for p in discovery.result.papers]
    engine=PrimaryCandidatePipeline(encoder=Encoder(),verifier=verifier(),selector=ExactSelector(),tokenizer=Tokenizer())
    snapshot=engine.run_from_discovery(discovery,ids)
    assert snapshot.filters['abstract_policy']=='yuyue_primary'
    assert snapshot.filters['person3_commit'].startswith('ba8cdd6') and snapshot.filters['person3_base_commit'].startswith('7ed93fa')
    assert json.loads(snapshot.filters['context_configuration_json'])['min_paper_score']==.65
    assert snapshot.provisional.aggregated_verdict=='CONFLICTING'
    aggregate=engine.aggregate(snapshot,ids[:1]);assert aggregate.aggregated_verdict=='SUPPORTED'
    ctx=make_context(snapshot,ids[:1],aggregate);bundle=export_snapshot(ctx,build_template(ctx))
    assert import_snapshot(bundle).snapshot.fingerprint==snapshot.fingerprint

def test_default_factory_supports_explicit_baseline_and_rejects_unknown_policy(monkeypatch,tmp_path):
    import research_verifier.verification.primary_policy as module
    (tmp_path/'primary_policy.json').write_text(json.dumps({'default':'yuyue_primary'}))
    monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.delenv('EXPLANATION_ABSTRACT_POLICY',raising=False)
    monkeypatch.setattr(module,'create_candidate_provider',lambda:'candidate')
    monkeypatch.setattr(module,'create_baseline_provider',lambda:'baseline')
    assert module.create_provider()=='candidate'
    monkeypatch.setenv('EXPLANATION_ABSTRACT_POLICY','baseline');assert module.create_provider()=='baseline'
    monkeypatch.setenv('EXPLANATION_ABSTRACT_POLICY','typo')
    with pytest.raises(ValueError,match='Unknown'):module.create_provider()

def test_citation_bootstrap_clusters_transitive_paper_links_without_merging_unlinked_claims():
    from evaluation.independent_report import citation_clusters
    claims=[{'id':1,'cited_doc_ids':['a']},{'id':2,'cited_doc_ids':['a','b']},{'id':3,'cited_doc_ids':['b']},{'id':4,'cited_doc_ids':[]},{'id':5,'cited_doc_ids':[]}]
    groups=citation_clusters(claims)
    assert groups['1']==groups['2']==groups['3']
    assert groups['4']!=groups['5'] and groups['4']!=groups['1']
