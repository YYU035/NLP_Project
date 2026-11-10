"""Versioned abstract-context provider; original LocalPipeline remains the baseline."""
import json,os
from pathlib import Path
from .pipeline import LocalPipeline,create_provider as create_baseline_provider,local_verifier
from .semantic_integration import local_selector
from .models import RunSnapshot

ROOT=Path(__file__).parent
POLICY={'policy':'bounded_abstract','min_paper_score':.65,'max_pair_tokens':512,
        'base_selection':{'top_k':3,'min_score':.5,'context_radius':0},
        'source_commit':'ba8cdd6e7766306447913d8087b23f62195fa05a',
        'base_source_commit':'7ed93fa4be69c661422b33b6c88b949dd2636e12',
        'over_budget_policy':'unchanged_baseline_passages','text_policy':'exact_whole_original_abstract',
        'scope':'Selected abstract sources only; whole exact abstract when within verifier pair budget; baseline fallback otherwise. PDF selection is unchanged.'}

class LazyContextSelector:
    def select_abstract(self,claim,paper_id,abstract):
        from ._upstream.person3.contextual_evidence import BoundedAbstractSelector
        from dataclasses import asdict
        base=local_selector()
        if asdict(base.config)!=POLICY['base_selection']:raise ValueError('Base selection differs from frozen primary candidate')
        selector=BoundedAbstractSelector(base,local_verifier().tokenizer,min_paper_score=.65)
        return selector.select_abstract(claim,paper_id,abstract)

class PrimaryCandidatePipeline(LocalPipeline):
    def __init__(self,**kwargs):
        kwargs.setdefault('selector',LazyContextSelector());super().__init__(**kwargs)
    def run_from_discovery(self,source,paper_ids):
        snapshot=super().run_from_discovery(source,paper_ids)
        if self.selector_mode!='semantic':return snapshot
        data=snapshot.model_dump(mode='json')
        data['filters'].update(abstract_policy='yuyue_primary',context_configuration_json=json.dumps(POLICY),
            person3_commit=POLICY['source_commit'],person3_base_commit=POLICY['base_source_commit'],
            passage_selector='Yu Yue bounded exact abstract; paper cosine gate0.65; anchors0.5/top3; actual512-token pair budget')
        if self.selector_mode=='semantic':data['stage_statuses']['passages']='person3_bounded_context_gate065'
        result=RunSnapshot.model_validate(data)
        if self.adapter is not None:self.adapter._fingerprint=result.fingerprint
        self._snapshot=result
        return result

def create_candidate_provider():
    create_baseline_provider() # Same provisioning checks; no inference at construction.
    return PrimaryCandidatePipeline()

def create_provider():
    config=json.loads((ROOT/'primary_policy.json').read_text(encoding='utf-8-sig'))
    policy=os.environ.get('EXPLANATION_ABSTRACT_POLICY',config['default'])
    if policy=='baseline':return create_baseline_provider()
    if policy!='yuyue_primary':raise ValueError('Unknown EXPLANATION_ABSTRACT_POLICY; choose baseline or yuyue_primary')
    return create_candidate_provider()
