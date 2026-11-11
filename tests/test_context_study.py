from types import SimpleNamespace
import pytest
from evaluation.context_study.policy import chosen_arm
from evaluation.context_study.context_adapter import BoundedAbstractSelector
from evaluation.context_study.run_local import native_from_trace, evidence_metrics
from research_verifier.evidence.spans import EvidenceSpan
from research_verifier.evidence.selector import SelectionConfig

class CharacterBudgetTokenizer:
    def encode(self,text,claim,**kwargs):return list(range(len(text)+len(claim)+3))

class FixedSelector:
    config=SelectionConfig()
    def __init__(self,score=.7,empty=False):self.score=score;self.empty=empty
    def select_abstract(self,claim,pid,text):
        return [] if self.empty else [EvidenceSpan(pid,(0,),0,4,text[:4],self.score,'anchor')]

@pytest.mark.parametrize('focused,expected',[('SUPPORTED','baseline'),('CONTRADICTED','baseline'),('INSUFFICIENT','bounded_abstract'),('CONFLICTING','bounded_abstract')])
def test_guard_short_circuits_decisive_focus_and_expands_uncertain_focus(focused,expected):
    assert chosen_arm('protect_decisive',focused)==expected

def test_context_exact_pair_boundary_and_source_preservation():
    adapter=BoundedAbstractSelector(FixedSelector(),CharacterBudgetTokenizer())
    exact='A'*508
    out=adapter.select_abstract('C','p',exact)
    assert out[0].text==exact and out[0].start==0 and out[0].end==508
    over=exact+'A'
    out=adapter.select_abstract('C','p',over)
    assert len(out)==1 and out[0].text==over[:4] and out[0].passage_id=='anchor'

def test_gate_cannot_reintroduce_abstained_text_and_validates_threshold():
    assert BoundedAbstractSelector(FixedSelector(empty=True),CharacterBudgetTokenizer()).select_abstract('C','p','text')==[]
    assert BoundedAbstractSelector(FixedSelector(score=.6),CharacterBudgetTokenizer(),min_paper_score=.65).select_abstract('C','p','text')==[]
    assert BoundedAbstractSelector(FixedSelector(score=.65),CharacterBudgetTokenizer(),min_paper_score=.65).select_abstract('C','p','text')
    for bad in [True,float('nan'),float('inf'),1.1]:
        with pytest.raises(ValueError):BoundedAbstractSelector(FixedSelector(),CharacterBudgetTokenizer(),min_paper_score=bad)

def test_cached_frontend_response_is_translated_without_invented_raw_probabilities():
    row={'response':{'per_paper_verdicts':{'p':{'paper_id':'p','title':'Paper','verdict':'SUPPORTED','confidence':.9,'evidence_type':'abstract_only'}},
        'per_passage_labels':{'s':{'passage_id':'s','paper_id':'p','label':'SUPPORT','confidence':.9,'truncated':False}},
        'processing_failures':[],'evidence_passages':[{'passage_id':'s','paper_id':'p','section':'abstract','text':'text','score':.7}],
        'model_settings':{'checkpoint':'local-model','revision':'fixed','max_length':512,'threshold':.5,'seed':42}}}
    native=native_from_trace(row,'Claim')
    assert native.per_paper_verdicts[0].evidence_type=='abstract'
    assert native.per_passage_labels[0].raw_probabilities=={}
    assert native.model_settings.tokenizer_max_length==512
    assert row['response']['per_paper_verdicts']['p']['evidence_type']=='abstract_only'

def test_evidence_union_counts_extra_context_and_does_not_double_count_anchors():
    docs={'p':{'abstract':['Alpha.','Beta.']}}
    claim={'evidence':{'p':[{'sentences':[0],'label':'SUPPORT'}]}}
    out=evidence_metrics(claim,docs,{'p':[SimpleNamespace(start=0,end=6),SimpleNamespace(start=0,end=12)]})
    assert out['evidence']['tp']==1 and out['evidence']['fp']==1 and out['evidence']['fn']==0

@pytest.mark.parametrize('encoding',['utf-8-sig','utf-16'])
def test_research_report_reads_python_and_windows_powershell_logs(tmp_path,encoding):
    from evaluation.context_study.report import read_log
    path=tmp_path/'tests.log';path.write_bytes('All tests passed. Unicode: λ'.encode(encoding))
    assert read_log(path)=='All tests passed. Unicode: λ'
