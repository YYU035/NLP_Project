"""Validate actual adapter handoff invariants without downloading weights."""
from dataclasses import asdict
import pytest
from ranking_evaluation.contextual_evidence import BoundedAbstractSelector
from ranking_evaluation.semantic_evidence import SelectionConfig,select_scored_sentences
from ranking_evaluation.analysis.kireeti_validation_20261110.accuracy_experiment import spans_for,segments


class Tokenizer:
    def __init__(self,count):self.count=count
    def encode(self,premise,hypothesis,**kwargs):
        assert hypothesis == 'original claim'
        assert kwargs == {'add_special_tokens':True,'truncation':False}
        return [1]*self.count


class Base:
    config=SelectionConfig(3,.5,0)
    def __init__(self,scores):self.scores=scores
    def select_abstract(self,claim,pid,text):
        offsets=segments(text)
        spans=select_scored_sentences(pid,[text[a:b] for a,b in offsets],self.scores,config=self.config)
        from dataclasses import replace
        return [replace(s,start=offsets[s.sentence_indices[0]][0],end=offsets[s.sentence_indices[-1]][1],text=text[offsets[s.sentence_indices[0]][0]:offsets[s.sentence_indices[-1]][1]]) for s in spans]


@pytest.mark.parametrize('tokens',[512,513])
def test_adapter_equals_evaluated_policy_and_preserves_source(tokens):
    text='Background sentence.\n  Relevant finding.';scores=[.2,.8];tokenizer=Tokenizer(tokens)
    adapter=BoundedAbstractSelector(Base(scores),tokenizer)
    expected,_=spans_for('bounded_abstract','p',text,segments(text),scores,tokenizer,'original claim')
    actual=adapter.select_abstract('original claim','p',text)
    # Baseline fake IDs differ; full context IDs and all source attributes agree.
    assert [(s.start,s.end,s.text,s.score,s.sentence_indices) for s in actual] == [(s.start,s.end,s.text,s.score,s.sentence_indices) for s in expected]
    assert all(text[s.start:s.end]==s.text for s in actual)
    if tokens==512:assert asdict(actual[0])==asdict(expected[0])


def test_irrelevant_paper_does_not_receive_whole_abstract():
    assert BoundedAbstractSelector(Base([.1,.2]),Tokenizer(50)).select_abstract('original claim','p','First sentence. Second sentence.')==[]


def test_real_tokenizer_and_valid_budget_are_required():
    with pytest.raises(ValueError):BoundedAbstractSelector(Base([.8]),None)
    with pytest.raises(ValueError):BoundedAbstractSelector(Base([.8]),Tokenizer(50),max_pair_tokens=513)


def test_optional_paper_gate_abstains_for_weak_paper_and_keeps_strong_context():
    text='Background sentence. Relevant finding.'
    weak=BoundedAbstractSelector(Base([.2,.6]),Tokenizer(50),min_paper_score=.65)
    strong=BoundedAbstractSelector(Base([.2,.8]),Tokenizer(50),min_paper_score=.65)
    assert weak.select_abstract('original claim','p',text)==[]
    assert strong.select_abstract('original claim','p',text)[0].text==text
    assert strong.context_configuration['min_paper_score']==.65
    with pytest.raises(ValueError):BoundedAbstractSelector(Base([.8]),Tokenizer(50),min_paper_score=float('nan'))
