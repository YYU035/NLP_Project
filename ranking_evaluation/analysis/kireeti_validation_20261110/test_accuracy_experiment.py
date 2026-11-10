"""Guard the source-preserving whole-abstract policy and its length fallback."""
from ranking_evaluation.analysis.kireeti_validation_20261110.accuracy_experiment import spans_for,segments


class FixedLengthTokenizer:
    def __init__(self,count):self.count=count
    def encode(self,*args,**kwargs):
        assert kwargs=={'add_special_tokens':True,'truncation':False}
        return [1]*self.count


def test_whole_abstract_requires_relevance_and_preserves_source():
    source='Background statement.\n  Relevant finding.'
    spans,reason=spans_for('bounded_abstract','p',source,segments(source),[.2,.8],FixedLengthTokenizer(512),'claim')
    assert len(spans)==1 and spans[0].text==source
    assert source[spans[0].start:spans[0].end]==source
    assert reason=='whole_abstract_with_relevance_gate'
    spans,reason=spans_for('bounded_abstract','p',source,segments(source),[.2,.4],FixedLengthTokenizer(50),'claim')
    assert spans==[] and reason=='below_anchor_threshold'


def test_overlong_abstract_falls_back_to_baseline_without_trimming():
    source='Background statement. Relevant finding.'
    args=('p',source,segments(source),[.2,.8],FixedLengthTokenizer(513),'claim')
    fallback,reason=spans_for('bounded_abstract',*args)
    baseline,_=spans_for('baseline',*args)
    assert fallback==baseline and fallback[0].text=='Relevant finding.'
    assert reason=='whole_abstract_exceeds_budget_fallback_to_baseline'
