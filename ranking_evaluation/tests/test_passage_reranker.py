import pytest
from ranking_evaluation.evidence_selection import EvidenceSpan
from ranking_evaluation.passage_reranker import rerank_passages


def test_reranker_changes_order_but_not_source_and_stable_ties():
    rows=[EvidenceSpan('p',(i,),i,i+1,text,0.7,'p:'+str(i)) for i,text in enumerate(['A','B'])]
    class Scorer:
        def score_pairs(self,pairs):assert pairs==[('claim','A'),('claim','B')];return [0,2]
    result=rerank_passages('claim',rows,Scorer())
    assert result[0].passage_id=='p:1' and result[0].text=='B' and result[0].start==1
    assert rows[1].score==0.7
    class Ties:
        def score_pairs(self,pairs):return [1,1]
    assert rerank_passages('claim',rows,Ties())[0].passage_id=='p:0'


def test_bad_reranker_output_never_silently_falls_back():
    row=EvidenceSpan('p',(0,),0,1,'A',1,'p:0')
    class Broken:
        def score_pairs(self,pairs):return [float('nan')]
    with pytest.raises(ValueError,match='finite'):rerank_passages('claim',[row],Broken())
