"""Optional Cross-Encoder reranking of exact, already selected source passages."""
from dataclasses import replace
from math import isfinite


def rerank_passages(claim,spans,scorer,*,top_k=1):
    if not claim.strip() or type(top_k) is not int or top_k<0:
        raise ValueError('Nonempty original claim and nonnegative top_k required')
    if len({s.passage_id for s in spans})!=len(spans):raise ValueError('Duplicate passage identity')
    if not spans or top_k==0:return []
    scores=list(scorer.score_pairs([(claim,s.text) for s in spans]))
    if len(scores)!=len(spans):raise ValueError('Expected one reranker score per passage')
    normalized=[]
    for score in scores:
        if isinstance(score,(bool,str,bytes)) or getattr(score,'ndim',0)!=0 or not isfinite(float(score)):
            raise ValueError('Reranker scores must be finite scalars')
        normalized.append(float(score))
    order=sorted(range(len(spans)),key=lambda i:(-normalized[i],i))[:top_k]
    return [replace(spans[i],score=normalized[i]) for i in order]
