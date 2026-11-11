"""Shared frozen corpus scoring with actual production lexical/dense components."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from research_verifier.retrieval.engine import _top_k
from research_verifier.retrieval.bm25 import BM25Retriever
from research_verifier.retrieval.schemas import Paper
from research_verifier.retrieval.embeddings import SBERTRetriever
ROOT=Path(__file__).resolve().parents[2]

def encode_cached(encoder,texts,key):
    cache=ROOT/'outputs/validation/embeddings';cache.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256((key+'\0'+'\0'.join(texts)).encode()).hexdigest();path=cache/(digest+'.npy')
    if path.exists():
        arr=np.load(path,allow_pickle=False)
        if len(arr)!=len(texts) or not np.isfinite(arr).all():raise ValueError('Invalid cached embeddings')
        return arr
    arr=np.asarray(encoder.encode(texts,batch_size=64,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=len(texts)>500),dtype=np.float32)
    arr=arr/np.clip(np.linalg.norm(arr,axis=1,keepdims=True),1e-12,None)
    if not np.isfinite(arr).all():raise ValueError('Nonfinite embeddings')
    np.save(path,arr,allow_pickle=False);return arr

def rank_dataset(documents,queries,encoder,*,stemmer,model_key,branch_depth=None,progress=None):
    from research_verifier.retrieval.ranking import runtime
    runtime() # Bind the owned, pinned Person3 module namespace.
    from research_verifier.retrieval.fusion import fuse_ordered_ids
    papers=[Paper(paper_id=str(pid),title=row['title'],abstract=row['abstract']) for pid,row in documents.items()]
    ids=[p.paper_id for p in papers];positions={pid:i for i,pid in enumerate(ids)}
    dense=encode_cached(encoder,[p.index_text() for p in papers],model_key+':corpus')
    qids=list(queries);qvec=encode_cached(encoder,[queries[q]['text'] for q in qids],model_key+':queries')
    all_ids=set(ids);global_lexical=None
    if all(set(q['candidate_ids'])==all_ids for q in queries.values()):
        global_lexical=BM25Retriever(stemmer=stemmer).index(papers)
    results={m:{} for m in ['bm25','sbert','rrf']}
    for index,qid in enumerate(qids):
        query=queries[qid];pool=query['candidate_ids'];indices=[positions[pid] for pid in pool]
        lexical=global_lexical or BM25Retriever(stemmer=stemmer).index([papers[i] for i in indices])
        bs=lexical.score_all(query['text'])
        if global_lexical is not None:bs=bs[indices]
        ds=dense[indices]@qvec[index]
        b=[pool[i] for i in _top_k(bs,len(pool))];s=[pool[i] for i in _top_k(ds,len(pool))]
        results['bm25'][qid]=b;results['sbert'][qid]=s
        results['rrf'][qid]=[pid for pid,_ in fuse_ordered_ids(b[:branch_depth],s[:branch_depth],rrf_k=60)]
        if progress and (index%25==0 or index+1==len(qids)):progress(index+1,len(qids))
    return results
