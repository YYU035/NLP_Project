"""Train-selected context policies, evaluated on gold-paper SciFact development inputs.

This isolates selection from retrieval. Gold labels choose evaluation papers but
never passages. No verifier inference or end-to-end accuracy is implied.
"""
import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
from pathlib import Path
from time import perf_counter
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ranking_evaluation.evidence_selection import EvidenceSpan
from ranking_evaluation.verifier_budget import enforce_verifier_budget
from explanation_frontend.evaluation.validation.scifact import sentence_recovery
from explanation_frontend.evaluation.validation.metrics import evidence_score

POLICIES = [('baseline', None), ('all_neighbors', -1.), ('neighbors_0.3', .3), ('neighbors_0.4', .4), ('neighbors_0.5', .5), ('neighbors_0.6', .6)]


def segments(source):
    cuts = [0, *[m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z0-9])', source)], len(source)]
    result = []
    for a,b in zip(cuts,cuts[1:]):
        raw=source[a:b]
        a,b=a+len(raw)-len(raw.lstrip()),a+len(raw.rstrip())
        if a<b: result.append((a,b))
    return result


def select(pid, source, offsets, scores, neighbor_threshold):
    """Keep fixed anchors; include only adjacent segments meeting the policy score."""
    anchors=sorted((i for i,s in enumerate(scores) if s>=.5), key=lambda i:(-scores[i],i))[:3]
    intervals=[]
    for i in anchors:
        a,b=i,i+1
        if neighbor_threshold is not None:
            if i>0 and scores[i-1]>=neighbor_threshold:a-=1
            if i+1<len(scores) and scores[i+1]>=neighbor_threshold:b+=1
        intervals.append((a,b,float(scores[i])))
    merged=[]
    for a,b,score in sorted(intervals):
        if merged and a<merged[-1][1]:
            old=merged[-1];merged[-1]=(old[0],max(old[1],b),max(old[2],score))
        else:merged.append((a,b,score))
    spans=[]
    for first,after,score in merged:
        a,b=offsets[first][0],offsets[after-1][1];text=source[a:b]
        spans.append(EvidenceSpan(pid,tuple(range(first,after)),a,b,text,score,f'{pid}:abstract:{a}:{b}:{hashlib.sha256(text.encode()).hexdigest()[:12]}'))
    return sorted(spans,key=lambda s:(-s.score,s.start))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','mps'],default='cpu');args=ap.parse_args()
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    torch.set_num_threads(4)
    out=Path(__file__).resolve().parent/'train-context';out.mkdir(exist_ok=True)
    data=ROOT/'data/scifact';model=ROOT/'.cache/ranking/semantic-MiniLM-L6-v2'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=json.loads((model/'download_manifest.json').read_text())
    assert manifest['revision']=='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
    for name,h in manifest['files_sha256'].items():assert digest(model/name)==h
    docs={str(r['doc_id']):r for r in map(json.loads,(data/'corpus.jsonl').open())}
    splits={s:{str(r['id']):r for r in map(json.loads,(data/f'claims_{s}.jsonl').open())} for s in ['train','dev']}
    assert not set(splits['train'])&set(splits['dev'])
    train_ids=sorted(splits['train'],key=lambda i:hashlib.sha256(i.encode()).hexdigest())[:200]
    splits['train']={i:splits['train'][i] for i in train_ids}
    text_overlap=set(c['claim'] for c in splits['train'].values())&set(c['claim'] for c in splits['dev'].values())
    # Fix the grid, objective and sample before any development scoring.
    protocol={'status':'running','scope':'Gold-paper selection diagnostic; no retrieval/NLI inference. Already-inspected dev, not untouched test.','training_ids':train_ids,'policies':POLICIES,'objective':'Maximize train positive-only micro evidence F1; ties prefer precision then baseline then higher neighbor threshold.','source_sha256':{n:digest(data/n) for n in ['corpus.jsonl','claims_train.jsonl','claims_dev.jsonl']},'model':manifest,'device':args.device,'dtype':'float64' if args.device=='cpu' else 'float32','exact_claim_text_overlap':sorted(text_overlap),'limitations':['Gold evidence documents supplied; excludes retrieval errors and cannot estimate end-to-end performance.','Claims with empty annotated evidence have no gold-paper inputs and are excluded from selection metrics.','Train/dev claim IDs disjoint; related claims and documents may still overlap.','Encoder max sequence length 256; not verifier accuracy or inference latency.']}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    sources={pid:' '.join(docs[pid]['abstract']) for split in splits.values() for c in split.values() for pid in c['evidence']}
    offsets={pid:segments(source) for pid,source in sources.items()}
    texts=list(dict.fromkeys([c['claim'] for split in splits.values() for c in split.values()]+[source[a:b] for pid,source in sources.items() for a,b in offsets[pid]]))
    key=hashlib.sha256(json.dumps({'texts':texts,'revision':manifest['revision'],'device':args.device,'dtype':protocol['dtype']},sort_keys=True).encode()).hexdigest()
    cache=ROOT/'.cache/ranking/context-selection';cache.mkdir(exist_ok=True)
    array=cache/f'{key}.npy'
    if array.exists():vectors=np.load(array)
    else:
        encoder=SentenceTransformer(str(model),device=args.device,local_files_only=True,trust_remote_code=False)
        encoder.max_seq_length=256
        if args.device=='cpu':encoder.double()
        print('Encoding',len(texts),'unique texts',flush=True)
        vectors=encoder.encode(texts,batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
        np.save(array,vectors)
    assert vectors.shape==(len(texts),384) and np.isfinite(vectors).all()
    lookup=dict(zip(texts,vectors))
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'.cache/ranking/verifier-tokenizer'),local_files_only=True)
    protocol['packages']={name:importlib.metadata.version(name) for name in ['torch','sentence-transformers','transformers','numpy','tokenizers']}
    protocol['tokenizer_files_sha256']={p.name:digest(p) for p in (ROOT/'.cache/ranking/verifier-tokenizer').iterdir() if p.is_file()}
    def measure(split, policy):
        name,threshold=policy;rows=[];start=perf_counter()
        for cid,c in split.items():
            if not c['evidence']:continue
            recovered=set();partial=set();rejected=0;tokens=0;passages=0;complete=0
            for pid,evs in c['evidence'].items():
                source=sources[pid];bounds=offsets[pid]
                scores=np.array([lookup[source[a:b]]@lookup[c['claim']] for a,b in bounds])
                spans=select(pid,source,bounds,scores,threshold)
                assert all(source[s.start:s.end]==s.text for s in spans)
                decision=enforce_verifier_budget(c['claim'],spans,tokenizer)
                rejected+=len(decision.rejected);tokens+=sum(x['pair_tokens'] for x in decision.checked if x['accepted']);passages+=len(decision.accepted)
                whole,fragment=sentence_recovery(docs[pid]['abstract'],decision.accepted)
                recovered.update((pid,i) for i in whole);partial.update((pid,i) for i in fragment)
                complete+=any(set(ev['sentences'])<=whole for ev in evs)
            gold={(pid,i) for pid,evs in c['evidence'].items() for ev in evs for i in ev['sentences']}
            rows.append({'claim_id':cid,'evidence':evidence_score(len(recovered&gold),len(recovered-gold)+len(partial),len(gold-recovered)),'complete_rationale_docs':complete,'gold_rationale_docs':len(c['evidence']),'rejected_spans':rejected,'accepted_pair_tokens':tokens,'accepted_passages':passages})
        return {'policy':name,'positive_claims':len(rows),'excluded_no_gold_papers':len(split)-len(rows),'micro':evidence_score(*(sum(r['evidence'][k] for r in rows) for k in ['tp','fp','fn'])),'complete_rationale_docs':sum(r['complete_rationale_docs'] for r in rows),'gold_rationale_docs':sum(r['gold_rationale_docs'] for r in rows),'rejected_spans':sum(r['rejected_spans'] for r in rows),'accepted_pair_tokens':sum(r['accepted_pair_tokens'] for r in rows),'accepted_passages':sum(r['accepted_passages'] for r in rows),'selection_and_tokenization_seconds':perf_counter()-start},rows
    train=[]
    for policy in POLICIES:
        summary,_=measure(splits['train'],policy);train.append(summary);print('train',summary,flush=True)
    winner=max(train,key=lambda r:(r['micro']['f1'],r['micro']['precision'],r['policy']=='baseline',dict(POLICIES)[r['policy']] or -1))['policy']
    (out/'frozen-choice.json').write_text(json.dumps({'selected_policy':winner,'train_results':train,'selection_objective':protocol['objective']},indent=2)+'\n')
    # Freeze selection before comparing all predeclared development ablations.
    dev=[]
    with (out/'dev-cases.jsonl').open('w') as f:
        for policy in POLICIES:
            summary,rows=measure(splits['dev'],policy);dev.append(summary)
            for row in rows:f.write(json.dumps({'policy':policy[0],**row})+'\n')
            print('dev',summary,flush=True)
    (out/'dev-summary.json').write_text(json.dumps({'train_selected_policy':winner,'results':dev},indent=2)+'\n')
    protocol.update(status='complete',vectors_sha256=digest(array),script_sha256=digest(Path(__file__)))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')

if __name__=='__main__':main()
