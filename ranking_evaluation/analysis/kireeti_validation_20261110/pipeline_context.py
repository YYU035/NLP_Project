"""Prepare fixed-policy evidence comparisons and run real pinned NLI offline.

Full runs require all-300 frozen rankings. Replay samples are explicitly labeled.
No inference or verdict metric is fabricated when model weights are unavailable.
"""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from ranking_evaluation.analysis.kireeti_validation_20261110.train_context import segments,select
from ranking_evaluation.verifier_budget import enforce_verifier_budget
from explanation_frontend.evaluation.validation.scifact import sentence_recovery,gold_label
from explanation_frontend.evaluation.validation.metrics import evidence_score,classification,ranking_metrics

HERE=Path(__file__).resolve().parent
RESULTS=ROOT/'explanation_frontend/evaluation/validation/results'
NLI_REV='b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7'
NLI_HASH='c03cd208bf920b4fbbb182a0535859a566e8acc3477d2b536bf87b769978524b'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_rankings(raw,claims,docs):
    ranks=raw['rrf']
    if set(ranks)!=set(claims):raise ValueError('Frozen rankings must cover exactly all development claims')
    for cid,ids in ranks.items():
        if not isinstance(ids,list) or len(ids)<5 or len(ids)!=len(set(ids)) or set(ids)-set(docs):
            raise ValueError(f'Invalid ranking for claim {cid}')
    return {cid:ids[:5] for cid,ids in ranks.items()}


def prepare(args):
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    data=ROOT/'data/scifact'
    expected=json.loads((RESULTS/'BASELINE.json').read_text())['input_sha256']
    for name,digest in expected.items():
        if sha(data/name)!=digest:raise ValueError('Frozen data mismatch: '+name)
    docs={str(r['doc_id']):r for r in map(json.loads,(data/'corpus.jsonl').open())}
    claims={str(r['id']):r for r in map(json.loads,(data/'claims_dev.jsonl').open())}
    if args.rankings:
        scopes=validate_rankings(json.loads(args.rankings.read_text()),claims,docs)
        scope='full_300_frozen_rrf_top5';ranking_hash=sha(args.rankings)
    elif args.replay_sample:
        scopes={p.stem.removeprefix('claim-'):json.loads(p.read_text())['selected_paper_ids'] for p in (RESULTS/'replays').glob('claim-*.json')}
        scope='selected_15_archived_replays_not_representative';ranking_hash=None
    else:raise ValueError('Supply --rankings for full evaluation or explicitly choose --replay-sample')
    frozen=HERE/'train-context/frozen-choice.json'
    if json.loads(frozen.read_text())['selected_policy']!='neighbors_0.6':raise ValueError('Unexpected train-frozen policy')
    model=args.encoder
    manifest=json.loads((model/'download_manifest.json').read_text())
    if manifest['revision']!='1110a243fdf4706b3f48f1d95db1a4f5529b4d41':raise ValueError('Wrong semantic encoder revision')
    for name,h in manifest['files_sha256'].items():
        if sha(model/name)!=h:raise ValueError('Encoder file hash mismatch: '+name)
    torch.set_num_threads(4)
    encoder=SentenceTransformer(str(model.resolve()),device=args.device,local_files_only=True,trust_remote_code=False)
    encoder.max_seq_length=256
    if args.device=='cpu':encoder.double()
    tokenizer=AutoTokenizer.from_pretrained(str(args.tokenizer.resolve()),local_files_only=True)
    sources={pid:' '.join(docs[pid]['abstract']) for ids in scopes.values() for pid in ids}
    bounds={pid:segments(text) for pid,text in sources.items()}
    texts=list(dict.fromkeys([claims[cid]['claim'] for cid in scopes]+[text[a:b] for pid,text in sources.items() for a,b in bounds[pid]]))
    vectors=encoder.encode(texts,batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
    if not np.isfinite(vectors).all():raise ValueError('Nonfinite embeddings')
    lookup=dict(zip(texts,vectors));rows=[]
    for cid,pids in scopes.items():
        c=claims[cid]
        for policy,threshold in [('baseline',None),('neighbors_0.6',.6)]:
            passages={};checks=[];recovered=set();partials=set()
            for pid in pids:
                text=sources[pid]
                scores=[lookup[text[a:b]]@lookup[c['claim']] for a,b in bounds[pid]]
                proposed=select(pid,text,bounds[pid],scores,threshold)
                decision=enforce_verifier_budget(c['claim'],proposed,tokenizer)
                passages[pid]=[{'passage_id':s.passage_id,'paper_id':pid,'section':'abstract','text':s.text,'score':s.score} for s in decision.accepted]
                checks.extend(decision.checked)
                whole,part=sentence_recovery(docs[pid]['abstract'],decision.accepted)
                recovered.update((pid,i) for i in whole);partials.update((pid,i) for i in part)
            gold={(pid,i) for pid,evs in c['evidence'].items() for ev in evs for i in ev['sentences']}
            complete=sum(any({(pid,i) for i in e['sentences']}<=recovered for e in evs) for pid,evs in c['evidence'].items())
            rows.append({'claim_id':cid,'claim':c['claim'],'gold':gold_label(c),'policy':policy,'papers':[{'paper_id':pid,'title':docs[pid]['title']} for pid in pids],'passages':passages,'budget_checks':checks,'evidence':evidence_score(len(recovered&gold),len(recovered-gold)+len(partials),len(gold-recovered)),'complete_rationale_docs':complete,'gold_rationale_docs':len(c['evidence']),'retrieval':ranking_metrics(pids,c['evidence'])})
    args.output.mkdir(parents=True,exist_ok=False)
    path=args.output/'prepared.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    metadata={'scope':scope,'rows':len(rows),'claims':len(scopes),'prepared_sha256':sha(path),'rankings_sha256':ranking_hash,'train_choice_sha256':sha(frozen),'source_sha256':expected,'encoder_revision':manifest['revision'],'encoder_device':args.device,'encoder_dtype':'float64' if args.device=='cpu' else 'float32','tokenizer_files_sha256':{p.name:sha(p) for p in args.tokenizer.iterdir() if p.is_file()},'status':'evidence_prepared_no_nli_inference','script_sha256':sha(Path(__file__))}
    (args.output/'preparation.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


def infer(args):
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    from explanation_frontend._upstream.claim_verifier.verifier import ClaimVerifier
    from explanation_frontend._upstream.claim_verifier.schemas import EvidencePassage
    from explanation_frontend.pipeline import safe_aggregate
    metadata=json.loads((args.prepared.parent/'preparation.json').read_text())
    if sha(args.prepared)!=metadata['prepared_sha256']:raise ValueError('Prepared input hash mismatch')
    if sha(args.nli_model/'model.safetensors')!=NLI_HASH:raise ValueError('Pinned verifier weights mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(42);torch.set_num_threads(4)
    if args.device=='cuda':torch.backends.cuda.matmul.allow_tf32=False
    v=object.__new__(ClaimVerifier)
    v.checkpoint='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli';v.revision=NLI_REV
    v.max_length=512;v.threshold=.5;v.seed=42;v.device=args.device
    v.tokenizer=AutoTokenizer.from_pretrained(str(args.nli_model.resolve()),local_files_only=True)
    v.model=AutoModelForSequenceClassification.from_pretrained(str(args.nli_model.resolve()),local_files_only=True,trust_remote_code=False,dtype=torch.float32).to(args.device).eval()
    v.label_mapping=v._build_label_mapping()
    rows=[]
    synchronize=lambda:torch.cuda.synchronize() if args.device=='cuda' else (torch.mps.synchronize() if args.device=='mps' else None)
    with (args.output/'predictions.jsonl').open('w') as stream:
        for entry in map(json.loads,args.prepared.open()):
            try:
                passages={pid:[EvidencePassage.model_validate(p) for p in ps] for pid,ps in entry['passages'].items()}
                # Recheck the actual inference tokenizer; never trust preparation counts alone.
                for ps in passages.values():
                    for p in ps:
                        if len(v.tokenizer.encode(p.text,entry['claim'],add_special_tokens=True,truncation=False))>512:raise ValueError('Inference pair exceeds token budget')
                synchronize();start=perf_counter()
                response=v.verify(entry['claim'],entry['papers'],passages)
                agg=safe_aggregate(v,response,[p['paper_id'] for p in entry['papers']])
                synchronize();seconds=perf_counter()-start
                prediction=agg.aggregated_verdict.value if hasattr(agg.aggregated_verdict,'value') else agg.aggregated_verdict
                if any(not torch.isfinite(torch.tensor(p.confidence)) for p in response.per_passage_labels):raise ValueError('Nonfinite NLI confidence')
                row={**entry,'prediction':prediction,'seconds':seconds,'status':'ok','response':response.model_dump(mode='json'),'aggregation':agg.model_dump(mode='json')}
            except Exception as exc:
                row={**entry,'prediction':None,'status':'failed','error':f'{type(exc).__name__}: {exc}'}
            rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            print(entry['claim_id'],entry['policy'],row['prediction'],row['status'],flush=True)
    summary={}
    for policy in ['baseline','neighbors_0.6']:
        group=[r for r in rows if r['policy']==policy];positive=[r for r in group if r['gold']!='INSUFFICIENT']
        summary[policy]={'classification_all':classification(group),'classification_positive':classification(positive),'evidence_micro_positive_only':evidence_score(*(sum(r['evidence'][k] for r in positive) for k in ['tp','fp','fn'])),'complete_rationale_docs':sum(r['complete_rationale_docs'] for r in positive),'gold_rationale_docs':sum(r['gold_rationale_docs'] for r in positive),'operational_failures':sum(r['status']=='failed' for r in group),'measured_verification_seconds':sum(r.get('seconds',0) for r in group)}
    result={'preparation':metadata,'nli_device':args.device,'nli_dtype':'float32','nli_revision':NLI_REV,'nli_weights_sha256':NLI_HASH,'nli_config_sha256':sha(args.nli_model/'config.json'),'packages':{n:importlib.metadata.version(n) for n in ['torch','transformers','numpy']},'conditions':summary,'limitations':['Offline fixed-corpus evaluation, not web discovery.','Inspected dev; not unseen test.','Latency is verification/aggregation only, excluding preparation, retrieval and model load; no cross-device historical latency comparison.','Both policies rerun on the same device; GPU floating point may differ from archived CPU results.','Selected replay scope is diagnostic only when full frozen rankings are not supplied.']}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--rankings',type=Path);p.add_argument('--replay-sample',action='store_true');p.add_argument('--encoder',type=Path,default=ROOT/'.cache/ranking/semantic-MiniLM-L6-v2');p.add_argument('--tokenizer',type=Path,default=ROOT/'.cache/ranking/verifier-tokenizer');p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('infer');p.add_argument('--prepared',type=Path,required=True);p.add_argument('--nli-model',type=Path,required=True);p.add_argument('--device',choices=['cpu','mps','cuda'],default='cuda');p.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.command=='prepare':prepare(args)
    else:infer(args)
if __name__=='__main__':main()
