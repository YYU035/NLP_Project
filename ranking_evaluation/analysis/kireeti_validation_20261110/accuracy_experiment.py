"""Train-selected, source-preserving evidence policies with real full-corpus ranking/NLI.

Offline diagnostic evaluation only. No frontend changes or development tuning.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'live_search/src'))
from ranking_evaluation.analysis.kireeti_validation_20261110.train_context import segments,select
from ranking_evaluation.evidence_selection import EvidenceSpan
from ranking_evaluation.verifier_budget import enforce_verifier_budget
from explanation_frontend.evaluation.validation.metrics import classification,evidence_score,ranking_metrics
from explanation_frontend.evaluation.validation.scifact import gold_label,sentence_recovery
from explanation_frontend.evaluation.validation.retrieval import rank_dataset
from explanation_frontend.pipeline import safe_aggregate
from explanation_frontend._upstream.claim_verifier.verifier import ClaimVerifier
from explanation_frontend._upstream.claim_verifier.schemas import EvidencePassage,PassageVerdict
from ranking_evaluation.analysis.kireeti_validation_20261110.pipeline_context import NLI_HASH,NLI_REV
from ranking_evaluation.analysis.kireeti_validation_20261110.batch_verification import classify_batched

HERE=Path(__file__).resolve().parent
POLICIES=['baseline','neighbors_0.6','bounded_abstract']

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def spans_for(policy,pid,source,bounds,scores,tokenizer,claim):
    baseline=select(pid,source,bounds,scores,None)
    if policy=='baseline':return baseline,'baseline'
    if policy=='neighbors_0.6':return select(pid,source,bounds,scores,.6),'neighbors_0.6'
    if policy!='bounded_abstract':raise ValueError('Unknown policy')
    if not baseline:return [],'below_anchor_threshold'
    candidate=EvidenceSpan(pid,tuple(range(len(bounds))),0,len(source),source,float(max(scores)),f'{pid}:abstract:0:{len(source)}:{hashlib.sha256(source.encode()).hexdigest()[:12]}')
    if len(tokenizer.encode(source,claim,add_special_tokens=True,truncation=False))<=512:
        return [candidate],'whole_abstract_with_relevance_gate'
    return baseline,'whole_abstract_exceeds_budget_fallback_to_baseline'

def measure(rows):
    positive=[r for r in rows if r['gold']!='INSUFFICIENT']
    return {'all':classification(rows),'positive':classification(positive),'evidence_micro_positive_only':evidence_score(*(sum(r['evidence'][k] for r in positive) for k in ['tp','fp','fn'])),'complete_rationale_docs':sum(r['complete_rationale_docs'] for r in positive),'gold_rationale_docs':sum(r['gold_rationale_docs'] for r in positive),'rejected_spans':sum(r['rejected_spans'] for r in rows),'no_positive_selected_sentence_count':sum(r['selected_sentences'] for r in rows if r['gold']=='INSUFFICIENT'),'fallback_papers':sum(r['fallback_papers'] for r in rows)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--encoder-device',choices=['cpu','mps'],default='mps');ap.add_argument('--nli-device',choices=['cpu','mps'],default='cpu');ap.add_argument('--batch-size',type=int,default=1);ap.add_argument('--seed-cache',type=Path);args=ap.parse_args()
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    torch.set_num_threads(4);torch.manual_seed(42)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    data=ROOT/'data/scifact';model=ROOT/'.cache/ranking/semantic-MiniLM-L6-v2';nli=ROOT/'.cache/ranking/verifier-model'
    manifest=json.loads((model/'download_manifest.json').read_text())
    for name,h in manifest['files_sha256'].items():
        if sha(model/name)!=h:raise ValueError('Model integrity mismatch '+name)
    if sha(nli/'model.safetensors')!=NLI_HASH:raise ValueError('NLI integrity mismatch')
    baseline=json.loads((ROOT/'explanation_frontend/evaluation/validation/results/BASELINE.json').read_text())
    for name,h in baseline['input_sha256'].items():
        if sha(data/name)!=h:raise ValueError('Data integrity mismatch '+name)
    docs={str(r['doc_id']):r for r in map(json.loads,(data/'corpus.jsonl').open())}
    train_all={str(r['id']):r for r in map(json.loads,(data/'claims_train.jsonl').open())}
    train_ids=sorted(train_all,key=lambda i:hashlib.sha256(i.encode()).hexdigest())[:200]
    train={i:train_all[i] for i in train_ids}
    dev={str(r['id']):r for r in map(json.loads,(data/'claims_dev.jsonl').open())}
    assert not set(train)&set(dev)
    all_claims={**train,**dev}
    # The grid, training objective and depth comparisons are fixed before inference.
    protocol={'status':'running','train_ids':train_ids,'train_total':len(train),'dev_total':len(dev),'training_paper_depth':5,'development_depths':[5,10,20],'policies':POLICIES,'objective':'Highest train all-claim accuracy at depth5; ties prefer Macro-F1 then unchanged baseline. No dev policy selection.','scope':'Offline full5183 corpus retrieval; no gold paper injection. Train-selected exploratory development diagnostic, not unseen test.','encoder_device':args.encoder_device,'encoder_dtype':'float64' if args.encoder_device=='cpu' else 'float32','nli_device':args.nli_device,'nli_dtype':'float32','nli_revision':NLI_REV,'nli_weights_sha256':NLI_HASH,'encoder_revision':manifest['revision'],'retrieval_weights_note':'Existing pinned MiniLM assets; pytorch_model.bin hash also matches the frontend production baseline. Reconstructed ranks must be checked against archived metrics/replays, not assumed identical across numerical environments.','source_sha256':{n:sha(data/n) for n in ['corpus.jsonl','claims_train.jsonl','claims_dev.jsonl']},'script_sha256':sha(Path(__file__)),'packages':{n:importlib.metadata.version(n) for n in ['torch','transformers','sentence-transformers','numpy']},'limitations':['Already inspected dev, no untouched-test improvement claim.','Some related claims/documents can cross train/dev IDs.','Cached identical claim/text NLI predictions reused; no per-condition latency claim.','Whole abstracts retain original source text, selected only by text relevance; gold annotations used only for scoring.','Bounded abstract uses one whole abstract if pair <=512 tokens; otherwise explicit baseline fallback. No gold-dependent truncation.']}
    p=out/'protocol.json'
    protocol['batch_size']=args.batch_size
    protocol['batch_forward_source_sha256']=sha(HERE/'batch_verification.py')
    if args.seed_cache:
        origin=json.loads((args.seed_cache.parent/'protocol.json').read_text())
        if origin['nli_device']!=args.nli_device or origin['nli_weights_sha256']!=NLI_HASH:raise ValueError('Seed cache model/device mismatch')
        protocol['seed_cache_sha256']=sha(args.seed_cache)
        protocol['cache_note']='Exact claim/text owner predictions from the prior sequential run on the same weights/device; newly missing pairs use the validated batch forward pass.'
    if p.exists():
        old=json.loads(p.read_text())
        for key in ['source_sha256','script_sha256','encoder_device','nli_device','policies','train_ids']:
            if old[key]!=protocol[key]:raise ValueError('Resume protocol mismatch: '+key)
    p.write_text(json.dumps(protocol,indent=2)+'\n')
    encoder=SentenceTransformer(str(model.resolve()),device=args.encoder_device,local_files_only=True,trust_remote_code=False)
    encoder.max_seq_length=256
    if args.encoder_device=='cpu':encoder.double()
    rankings_path=out/'rankings.json'
    if rankings_path.exists():ranks=json.loads(rankings_path.read_text())
    else:
        queries={i:{'text':c['claim'],'candidate_ids':list(docs)} for i,c in all_claims.items()}
        print('Ranking full corpus for',len(queries),'queries',flush=True)
        ranks=rank_dataset(docs,queries,encoder,stemmer='porter',model_key='accuracy-context-'+manifest['revision']+'-'+args.encoder_device+'-'+protocol['encoder_dtype'],branch_depth=100,progress=lambda i,n:print('Ranked',i,'/',n,flush=True))
        ranks={m:{cid:ids[:100] for cid,ids in rr.items()} for m,rr in ranks.items()}
        rankings_path.write_text(json.dumps(ranks)+'\n')
    assert set(ranks['rrf'])==set(all_claims)
    retrieval_summary={}
    for method,rr in ranks.items():
        eligible=[ranking_metrics(rr[i],c['evidence']) for i,c in dev.items() if c['evidence']]
        retrieval_summary[method]={k:float(np.mean([r[k] for r in eligible])) for k in eligible[0]}
    (out/'retrieval-summary.json').write_text(json.dumps(retrieval_summary,indent=2)+'\n')
    print('Dev retrieval',retrieval_summary,flush=True)
    sources={pid:' '.join(docs[pid]['abstract']) for cid in all_claims for pid in ranks['rrf'][cid][:20]}
    bounds={pid:segments(s) for pid,s in sources.items()}
    texts=list(dict.fromkeys([c['claim'] for c in all_claims.values()]+[s[a:b] for pid,s in sources.items() for a,b in bounds[pid]]))
    array=out/'selection-vectors.npy'
    if array.exists():vectors=np.load(array)
    else:
        print('Encoding selection texts',len(texts),flush=True)
        vectors=encoder.encode(texts,batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False);np.save(array,vectors)
    if vectors.shape!=(len(texts),384) or not np.isfinite(vectors).all():raise ValueError('Invalid vectors')
    lookup=dict(zip(texts,vectors));del encoder
    v=object.__new__(ClaimVerifier);v.checkpoint='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli';v.revision=NLI_REV;v.max_length=512;v.threshold=.5;v.seed=42;v.device=args.nli_device
    v.tokenizer=AutoTokenizer.from_pretrained(str(nli.resolve()),local_files_only=True)
    v.model=AutoModelForSequenceClassification.from_pretrained(str(nli.resolve()),local_files_only=True,trust_remote_code=False,dtype=torch.float32).to(v.device).eval();v.label_mapping=v._build_label_mapping()
    original=v._classify_passage;cache_path=out/'nli-cache.jsonl';cache={};new_calls=0;neural_seconds=0.
    if cache_path.exists():
        for line in cache_path.open():
            r=json.loads(line);cache[r['key']]=r['prediction']
    cache_stream=cache_path.open('a')
    if args.seed_cache and not cache:
        for line in args.seed_cache.open():
            r=json.loads(line);cache[r['key']]=r['prediction'];cache_stream.write(line)
        cache_stream.flush()
    def cached(claim,passage):
        nonlocal new_calls,neural_seconds
        key=hashlib.sha256(json.dumps([claim,passage.text]).encode()).hexdigest()
        if key not in cache:
            start=perf_counter();value=original(claim,passage);neural_seconds+=perf_counter()-start
            if not np.isfinite(value.confidence) or not all(np.isfinite(x) for x in value.raw_probabilities.values()):raise ValueError('Nonfinite NLI output')
            cache[key]=value.model_dump(mode='json');cache_stream.write(json.dumps({'key':key,'prediction':cache[key]})+'\n');cache_stream.flush();new_calls+=1
        return PassageVerdict.model_validate(cache[key]).model_copy(update={'paper_id':passage.paper_id,'passage_id':passage.passage_id})
    v._classify_passage=cached
    def run_split(split,name,policies,depths):
        nonlocal new_calls,neural_seconds
        path=out/f'{name}-cases.jsonl';rows=[];done=set()
        if path.exists():
            rows=[json.loads(l) for l in path.open()];done={(r['claim_id'],r['policy'],r['depth']) for r in rows}
        with path.open('a') as stream:
            for index,(cid,c) in enumerate(split.items()):
                for policy in policies:
                    if all((cid,policy,k) in done for k in depths):continue
                    top=ranks['rrf'][cid][:max(depths)];spans_by_doc={};passages={};checks={};fallback={}
                    for pid in top:
                        source=sources[pid];offsets=bounds[pid]
                        scores=[float(lookup[source[a:b]]@lookup[c['claim']]) for a,b in offsets]
                        proposed,reason=spans_for(policy,pid,source,offsets,scores,v.tokenizer,c['claim'])
                        assert all(source[s.start:s.end]==s.text for s in proposed)
                        decision=enforce_verifier_budget(c['claim'],proposed,v.tokenizer)
                        spans_by_doc[pid]=decision.accepted;checks[pid]=list(decision.checked);fallback[pid]='fallback' in reason
                        passages[pid]=[EvidencePassage(passage_id=s.passage_id,paper_id=pid,section='abstract',text=s.text,score=s.score) for s in decision.accepted]
                    if args.batch_size>1:
                        missing={}
                        for ps in passages.values():
                            for passage in ps:
                                key=hashlib.sha256(json.dumps([c['claim'],passage.text]).encode()).hexdigest()
                                if key not in cache:missing[key]=(c['claim'],passage)
                        if missing:
                            start=perf_counter();values=classify_batched(v,list(missing.values()),batch_size=args.batch_size);neural_seconds+=perf_counter()-start
                            for key,value in zip(missing,values):
                                cache[key]=value.model_dump(mode='json');cache_stream.write(json.dumps({'key':key,'prediction':cache[key]})+'\n');new_calls+=1
                            cache_stream.flush()
                    response=v.verify(c['claim'],[{'paper_id':pid,'title':docs[pid]['title']} for pid in top],passages)
                    for depth in depths:
                        if (cid,policy,depth) in done:continue
                        pids=top[:depth];agg=safe_aggregate(v,response,pids);prediction=agg.aggregated_verdict.value if hasattr(agg.aggregated_verdict,'value') else agg.aggregated_verdict
                        recovered=set();partial=set()
                        for pid in pids:
                            whole,part=sentence_recovery(docs[pid]['abstract'],spans_by_doc[pid]);recovered.update((pid,i) for i in whole);partial.update((pid,i) for i in part)
                        gold={(pid,i) for pid,evs in c['evidence'].items() for ev in evs for i in ev['sentences']}
                        complete=sum(any({(pid,i) for i in e['sentences']}<=recovered for e in evs) for pid,evs in c['evidence'].items())
                        row={'claim_id':cid,'claim':c['claim'],'gold':gold_label(c),'policy':policy,'depth':depth,'prediction':prediction,'evidence':evidence_score(len(recovered&gold),len(recovered-gold)+len(partial),len(gold-recovered)),'selected_sentences':len(recovered),'complete_rationale_docs':complete,'gold_rationale_docs':len(c['evidence']),'rejected_spans':sum(not x['accepted'] for pid in pids for x in checks[pid]),'fallback_papers':sum(fallback[pid] for pid in pids),'selected_paper_ids':pids}
                        rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                    # Retain the actual source passages and predictions for failure review.
                    traces=out/f'{name}-traces';traces.mkdir(exist_ok=True)
                    (traces/f'{cid}-{policy}.json').write_text(json.dumps({'claim_id':cid,'policy':policy,'scope_paper_ids':top,'response':response.model_dump(mode='json'),'budget_checks':checks},indent=2)+'\n')
                if index%10==0 or index+1==len(split):print(name,index+1,'/',len(split),'new NLI calls',new_calls,'cache size',len(cache),flush=True)
        return rows
    train_rows=run_split(train,'train',POLICIES,[5])
    train_summary={policy:measure([r for r in train_rows if r['policy']==policy]) for policy in POLICIES}
    winner=max(POLICIES,key=lambda policy:(train_summary[policy]['all']['accuracy'],train_summary[policy]['all']['macro_f1'],policy=='baseline'))
    choice={'policy':winner,'depth':5,'train_summary':train_summary,'objective':protocol['objective']}
    (out/'train-frozen-choice.json').write_text(json.dumps(choice,indent=2)+'\n');print('Train choice',winner,train_summary,flush=True)
    # All preregistered policies/depths are reported; the train winner stays fixed.
    dev_rows=run_split(dev,'dev',POLICIES,[5,10,20])
    summary={f'{policy}@{depth}':measure([r for r in dev_rows if r['policy']==policy and r['depth']==depth]) for policy in POLICIES for depth in [5,10,20]}
    (out/'dev-summary.json').write_text(json.dumps({'train_selected_policy':winner,'train_selected_depth':5,'conditions':summary},indent=2)+'\n')
    archived={r['claim_id']:r for r in csv.DictReader((ROOT/'explanation_frontend/evaluation/validation/results/test_results.csv').open(encoding='utf-8-sig')) if r['condition']=='retrieved_papers_production_selector'}
    baseline_rows=[r for r in dev_rows if r['policy']=='baseline' and r['depth']==5]
    agreement={'verdict_matches':sum(r['prediction']==archived[r['claim_id']]['prediction'] for r in baseline_rows),'evidence_score_matches':sum(r['evidence']==json.loads(archived[r['claim_id']]['evidence']) for r in baseline_rows),'claims':len(baseline_rows)}
    (out/'baseline-agreement.json').write_text(json.dumps(agreement,indent=2)+'\n')
    protocol.update(status='complete',rankings_sha256=sha(rankings_path),selection_vectors_sha256=sha(array),unique_cached_pairs=len(cache),new_neural_calls_this_invocation=new_calls,new_neural_seconds_this_invocation=neural_seconds)
    p.write_text(json.dumps(protocol,indent=2)+'\n');cache_stream.close()
    print('COMPLETE',agreement,json.dumps(summary),flush=True)

if __name__=='__main__':main()
