"""Choose a context-only paper gate on train200, keeping the frozen decision guard."""
import json,hashlib,subprocess,time
from pathlib import Path
from types import SimpleNamespace
from .freeze_train import ROOT,OUT,REV
from .policy import chosen_arm
from .context_adapter import BoundedAbstractSelector
from .run_local import source_intervals,evidence_metrics
from ..validation.metrics import classification,evidence_score
GRID=(.5,.55,.6,.65,.7)

def main():
    import torch
    torch.set_num_threads(6)
    from ...pipeline import local_verifier,safe_aggregate
    from ...semantic_integration import local_selector
    from ..._upstream.person3.verifier_budget import enforce_verifier_budget
    from ..._upstream.person3.semantic_evidence import person4_passages
    from ..._upstream.claim_verifier.schemas import EvidencePassage,PassageVerdict
    frozen=json.loads((OUT/'TRAIN_FROZEN.json').read_text(encoding='utf-8'));target=OUT/'GATE_TRAIN_FROZEN.json'
    if target.exists():raise ValueError('Do not overwrite frozen gate')
    training_path=OUT/'claims_train.jsonl'
    expected='f4c8fa82d8bd0653a9cc8d61a6ea48c25eacea64e90af5dbf390ebb1b74372f0'
    if not training_path.exists():
        import requests,tarfile,io
        with requests.get('https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz',timeout=60,stream=True) as response:
            response.raise_for_status();chunks=[];size=0
            for chunk in response.iter_content(1024*1024):
                size+=len(chunk)
                if size>50*1024*1024:raise ValueError('Dataset archive too large')
                chunks.append(chunk)
        with tarfile.open(fileobj=io.BytesIO(b''.join(chunks)),mode='r:gz') as archive:
            matches=[m for m in archive.getmembers() if m.name.endswith('/claims_train.jsonl') and m.isfile()]
            if len(matches)!=1 or matches[0].size>10*1024*1024:raise ValueError('Invalid training member')
            source=archive.extractfile(matches[0]).read()
        if hashlib.sha256(source).hexdigest()!=expected:raise ValueError('Official training bytes differ from owner protocol')
        training_path.write_bytes(source)
    source=training_path.read_bytes()
    if hashlib.sha256(source).hexdigest()!=expected:raise ValueError('Training data hash mismatch')
    claims={str(r['id']):r for r in map(json.loads,source.decode().splitlines())}
    docs={str(r['doc_id']):r for r in map(json.loads,(ROOT/'outputs/xai/data/corpus.jsonl').read_text(encoding='utf-8').splitlines())}
    owner=[json.loads(s) for s in (OUT/'owner_train_cases.jsonl').read_text(encoding='utf-8').splitlines()]
    base={r['claim_id']:r for r in owner if r['policy']=='baseline'}
    assert set(base)==set(frozen['train_ids']);v=local_verifier();original=v._classify_passage;calls=[0];hits=[0]
    selector=BoundedAbstractSelector(local_selector(),v.tokenizer);cache=OUT/'nli_cache';cache.mkdir(exist_ok=True);trace_dir=OUT/'gate_train_traces';trace_dir.mkdir(exist_ok=True)
    protocol={'phase':'Train-selected optional gate after protected decisive predictions','grid':list(GRID),'fixed_guard':frozen['selected_policy'],
      'objective':'Max train accuracy then macroF1, subject to contradiction recall>=original baseline; tie chooses lower gate. Gate never filters protected baseline decisions.',
      'train_ids':frozen['train_ids'],'source_claims_sha256':hashlib.sha256(source).hexdigest(),'context_evidence':'Score union of ALL focused input and retained context; no hiding auxiliary evidence.',
      'scope':'Exploratory already-inspected dev benchmark; gate choice uses train labels only. No heldout claim.'}
    (OUT/'GATE_PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    def cached(claim,passage):
        key=(claim,passage.text);path=cache/(hashlib.sha256(json.dumps(key,ensure_ascii=False).encode()).hexdigest()+'.json')
        if path.exists():r=PassageVerdict.model_validate_json(path.read_text(encoding='utf-8'));hits[0]+=1
        else:
            r=original(claim,passage);calls[0]+=1
            if r.operational_status.value!='OK' or r.truncated:raise ValueError('Training NLI unavailable')
            path.write_text(r.model_dump_json(),encoding='utf-8')
        return r.model_copy(update={'passage_id':passage.passage_id,'paper_id':passage.paper_id})
    v._classify_passage=cached;rows=[];training_scope_checks=[]
    try:
        for i,cid in enumerate(frozen['train_ids']):
            claim=claims[cid];row=base[cid];arm=chosen_arm(frozen['selected_policy'],row['prediction'])
            if arm=='baseline':
                for gate in GRID:rows.append({**row,'paper_gate':gate,'condition':'protected_context_gate','guard_arm':arm})
            else:
                paths={};focus={};pids=row['selected_paper_ids']
                for pid in pids:
                    text=' '.join(docs[pid]['abstract'])
                    focused=selector.base_selector.select_abstract(claim['claim'],pid,text)
                    fb=enforce_verifier_budget(claim['claim'],focused,v.tokenizer)
                    if fb.rejected:raise ValueError('Unexpected train focus rejection')
                    focus[pid]=list(fb.accepted)
                    ss=selector.select_abstract(claim['claim'],pid,text);b=enforce_verifier_budget(claim['claim'],ss,v.tokenizer)
                    if b.rejected:raise ValueError('Unexpected train context rejection')
                    paths[pid]=[EvidencePassage.model_validate(p) for p in person4_passages(b.accepted)]
                measured=evidence_metrics(claim,docs,focus)['evidence'];training_scope_checks.append(measured==row['evidence'])
                response=v.verify(claim['claim'],[{'paper_id':pid,'title':docs[pid]['title']} for pid in pids],paths)
                if any(p.operational_status.value!='OK' for p in response.per_passage_labels):raise ValueError('Training context response failed')
                context=source_intervals(response,docs)
                for gate in GRID:
                    kept=[pid for pid in pids if max((p.score for p in response.evidence_passages if p.paper_id==pid),default=-1)>=gate]
                    agg=safe_aggregate(v,response,kept)
                    union={pid:focus.get(pid,[])+(context.get(pid,[]) if pid in kept else []) for pid in set(focus)|set(context)}
                    ev=evidence_metrics(claim,docs,union)
                    rows.append({**row,'prediction':agg.aggregated_verdict.value,'paper_gate':gate,'condition':'protected_context_gate','guard_arm':arm,'selected_paper_ids':kept,**ev})
                (trace_dir/(cid+'.json')).write_text(json.dumps({'claim_id':cid,'baseline_prediction':row['prediction'],'context':response.model_dump(mode='json'),'focus_intervals':{pid:[{'start':s.start,'end':s.end} for s in ss] for pid,ss in focus.items()}}),encoding='utf-8')
            if i%20==0 or i+1==len(base):print('Gate TRAIN',i+1,'/200','new NLI',calls[0],'cache hits',hits[0],flush=True)
    finally:v._classify_passage=original
    summaries=[];baseline=classification(list(base.values()))
    for gate in GRID:
        group=[r for r in rows if r['paper_gate']==gate];positive=[r for r in group if r['gold']!='INSUFFICIENT']
        summaries.append({'gate':gate,'all':classification(group),'evidence':evidence_score(*(sum(r['evidence'][k] for r in positive) for k in ['tp','fp','fn']))})
    eligible=[r for r in summaries if r['all']['per_class']['CONTRADICTED']['recall']>=baseline['per_class']['CONTRADICTED']['recall'] and r['all']['accuracy']>=baseline['accuracy']]
    selected=max(eligible,key=lambda r:(r['all']['accuracy'],r['all']['macro_f1'],-r['gate']))
    record={**protocol,'selected_gate':selected['gate'],'training_results':summaries,'baseline_scope_matches':{'passed':sum(training_scope_checks),'total':len(training_scope_checks)},'new_local_nli_calls':calls[0],'cached_passage_reuses':hits[0],
      'provenance':'Focused train predictions reused from pinned owner real-model cache; context evaluated locally on CPU. Raw source scopes checked against original measured train evidence.'}
    (OUT/'gate_train_case_rows.json').write_text(json.dumps(rows),encoding='utf-8');target.write_text(json.dumps(record,indent=2),encoding='utf-8')
    (ROOT/'evaluation/context_study/GATE_TRAIN_FROZEN.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('TRAIN GATE FROZEN',selected['gate'],'accuracy',selected['all']['accuracy'],'contraRecall',selected['all']['per_class']['CONTRADICTED']['recall'],flush=True)
if __name__=='__main__':main()
