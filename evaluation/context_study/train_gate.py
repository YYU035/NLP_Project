"""Choose a context-only paper gate on train200, keeping the frozen decision guard."""
import json,hashlib,time
from evaluation.datasets import read_scifact_file, read_reference_predictions
from pathlib import Path
from types import SimpleNamespace
from evaluation.context_study.freeze_train import ROOT, OUT, REV
from evaluation.context_study.policy import chosen_arm
from evaluation.context_study.context_adapter import BoundedAbstractSelector
from evaluation.context_study.run_local import source_intervals, evidence_metrics
from evaluation.validation.metrics import classification, evidence_score
GRID=(.5,.55,.6,.65,.7)

def main():
    import torch
    torch.set_num_threads(6)
    from research_verifier.pipeline import local_verifier, safe_aggregate
    from research_verifier.evidence.semantic import local_selector
    from research_verifier.evidence.budget import enforce_verifier_budget
    from research_verifier.evidence.selector import person4_passages
    from research_verifier.verification.schemas import EvidencePassage, PassageVerdict
    frozen=json.loads((OUT/'TRAIN_FROZEN.json').read_text(encoding='utf-8'));target=OUT/'GATE_TRAIN_FROZEN.json'
    if target.exists():raise ValueError('Do not overwrite frozen gate')
    training_path=OUT/'claims_train.jsonl'
    source=read_scifact_file('claims_train.jsonl')
    if training_path.exists() and training_path.read_bytes()!=source:
        raise ValueError('Existing training data differs from the pinned public release')
    if not training_path.exists():training_path.write_bytes(source)
    claims={str(r['id']):r for r in map(json.loads,source.decode().splitlines())}
    docs={str(r['doc_id']):r for r in map(json.loads,read_scifact_file('corpus.jsonl').decode('utf-8').splitlines())}
    owner=[json.loads(s) for s in read_reference_predictions('context_train').decode('utf-8').splitlines()]
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
    print('TRAIN GATE FROZEN',selected['gate'],'accuracy',selected['all']['accuracy'],'contraRecall',selected['all']['per_class']['CONTRADICTED']['recall'],flush=True)
if __name__=='__main__':main()
