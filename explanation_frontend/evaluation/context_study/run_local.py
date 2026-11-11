"""Actual CPU context ablations with a training-frozen guard and honest input accounting."""
import json,hashlib,time
from pathlib import Path
from types import SimpleNamespace
from .freeze_train import ROOT,OUT
from .policy import chosen_arm
from .context_adapter import BoundedAbstractSelector
from ..validation.freeze import freeze
from ..validation.metrics import classification,evidence_score,bootstrap
from ..validation.scifact import sentence_recovery,gold_label

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def native_from_trace(row,claim):
    from ..._upstream.claim_verifier.schemas import VerificationResponse
    data=row['response'];settings=data['model_settings']
    return VerificationResponse.model_validate({'claim':claim,'per_paper_verdicts':[{**p,'evidence_type':{'abstract_only':'abstract','full_text':'full_text'}[p['evidence_type']]} for p in data['per_paper_verdicts'].values()],
      'per_passage_labels':list(data['per_passage_labels'].values()),'processing_failures':[],
      'evidence_passages':data['evidence_passages'],'model_settings':{**settings,'tokenizer_max_length':settings['max_length']}})

def source_intervals(response,docs):
    spans={}
    for p in response.evidence_passages:
        parts=p.passage_id.split(':');a,b=int(parts[-3]),int(parts[-2]);source=' '.join(docs[p.paper_id]['abstract'])
        if source[a:b]!=p.text:raise ValueError('Exact-source span mismatch: '+p.passage_id)
        spans.setdefault(p.paper_id,[]).append(SimpleNamespace(start=a,end=b))
    return spans

def evidence_metrics(claim,docs,spans):
    recovered=set();partial=set()
    for pid,ss in spans.items():
        whole,fragments=sentence_recovery(docs[pid]['abstract'],ss)
        recovered.update((pid,i) for i in whole);partial.update((pid,i) for i in fragments)
    gold={(pid,i) for pid,evs in claim.get('evidence',{}).items() for ev in evs for i in ev['sentences']}
    complete=sum(any({(pid,i) for i in e['sentences']}<=recovered for e in evs) for pid,evs in claim.get('evidence',{}).items())
    return {'evidence':evidence_score(len(gold&recovered),len(recovered-gold)+len(partial),len(gold-recovered)),
      'complete_rationale_docs':complete,'gold_rationale_docs':len(claim.get('evidence',{})),'partial_fragments':len(partial),'input_sentences':len(recovered)}

def main():
    import torch
    torch.set_num_threads(6)
    baseline=freeze();frozen=read(OUT/'TRAIN_FROZEN.json')
    from ...pipeline import local_verifier,safe_aggregate
    from ...semantic_integration import local_selector
    from ..._upstream.person3.verifier_budget import enforce_verifier_budget
    from ..._upstream.person3.semantic_evidence import person4_passages
    from ..._upstream.claim_verifier.schemas import EvidencePassage,PassageVerdict
    from ...models import Paper,make_context
    from ...person4 import Person4Adapter
    from ...templates import build_template
    from ...validation import validate_explanation
    from ...export import export_snapshot,import_snapshot
    old=ROOT/'outputs/validation/scifact_full';original_rows={str(r['claim_id']):r for r in read(old/'case_rows.json') if r['condition']=='retrieved_papers_production_selector'}
    claims=list(map(json.loads,(ROOT/'outputs/xai/data/claims_dev.jsonl').read_text(encoding='utf-8').splitlines()))
    docs={str(r['doc_id']):r for r in map(json.loads,(ROOT/'outputs/xai/data/corpus.jsonl').read_text(encoding='utf-8').splitlines())}
    traces={str(r['claim_id']):r for r in map(json.loads,(old/'case_details.jsonl').read_text(encoding='utf-8').splitlines()) if r['condition']=='retrieved_papers_production_selector'}
    rankings=read(old/'rankings.json')['rrf'];v=local_verifier();selector=BoundedAbstractSelector(local_selector(),v.tokenizer)
    original=v._classify_passage;seed={};calls=[0];hits=[0]
    for claim in claims:
        n=native_from_trace(traces[str(claim['id'])],claim['claim']);labels={p.passage_id:p for p in n.per_passage_labels}
        for passage in n.evidence_passages:seed[(claim['claim'],passage.text)]=labels[passage.passage_id]
    cache=OUT/'nli_cache';cache.mkdir(exist_ok=True);trace_dir=OUT/'local_traces';trace_dir.mkdir(exist_ok=True);replays=OUT/'replays';replays.mkdir(exist_ok=True)
    def cached(claim,passage):
        key=(claim,passage.text);path=cache/(hashlib.sha256(json.dumps(key,ensure_ascii=False).encode()).hexdigest()+'.json')
        if key in seed:r=seed[key];hits[0]+=1
        elif path.exists():r=PassageVerdict.model_validate(read(path));hits[0]+=1
        else:
            r=original(claim,passage);calls[0]+=1
            if r.operational_status.value!='OK' or r.truncated:raise ValueError('Unusable local inference')
            path.write_text(r.model_dump_json(),encoding='utf-8')
        return r.model_copy(update={'passage_id':passage.passage_id,'paper_id':passage.paper_id})
    v._classify_passage=cached;rows=[];started=time.perf_counter()
    try:
        for index,claim in enumerate(claims):
            cid=str(claim['id']);pids=rankings[cid][:5];base=native_from_trace(traces[cid],claim['claim'])
            base_agg=safe_aggregate(v,base,pids);assert base_agg.aggregated_verdict.value==original_rows[cid]['prediction']
            paths={};abstained=[];checks=[]
            for pid in pids:
                source=' '.join(docs[pid]['abstract']);spans=selector.select_abstract(claim['claim'],pid,source)
                decision=enforce_verifier_budget(claim['claim'],spans,v.tokenizer);checks.extend(decision.checked)
                if decision.rejected:raise ValueError('Unexpected context budget rejection; retain and diagnose instead of dropping')
                if not spans:abstained.append(pid)
                paths[pid]=[EvidencePassage.model_validate(p) for p in person4_passages(decision.accepted)]
            context=v.verify(claim['claim'],[{'paper_id':pid,'title':docs[pid]['title']} for pid in pids],paths)
            if any(p.operational_status.value!='OK' or p.truncated for p in context.per_passage_labels):raise ValueError('Context inference failed')
            base_spans=source_intervals(base,docs);context_spans=source_intervals(context,docs)
            # Explicit union includes auxiliary baseline input; never score just a short anchor.
            union={pid:base_spans.get(pid,[])+context_spans.get(pid,[]) for pid in set(base_spans)|set(context_spans)}
            guarded_arm=chosen_arm(frozen['selected_policy'],base_agg.aggregated_verdict.value)
            conditions=[('baseline',base,pids,base_spans),('gate_only_0.65',base,[pid for pid in pids if max((p.score for p in base.evidence_passages if p.paper_id==pid),default=-1)>=.65],base_spans),
              ('context_only',context,pids,context_spans),('context_and_gate_0.65',context,[pid for pid in pids if max((p.score for p in context.evidence_passages if p.paper_id==pid),default=-1)>=.65],context_spans),
              ('train_selected_guard',base if guarded_arm=='baseline' else context,pids,base_spans if guarded_arm=='baseline' else union)]
            for name,response,kept,spans in conditions:
                agg=safe_aggregate(v,response,kept);prediction=agg.aggregated_verdict.value
                measured=evidence_metrics(claim,docs,{pid:ss for pid,ss in spans.items() if pid in kept})
                entry={'claim_id':cid,'claim':claim['claim'],'gold':gold_label(claim),'condition':name,'prediction':prediction,'correct':prediction==gold_label(claim),
                  'selected_paper_ids':kept,'guard_arm':guarded_arm if name=='train_selected_guard' else None,**measured}
                rows.append(entry)
                if name=='train_selected_guard' and (index<5 or (prediction!=original_rows[cid]['prediction'] and index%7==0)):
                    metadata=[Paper(paper_id=pid,title=docs[pid]['title'],source_status='unverified',source_version=baseline['input_sha256']['corpus.jsonl']) for pid in pids]
                    adapter=Person4Adapter(lambda response,subset_of_paper_ids:safe_aggregate(v,response,subset_of_paper_ids))
                    empty=[pid for pid in pids if not any(p.paper_id==pid for p in response.evidence_passages)]
                    snapshot=adapter.snapshot(response,metadata,mode='cached',abstained_paper_ids=empty)
                    data=snapshot.model_dump(mode='json');data['filters']['research_policy']=frozen['selected_policy'];data['filters']['decision_arm']=guarded_arm
                    data['filters']['auxiliary_inference']='Focused route always required. Context route may additionally be required; both actual input traces retained in local_traces. Replay displays the decisive arm.'
                    from ...models import RunSnapshot
                    snapshot=RunSnapshot.model_validate(data);ctx=make_context(snapshot,pids,snapshot.provisional);ex=build_template(ctx)
                    if not validate_explanation(ex,ctx).displayable:raise ValueError('Grounding failure')
                    bundle=export_snapshot(ctx,ex);assert import_snapshot(bundle).snapshot.fingerprint==snapshot.fingerprint
                    (replays/f'claim-{cid}.json').write_text(bundle,encoding='utf-8')
            (trace_dir/f'{cid}.json').write_text(json.dumps({'claim_id':cid,'paper_ids':pids,'baseline':base.model_dump(mode='json'),'context':context.model_dump(mode='json'),'guard_arm':guarded_arm,'budget_checks':checks}),encoding='utf-8')
            (OUT/'rows_progress.json').write_text(json.dumps(rows),encoding='utf-8')
            if index%10==0 or index+1==len(claims):print('Local context validation',index+1,'/300 new NLI',calls[0],'cache hits',hits[0],flush=True)
    finally:v._classify_passage=original
    result={'date':'2026-11-11','claims':300,'selected_policy':frozen['selected_policy'],'new_local_nli_calls':calls[0],'cached_passage_reuses':hits[0],'seconds':time.perf_counter()-started,'conditions':{},
      'scope':'Offline full-corpus RRF top5; no gold injection. Original CPU focused predictions reused exactly; new context predictions run locally. All five arms are predeclared. Dev is already inspected; no unseen-test or generalization claim.',
      'missing_raw_probabilities':'Original focused cache contains measured label/confidence but not raw class probabilities. These remain empty, not reconstructed or fabricated.',
      'evidence':'Guard contextual routes score the union of focused and contextual inputs. Gate arms report their retained scope. Auxiliary calls made to run ablations are not hidden; no latency comparison.'}
    for name in [c[0] for c in conditions]:
        group=[r for r in rows if r['condition']==name];positive=[r for r in group if r['gold']!='INSUFFICIENT']
        result['conditions'][name]={'all':classification(group),'accuracy_ci':bootstrap([int(r['correct']) for r in group]),'evidence':evidence_score(*(sum(r['evidence'][k] for r in positive) for k in ['tp','fp','fn'])),
          'complete_rationale_docs':sum(r['complete_rationale_docs'] for r in positive),'gold_rationale_docs':sum(r['gold_rationale_docs'] for r in positive)}
    (OUT/'case_rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:{'accuracy':r['all']['accuracy'],'contradiction_recall':r['all']['per_class']['CONTRADICTED']['recall']} for k,r in result['conditions'].items()}),flush=True)
if __name__=='__main__':main()
