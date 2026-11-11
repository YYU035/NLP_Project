"""Apply frozen train gate to the complete local dev traces; never choose it on dev."""
import json
from types import SimpleNamespace
from .freeze_train import ROOT,OUT
from .run_local import source_intervals,evidence_metrics
from .policy import chosen_arm
from ..validation.metrics import classification,evidence_score,bootstrap

def main():
    from ...pipeline import safe_aggregate
    from ..._upstream.claim_verifier.verifier import ClaimVerifier
    from ..._upstream.claim_verifier.schemas import VerificationResponse
    frozen=json.loads((OUT/'GATE_TRAIN_FROZEN.json').read_text(encoding='utf-8'));gate=frozen['selected_gate']
    evaluator=object.__new__(ClaimVerifier)
    claims=list(map(json.loads,(ROOT/'outputs/xai/data/claims_dev.jsonl').read_text(encoding='utf-8').splitlines()))
    docs={str(r['doc_id']):r for r in map(json.loads,(ROOT/'outputs/xai/data/corpus.jsonl').read_text(encoding='utf-8').splitlines())}
    base_rows={r['claim_id']:r for r in json.loads((OUT/'case_rows.json').read_text(encoding='utf-8')) if r['condition']=='baseline'}
    rows=[]
    replays=OUT/'gate_replays';replays.mkdir(exist_ok=True)
    for index,claim in enumerate(claims):
        cid=str(claim['id']);trace=json.loads((OUT/'local_traces'/f'{cid}.json').read_text(encoding='utf-8'));pids=trace['paper_ids']
        baseline=VerificationResponse.model_validate(trace['baseline']);context=VerificationResponse.model_validate(trace['context'])
        arm=chosen_arm(frozen['fixed_guard'],base_rows[cid]['prediction'])
        if arm=='baseline':
            row={**base_rows[cid],'condition':'train_selected_guard_and_gate','paper_gate':gate,'guard_arm':arm}
        else:
            kept=[pid for pid in pids if max((p.score for p in context.evidence_passages if p.paper_id==pid),default=-1)>=gate]
            aggregate=safe_aggregate(evaluator,context,kept)
            focused_spans=source_intervals(baseline,docs);context_spans=source_intervals(context,docs)
            union={pid:focused_spans.get(pid,[])+(context_spans.get(pid,[]) if pid in kept else []) for pid in set(focused_spans)|set(context_spans)}
            row={**base_rows[cid],'condition':'train_selected_guard_and_gate','paper_gate':gate,'guard_arm':arm,'prediction':aggregate.aggregated_verdict.value,'selected_paper_ids':kept,**evidence_metrics(claim,docs,union)}
            row['correct']=row['prediction']==row['gold']
        rows.append(row)
        if index<5 or (row['prediction']!=base_rows[cid]['prediction'] and index%7==0):
            from ...person4 import Person4Adapter
            from ...models import Paper,RunSnapshot,make_context
            from ...templates import build_template
            from ...validation import validate_explanation
            from ...export import export_snapshot,import_snapshot
            response=baseline if arm=='baseline' else context
            selected=pids if arm=='baseline' else kept
            adapter=Person4Adapter(lambda response,subset_of_paper_ids:safe_aggregate(evaluator,response,subset_of_paper_ids))
            metadata=[Paper(paper_id=pid,title=docs[pid]['title'],source_status='unverified') for pid in pids]
            empty=[pid for pid in pids if not any(p.paper_id==pid for p in response.evidence_passages)]
            snap=adapter.snapshot(response,metadata,mode='cached',abstained_paper_ids=empty)
            aggregate=adapter.aggregate(snap,selected)
            data=snap.model_dump(mode='json');data['filters'].update(research_policy=frozen['fixed_guard'],decision_arm=arm,context_paper_gate=str(gate),auxiliary_inputs='Focused input always required. Context applies only to uncertain focus. Full auxiliary source traces retained; replay shows decisive arm and selected scope.')
            snap=RunSnapshot.model_validate(data);ctx=make_context(snap,selected,aggregate);ex=build_template(ctx)
            if not validate_explanation(ex,ctx).displayable:raise ValueError('Gated replay grounding failed')
            bundle=export_snapshot(ctx,ex);assert import_snapshot(bundle).snapshot.fingerprint==snap.fingerprint
            (replays/f'claim-{cid}.json').write_text(bundle,encoding='utf-8')
    positive=[r for r in rows if r['gold']!='INSUFFICIENT'];base=[base_rows[r['claim_id']] for r in rows]
    result={'date':'2026-11-11','selected_gate':gate,'policy':frozen['fixed_guard'],'all':classification(rows),
      'evidence':evidence_score(*(sum(r['evidence'][k] for r in positive) for k in ['tp','fp','fn'])),
      'complete_rationale_docs':sum(r['complete_rationale_docs'] for r in positive),'gold_rationale_docs':sum(r['gold_rationale_docs'] for r in positive),
      'accuracy_ci':bootstrap([int(r['correct']) for r in rows]),'paired_accuracy_delta_ci':bootstrap([int(r['correct'])-int(b['correct']) for r,b in zip(rows,base)]),
      'focused_correct_contradictions_preserved':all(r['prediction']=='CONTRADICTED' for r,b in zip(rows,base) if b['gold']=='CONTRADICTED' and b['prediction']=='CONTRADICTED'),
      'scope':'Frozen train gate applied once to all300 already-inspected dev claims. Uses real cached/local NLI traces. No new inference for reaggregation; no gold in policy decisions. Auxiliary focused text included in evidence scoring.'}
    (OUT/'gate_dev_case_rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(OUT/'gate_dev_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'gate':gate,'accuracy':result['all']['accuracy'],'macro_f1':result['all']['macro_f1'],'contradiction_recall':result['all']['per_class']['CONTRADICTED']['recall'],'evidence_f1':result['evidence']['f1']}),flush=True)
if __name__=='__main__':main()
