"""Train-only paper relevance gating; reaggregate real cached predictions, no new NLI.

A stricter paper gate removes weakly matching papers before verdict aggregation.
It never adds papers or uses development labels to choose the gate.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from explanation_frontend._upstream.claim_verifier.verifier import ClaimVerifier
from explanation_frontend._upstream.claim_verifier.schemas import VerificationResponse
from explanation_frontend.pipeline import safe_aggregate
from explanation_frontend.evaluation.validation.metrics import classification,evidence_score
from explanation_frontend.evaluation.validation.scifact import sentence_recovery,gold_label

HERE=Path(__file__).resolve().parent
GRID=[(p,t) for p in ['baseline','neighbors_0.6','bounded_abstract'] for t in [.5,.55,.6,.65,.7]]

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def evaluate(trace,claim,docs,threshold,depth):
    response=VerificationResponse.model_validate(trace['response'])
    pool=trace['scope_paper_ids'][:depth]
    scores={pid:max((p.score for p in response.evidence_passages if p.paper_id==pid),default=-1.) for pid in pool}
    kept=[pid for pid in pool if scores[pid]>=threshold]
    agg=safe_aggregate(object.__new__(ClaimVerifier),response,kept)
    prediction=agg.aggregated_verdict.value if hasattr(agg.aggregated_verdict,'value') else agg.aggregated_verdict
    recovered=set();partial=set()
    for pid in kept:
        source=' '.join(docs[pid]['abstract']);spans=[]
        for p in response.evidence_passages:
            if p.paper_id!=pid:continue
            parts=p.passage_id.split(':');a,b=int(parts[-3]),int(parts[-2]);assert source[a:b]==p.text
            spans.append(SimpleNamespace(start=a,end=b))
        whole,fragment=sentence_recovery(docs[pid]['abstract'],spans)
        recovered.update((pid,i) for i in whole);partial.update((pid,i) for i in fragment)
    gold={(pid,i) for pid,evs in claim['evidence'].items() for ev in evs for i in ev['sentences']}
    return {'claim_id':str(claim['id']),'claim':claim['claim'],'gold':gold_label(claim),'prediction':prediction,'depth':depth,'paper_gate':threshold,'policy':trace['policy'],'selected_paper_ids':kept,'candidate_paper_ids':pool,'paper_scores':scores,'evidence':evidence_score(len(recovered&gold),len(recovered-gold)+len(partial),len(gold-recovered)),'complete_rationale_docs':sum(any({(pid,i) for i in e['sentences']}<=recovered for e in evs) for pid,evs in claim['evidence'].items()),'gold_rationale_docs':len(claim['evidence'])}

def summary(rows):
    positive=[r for r in rows if r['gold']!='INSUFFICIENT']
    return {'all':classification(rows),'positive':classification(positive),'evidence_micro_positive_only':evidence_score(*(sum(r['evidence'][k] for r in positive) for k in ['tp','fp','fn'])),'complete_rationale_docs':sum(r['complete_rationale_docs'] for r in positive),'gold_rationale_docs':sum(r['gold_rationale_docs'] for r in positive),'kept_paper_count':sum(len(r['selected_paper_ids']) for r in rows)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['train','dev']);ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
    run=args.run;out=HERE/'relevance-gate-results';out.mkdir(exist_ok=True)
    docs={str(r['doc_id']):r for r in map(json.loads,(ROOT/'data/scifact/corpus.jsonl').open())}
    protocol=json.loads((run/'protocol.json').read_text())
    train_ids=protocol['train_ids']
    if args.command=='train':
        if (out/'train-frozen-choice.json').exists():raise ValueError('Frozen choice already exists; do not overwrite after dev inspection')
        claims={str(r['id']):r for r in map(json.loads,(ROOT/'data/scifact/claims_train.jsonl').open())}
        fixed={'grid':GRID,'train_ids':train_ids,'objective':'Highest train all-claim Accuracy, then Macro-F1; ties prefer baseline policy and lower paper gate. Depth5 fixed.','method':'Remove papers below maximum accepted-passage relevance, then unchanged owner/frontend aggregation. No NLI retraining or new model calls.','script_sha256':digest(Path(__file__)),'source_run':str(run),'source_protocol_sha256_at_training':digest(run/'protocol.json'),'scope':'Exploratory train-selected calibration on inspected development benchmark; no unseen-test claim.'}
        (out/'protocol.json').write_text(json.dumps(fixed,indent=2)+'\n')
        results=[];trace_hashes={}
        for policy,gate in GRID:
            rows=[]
            for cid in train_ids:
                path=run/f'train-traces/{cid}-{policy}.json';trace_hashes[str(path.relative_to(run))]=digest(path)
                rows.append(evaluate(json.loads(path.read_text()),claims[cid],docs,gate,5))
            results.append({'policy':policy,'paper_gate':gate,'metrics':summary(rows)})
        winner=max(results,key=lambda r:(r['metrics']['all']['accuracy'],r['metrics']['all']['macro_f1'],r['policy']=='baseline',-r['paper_gate']))
        frozen={'selected_policy':winner['policy'],'paper_gate':winner['paper_gate'],'depth':5,'train_summary':results,'train_trace_sha256':trace_hashes,'protocol_sha256':digest(out/'protocol.json')}
        (out/'train-frozen-choice.json').write_text(json.dumps(frozen,indent=2)+'\n')
        print('TRAIN CHOICE',winner['policy'],winner['paper_gate'],winner['metrics']['all']['accuracy'],winner['metrics']['all']['macro_f1'])
    else:
        if protocol['status']!='complete':raise ValueError('Wait for the full source experiment to complete')
        frozen=json.loads((out/'train-frozen-choice.json').read_text())
        assert digest(out/'protocol.json')==frozen['protocol_sha256']
        for name,h in frozen['train_trace_sha256'].items():assert digest(run/name)==h
        claims={str(r['id']):r for r in map(json.loads,(ROOT/'data/scifact/claims_dev.jsonl').open())}
        assert not set(train_ids)&set(claims)
        rows=[];conditions={}
        for name,policy,gate in [('baseline','baseline',.5),('train_selected',frozen['selected_policy'],frozen['paper_gate'])]:
            for depth in [5,10,20]:
                group=[]
                for cid,claim in claims.items():
                    trace=json.loads((run/f'dev-traces/{cid}-{policy}.json').read_text());r=evaluate(trace,claim,docs,gate,depth);r['condition']=name;group.append(r)
                conditions[f'{name}@{depth}']=summary(group);rows.extend(group)
        (out/'dev-cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (out/'dev-summary.json').write_text(json.dumps({'frozen_choice_sha256':digest(out/'train-frozen-choice.json'),'selected_policy':frozen['selected_policy'],'selected_gate':frozen['paper_gate'],'conditions':conditions},indent=2)+'\n')
        print(json.dumps({k:(v['all']['accuracy'],v['all']['macro_f1']) for k,v in conditions.items()},indent=2))
if __name__=='__main__':main()
