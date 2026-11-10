"""Independently validate and export a completed full accuracy experiment."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import numpy as np
from sklearn.metrics import accuracy_score,f1_score

HERE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();run=args.run
    protocol=json.loads((run/'protocol.json').read_text())
    assert protocol['status']=='complete'
    choice=json.loads((run/'train-frozen-choice.json').read_text())
    summaries=json.loads((run/'dev-summary.json').read_text())['conditions']
    rows=[json.loads(l) for l in (run/'dev-cases.jsonl').open()]
    assert len(rows)==2700 and len({(r['claim_id'],r['policy'],r['depth']) for r in rows})==2700
    for name,summary in summaries.items():
        policy,depth=name.split('@');rr=[r for r in rows if r['policy']==policy and r['depth']==int(depth)]
        assert len(rr)==300
        y=[r['gold'] for r in rr];p=[r['prediction'] for r in rr]
        assert abs(accuracy_score(y,p)-summary['all']['accuracy'])<1e-12
        assert abs(f1_score(y,p,labels=sorted(set(y)),average='macro',zero_division=0)-summary['all']['macro_f1'])<1e-12
    groups={(r['policy'],r['depth']):{} for r in rows}
    for r in rows:groups[(r['policy'],r['depth'])][r['claim_id']]=r
    baseline=groups[('baseline',5)];candidate=groups[(choice['policy'],5)]
    ids=sorted(baseline)
    delta=np.array([int(candidate[i]['prediction']==candidate[i]['gold'])-int(baseline[i]['prediction']==baseline[i]['gold']) for i in ids])
    rng=np.random.default_rng(20261110)
    boot=np.array([delta[rng.integers(0,len(ids),len(ids))].mean() for _ in range(2000)])
    changes=[{'claim_id':i,'claim':baseline[i]['claim'],'gold':baseline[i]['gold'],'baseline':baseline[i]['prediction'],'candidate':candidate[i]['prediction'],'baseline_correct':baseline[i]['prediction']==baseline[i]['gold'],'candidate_correct':candidate[i]['prediction']==candidate[i]['gold']} for i in ids if baseline[i]['prediction']!=candidate[i]['prediction']]
    uncertainty={'train_selected_policy':choice['policy'],'depth':5,'accuracy_delta':float(delta.mean()),'paired_claim_bootstrap_95_percentile_interval':list(np.quantile(boot,[.025,.975])),'seed':20261110,'repeats':2000,'transitions':dict(Counter(('correct' if baseline[i]['prediction']==baseline[i]['gold'] else 'wrong')+' -> '+('correct' if candidate[i]['prediction']==candidate[i]['gold'] else 'wrong') for i in ids)),'limitations':'Exploratory inspected dev; related claim/document dependence and multiple comparisons not adjusted. No untouched-test guarantee.'}
    target=HERE/'accuracy-full-results';target.mkdir(exist_ok=True)
    for name in ['protocol.json','retrieval-summary.json','train-frozen-choice.json','dev-summary.json','baseline-agreement.json','train-cases.jsonl','dev-cases.jsonl']:shutil.copy2(run/name,target/name)
    (target/'paired-comparison.json').write_text(json.dumps({'comparison':uncertainty,'verdict_changes':changes},indent=2)+'\n')
    lines=['# Full-corpus ranking and evidence accuracy experiment','','## Result','','This is a completed real-model experiment on the fixed SciFact corpus. Train configuration selection used 200 deterministically sampled training claims; all 300 development claims were evaluated. Development data was already inspected and is not an untouched test set.','','The training objective selected `'+choice['policy']+'` at depth five. All predeclared policy/depth ablations are reported; development performance did not select that policy.','','| Policy | Papers verified | Accuracy | Macro-F1 | Positive-only accuracy | Evidence F1 | Complete rationale document coverage |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,s in summaries.items():
        policy,depth=name.split('@')
        lines.append(f"| {policy} | {depth} | {s['all']['accuracy']:.4f} | {s['all']['macro_f1']:.4f} | {s['positive']['accuracy']:.4f} | {s['evidence_micro_positive_only']['f1']:.4f} | {s['complete_rationale_docs']}/{s['gold_rationale_docs']} |")
    lines+=['','Train-selected depth-five accuracy change versus baseline: '+str(uncertainty['accuracy_delta'])+'. Descriptive paired claim bootstrap 95% interval: '+str(uncertainty['paired_claim_bootstrap_95_percentile_interval'])+'.','','Paired correctness transitions: `'+str(uncertainty['transitions'])+'`.','','## Scope and interpretation','','RRF uses BM25 Porter stemming and dense MiniLM retrieval, full 5183-paper corpus, top100 per branch, RRF k=60. No gold papers are injected. The training sample is the first 200 SHA-256-ordered claim IDs, independent of labels. All train claims, including those without positive evidence, participate in the accuracy objective.','','Baseline: three relevance-ranked sentence segments at cosine >=0.5. Selective context: add immediate neighbors with cosine >=0.6. Bounded abstract: if the same relevance gate finds any baseline anchor, use the exact whole abstract when the evidence-claim pair fits 512 tokenizer tokens; otherwise retain baseline passages. No text truncation or gold-based passage choice. Thresholds and policies were fixed before this run.','','The fixed pretrained verifier, threshold 0.5 and owner/frontend aggregation are unchanged. MPS float32 results are recorded separately from CPU. The actual passage predictions are cached by exact claim and text, then re-aggregated at Top-5/10/20. Per-paper verification is independent of other papers; therefore prefix re-aggregation does not require repeated neural inference. No per-condition latency comparison is inferred from cached execution.','','Longer passages may improve verdict decisions while lowering complete-sentence evidence precision by including unannotated material. Gold absence is benchmark INSUFFICIENT, not a universal scientific verdict. Correctness and evidence quality must be reported separately.','','## Verification','','All 2700 development condition rows are present and unique. Accuracy and Macro-F1 were independently recomputed with scikit-learn for every condition. Model/data hashes, numerical environment, training IDs, prediction cache accounting and source provenance are recorded in the exported protocol. Compare `baseline-agreement.json` with the original frontend run; reconstruction is explicitly checked rather than assumed.','','Full passage-level traces and reusable neural predictions remain under `'+str(run)+'`. Compact train/dev case rows, results and every changed verdict are exported under `accuracy-full-results/`. No models or vector arrays are included in the handoff.','','This report does not modify production configuration or frontend behavior. Any adoption should use the train-selected depth-five policy and preserve these baseline results; an independent test is still needed for a generalization claim.']
    (HERE/'FULL_ACCURACY_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(uncertainty,indent=2))
    print('All 9 conditions independently verified; compact artifacts exported.')
if __name__=='__main__':main()
