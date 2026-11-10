"""Verify the frozen paper-gate comparison and export paired case diagnostics."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score,f1_score

HERE=Path(__file__).resolve().parent
OUT=HERE/'relevance-gate-results'
s=json.loads((OUT/'dev-summary.json').read_text())
rows=[json.loads(l) for l in (OUT/'dev-cases.jsonl').open()]
assert len(rows)==1800 and len({(r['condition'],r['depth'],r['claim_id']) for r in rows})==1800
for name,m in s['conditions'].items():
    condition,k=name.split('@');rr=[r for r in rows if r['condition']==condition and r['depth']==int(k)]
    y=[r['gold'] for r in rr];pred=[r['prediction'] for r in rr]
    assert len(rr)==300 and abs(accuracy_score(y,pred)-m['all']['accuracy'])<1e-12
    assert abs(f1_score(y,pred,labels=sorted(set(y)),average='macro',zero_division=0)-m['all']['macro_f1'])<1e-12
base={r['claim_id']:r for r in rows if r['condition']=='baseline' and r['depth']==5}
new={r['claim_id']:r for r in rows if r['condition']=='train_selected' and r['depth']==5}
ids=sorted(base)
delta=np.array([int(new[i]['gold']==new[i]['prediction'])-int(base[i]['gold']==base[i]['prediction']) for i in ids])
rng=np.random.default_rng(20261110);draws=[float(delta[rng.integers(0,len(ids),len(ids))].mean()) for _ in range(2000)]
paired={'accuracy_delta':float(delta.mean()),'paired_claim_bootstrap_95_percentile_interval':list(np.quantile(draws,[.025,.975])),'repeats':2000,'seed':20261110,'transitions':dict(Counter(('correct' if base[i]['gold']==base[i]['prediction'] else 'wrong')+' -> '+('correct' if new[i]['gold']==new[i]['prediction'] else 'wrong') for i in ids)),'limitations':'Exploratory previously inspected dev; claim bootstrap ignores related-claim/document dependence and multiple comparisons.'}
changes=[{'claim_id':i,'claim':base[i]['claim'],'gold':base[i]['gold'],'baseline':base[i]['prediction'],'candidate':new[i]['prediction'],'baseline_correct':base[i]['gold']==base[i]['prediction'],'candidate_correct':new[i]['gold']==new[i]['prediction']} for i in ids if base[i]['prediction']!=new[i]['prediction']]
(OUT/'paired-comparison.json').write_text(json.dumps({'paired':paired,'changes':changes},indent=2)+'\n')
lines=['# Train-selected paper gate and verifier context','','## Result','','The policy and paper threshold were selected using 200 training claims before applying this comparison to development labels. The chosen policy is `'+s['selected_policy']+'`, paper relevance threshold '+str(s['selected_gate'])+', fixed primary scope Top-5.','','All 300 SciFact development claims are included. This is inspected development data, not a new unseen test.','','| Condition | Top-K candidate scope | Accuracy | Macro-F1 | Positive-only accuracy | Evidence precision | Evidence recall | Evidence F1 | Complete rationale pairs |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for name,m in s['conditions'].items():
 c,k=name.split('@');e=m['evidence_micro_positive_only'];lines.append(f"| {c} | {k} | {m['all']['accuracy']:.4f} | {m['all']['macro_f1']:.4f} | {m['positive']['accuracy']:.4f} | {e['precision']:.4f} | {e['recall']:.4f} | {e['f1']:.4f} | {m['complete_rationale_docs']}/{m['gold_rationale_docs']} |")
lines+=['',f"Primary Top-5 accuracy delta: {paired['accuracy_delta']:+.4f}; descriptive paired bootstrap interval {paired['paired_claim_bootstrap_95_percentile_interval']}.",'',f"Paired correctness transitions: `{paired['transitions']}`.",'','## What changed','','Keep the same RRF rankings and pretrained verifier. Whole-abstract context is permitted only after the existing relevance gate and only when the exact premise/claim pair fits 512 tokens; long abstracts fall back to original selected spans. An additional training-selected paper gate removes papers whose maximum accepted-passage relevance is below the frozen threshold. No new papers are injected, no source text or claim is rewritten, no model weights or NLI threshold changed. The same owner/frontend aggregation runs on the retained paper scope.','','The gate uses cosine relevance, not a calibrated truth probability. It trades evidence coverage for fewer weakly matching papers. Full-abstract context also includes unannotated sentences: evidence precision/F1 can decrease even when verdict accuracy rises. The table reports actual verifier-input evidence scores; it does not hide extra context by scoring only the short anchors.','','Top-10/20 results are ablations, not development-selected deployment choices. The primary candidate remains the train-frozen Top-5 configuration.','','## Case review','']
for cid in ['137','1163','113','1274']:
 r=new[cid];b=base[cid];lines.append(f"- Claim {cid}: gold {r['gold']}; baseline {b['prediction']}; candidate {r['prediction']}.")
lines+=['','Every changed verdict, including regressions, is recorded in `relevance-gate-results/paired-comparison.json`. Full input/output traces are retained in the source evaluation run.','','## Backend use','','The reusable adapter is `ranking_evaluation.contextual_evidence.BoundedAbstractSelector`. Load `ranking_evaluation/experiments/contextual-accuracy-train-frozen.json`, wrap the original configured selector and actual verifier tokenizer, and explicitly pass `min_paper_score`. Persist `context_configuration` with the output. The default selector/frontend was not changed. See `ranking_evaluation/CONTEXTUAL_EVIDENCE.md`.','','## Verification and limits','','All 1800 condition rows are unique and complete. Accuracy and Macro-F1 were independently verified with scikit-learn. Training trace hashes are frozen and checked before dev replay. Filtering/reaggregation uses real cached NLI predictions: changing paper scope never changes the individual evidence-premise/original-claim inputs. No extra inference is required to remove papers.','','Related claims or papers may cross train/dev IDs. The descriptive paired bootstrap is not a guarantee of independent-test gains. Source benchmark negatives reflect absent annotations, not a universal claim about all literature. These experiments evaluate SciFact verdicts; the other two datasets do not provide equivalent verdict labels.','','No frontend modifications or git push were performed. Independent test evaluation and teammate integration remain separate tasks.']
(HERE/'RELEVANCE_GATE_RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(paired,indent=2))
print('Independently checked all six conditions.')
