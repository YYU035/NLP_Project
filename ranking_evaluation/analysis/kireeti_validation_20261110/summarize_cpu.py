"""Report actual CPU predictions and matched-case transitions after inference."""
import csv
import json
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RUN=HERE/'cpu-replay-nli'
s=json.loads((RUN/'summary.json').read_text())
rows=[json.loads(l) for l in (RUN/'predictions.jsonl').read_text().splitlines()]
groups={p:{r['claim_id']:r for r in rows if r['policy']==p} for p in ['baseline','neighbors_0.6']}
a,b=groups.values()
assert set(a)==set(b) and len(a)==15
original={r['claim_id']:r for r in csv.DictReader((ROOT/'explanation_frontend/evaluation/validation/results/test_results.csv').open(encoding='utf-8-sig')) if r['condition']=='retrieved_papers_production_selector'}
transitions=Counter(('correct' if a[i]['prediction']==a[i]['gold'] else 'wrong')+' -> '+('correct' if b[i]['prediction']==b[i]['gold'] else 'wrong') for i in a)
changes=[{'claim_id':i,'claim':a[i]['claim'],'gold':a[i]['gold'],'baseline':a[i]['prediction'],'candidate':b[i]['prediction']} for i in a if a[i]['prediction']!=b[i]['prediction']]
comparison={'transitions':dict(transitions),'changed_verdicts':changes,'baseline_verdicts_matching_archived':sum(a[i]['prediction']==original[i]['prediction'] for i in a),'claims':15}
(RUN/'paired_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
lines=['# Local CPU verifier comparison','','Actual pinned DeBERTa inference on all 15 prepared replay claims, with both baseline and candidate rerun on CPU float32. Semantic selection inputs were prepared using MPS float32; this is a CPU verifier run, not an entirely CPU preprocessing experiment.','','These are selected diagnostic cases enriched for failures. Results cannot be generalized to the full 300-claim development set or an unseen test set.','','| Condition | Accuracy | Macro-F1 | Positive-only accuracy | Verifier/aggregation seconds | Operational failures |','| --- | ---: | ---: | ---: | ---: | ---: |']
for p in ['baseline','neighbors_0.6']:
 c=s['conditions'][p]
 lines.append(f"| {p} | {c['classification_all']['accuracy']:.4f} | {c['classification_all']['macro_f1']:.4f} | {c['classification_positive']['accuracy']:.4f} | {c['measured_verification_seconds']:.2f} | {c['operational_failures']} |")
lines+=['',f"Matched-case transitions: `{dict(transitions)}`.",'',f"New CPU baseline verdicts matching the archived predictions: {comparison['baseline_verdicts_matching_archived']}/15.",'','## Verdict changes','']
for c in changes:lines.append(f"- Claim {c['claim_id']}: gold {c['gold']}; {c['baseline']} -> {c['candidate']}. {c['claim']}")
if not changes:lines.append('No verdict changes between conditions.')
lines+=['','## Interpretation and reproducibility','','Evidence scores alone cannot establish verdict improvement. Inspect the actual paired transitions above before deciding whether to change any configuration. No production configuration or frontend was changed.','','Latency includes actual verification and aggregation for each condition, without cross-condition prediction caching; excludes network download, model loading, retrieval, selection preparation and frontend rendering. CPU warmup/order effects are not controlled, so these are operational timings, not a rigorous latency benchmark. Claims are processed baseline then candidate.','','The model weight checksum, device, dtype, package versions, prepared input hash and condition metrics are recorded in `summary.json`. Full paper and passage predictions are in `predictions.jsonl`. The prepared input retains the original claims, paper scope, source evidence text and tokenizer budget checks.','','Reproduce from repository root with a new output directory using `pipeline_context.py infer --device cpu`, the existing prepared JSONL and pinned local NLI directory. No model threshold, weights, claim text or evidence labels were changed.']
(HERE/'CPU_REPLAY_RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(comparison,indent=2))
for p,c in s['conditions'].items():print(p,c['classification_all']['accuracy'],c['classification_all']['macro_f1'],c['measured_verification_seconds'],c['operational_failures'])
