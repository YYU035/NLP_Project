"""Independent accounting, frozen baseline agreement and paired micro-F1 bootstrap."""
import csv
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
rows=[json.loads(line) for line in (OUT/'train-context/dev-cases.jsonl').read_text().splitlines()]
groups={name:{r['claim_id']:r for r in rows if r['policy']==name} for name in sorted({r['policy'] for r in rows})}
base=groups['baseline']
archived={r['claim_id']:r for r in csv.DictReader((ROOT/'explanation_frontend/evaluation/validation/results/test_results.csv').open(encoding='utf-8-sig')) if r['condition']=='gold_papers_production_selector'}
assert set(base)==set(archived) and len(base)==188
for cid,r in base.items():
    assert r['evidence']==json.loads(archived[cid]['evidence'])
    assert r['complete_rationale_docs']==int(archived[cid]['complete_rationale_docs'])
selected=json.loads((OUT/'train-context/frozen-choice.json').read_text())['selected_policy']
ids=sorted(base)
a=np.array([[base[i]['evidence'][k] for k in ['tp','fp','fn']] for i in ids])
b=np.array([[groups[selected][i]['evidence'][k] for k in ['tp','fp','fn']] for i in ids])
def f1(counts):
    tp,fp,fn=counts.sum(axis=0);return float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.
rng=np.random.default_rng(20261110)
deltas=[]
for _ in range(2000):
    draw=rng.integers(0,len(ids),len(ids));deltas.append(f1(b[draw])-f1(a[draw]))
result={'baseline_claims_matching_archived_scores':188,'selected_policy':selected,'delta_micro_f1':f1(b)-f1(a),'paired_claim_bootstrap_95_percentile_interval':list(np.quantile(deltas,[.025,.975])),'repeats':2000,'seed':20261110,'limitations':'Descriptive paired claim bootstrap on inspected dev; related claim/document dependence is not modeled. No formal significance or unseen-test improvement claim.'}
(OUT/'train-context/verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
