"""Freeze guards from actual cached200-claim training predictions before dev evaluation."""
import json,hashlib
from evaluation.datasets import read_reference_predictions, read_scifact_file
from pathlib import Path
from evaluation.context_study.policy import POLICIES, chosen_arm
from evaluation.validation.metrics import classification, evidence_score
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/context_study_20261111'
REV='ba8cdd6e7766306447913d8087b23f62195fa05a'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'TRAIN_FROZEN.json'
    if target.exists():raise ValueError('Training record already frozen; do not overwrite after dev inspection')
    raw=read_reference_predictions('context_train')
    (OUT/'owner_train_cases.jsonl').write_bytes(raw)
    training=read_scifact_file('claims_train.jsonl')
    training_path=OUT/'claims_train.jsonl'
    if training_path.exists() and training_path.read_bytes()!=training:
        raise ValueError('Existing training data differs from the pinned public release')
    if not training_path.exists():training_path.write_bytes(training)
    rows=[json.loads(s) for s in raw.decode().splitlines()]
    arms={p:{r['claim_id']:r for r in rows if r['policy']==p and r['depth']==5} for p in ['baseline','bounded_abstract']}
    assert len(arms['baseline'])==200 and set(arms['baseline'])==set(arms['bounded_abstract'])
    summaries=[]
    for policy in POLICIES:
        selected=[arms[chosen_arm(policy,r['prediction'])][cid] for cid,r in arms['baseline'].items()]
        metrics=classification(selected);ev=[r['evidence'] for r in selected if r['gold']!='INSUFFICIENT']
        summaries.append({'policy':policy,'all':metrics,'evidence':evidence_score(*(sum(r[k] for r in ev) for k in ['tp','fp','fn']))})
    base=summaries[0]['all'];eligible=[r for r in summaries if r['all']['accuracy']>=base['accuracy'] and r['all']['per_class']['CONTRADICTED']['recall']>=base['per_class']['CONTRADICTED']['recall']]
    winner=max(eligible,key=lambda r:(r['all']['accuracy'],r['all']['macro_f1'],r['policy']=='baseline'))
    record={'date':'2026-11-11','owner_source':REV,'training_sha256':hashlib.sha256(raw).hexdigest(),'train_ids':list(arms['baseline']),
      'predeclared_policies':list(POLICIES),'objective':'Max train all-claim accuracy, then macroF1, subject to accuracy and contradiction recall >= original baseline. Tie prefers baseline. No dev metric selects policy.',
      'selected_policy':winner['policy'],'training_results':summaries,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('policy.py'),Path(__file__).with_name('context_adapter.py')]},
      'execution':'Training uses actual pinned-owner cached predictions, no fabricated inference. Local CPU replay will be reported separately; numerical equivalence is not assumed.',
      'scope':'Exploratory response to already inspected dev errors. TrainingIDs were chosen by upstream SHA256 before evaluation; related claims/papers can overlap. Not untouched test.',
      'context':'No additional paper gate; whole original abstract if512-token evidence/claim budget fits, otherwise original baseline spans. Guards short-circuit on focused verdict and never read gold.',
      'evidence_accounting':'On contextual routes, report union of focused/context source intervals, including auxiliary inference. On short-circuit routes only baseline was required. Full input text is retained, not just anchor highlights.',
      'dev_arms':['baseline','gate_only_0.65','context_only','context_and_gate_0.65','train_selected_guard'],
      'default_change':'None; research runner only. Retain all regressions.'}
    target.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('TRAIN SELECTED',winner['policy'],flush=True)
    for r in summaries:print(r['policy'],'accuracy',r['all']['accuracy'],'contradiction recall',r['all']['per_class']['CONTRADICTED']['recall'],flush=True)
if __name__=='__main__':main()
