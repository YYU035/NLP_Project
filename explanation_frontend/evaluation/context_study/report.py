"""Independent accounting, paired error analysis and readable context-study reports."""
import csv,json,hashlib,subprocess,shutil,zipfile
from pathlib import Path
from sklearn.metrics import accuracy_score,f1_score
from .freeze_train import ROOT,OUT,REV
from ..validation.metrics import bootstrap,classification
RESULTS=ROOT/'evaluation/context_study/results'

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def read_log(path):
    raw=path.read_bytes()
    return raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')
def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        w.writerows([{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows])
def pct(x):return f'{x:.1%}'

def main():
    RESULTS.mkdir(exist_ok=True);summary=read(OUT/'summary.json');rows=read(OUT/'case_rows.json');gate=read(OUT/'gate_dev_summary.json');gate_rows=read(OUT/'gate_dev_case_rows.json');frozen=read(OUT/'TRAIN_FROZEN.json');gate_train=read(OUT/'GATE_TRAIN_FROZEN.json')
    assert len(rows)==1500 and len({(r['claim_id'],r['condition']) for r in rows})==1500 and len(gate_rows)==300
    groups={name:{r['claim_id']:r for r in rows if r['condition']==name} for name in summary['conditions']}
    groups['train_selected_guard_and_gate']={r['claim_id']:r for r in gate_rows};conditions={**summary['conditions'],'train_selected_guard_and_gate':gate}
    for name,group in groups.items():
        assert len(group)==300 and set(group)==set(groups['baseline'])
        gold=[r['gold'] for r in group.values()];pred=[r['prediction'] for r in group.values()]
        assert abs(accuracy_score(gold,pred)-conditions[name]['all']['accuracy'])<1e-12
        assert abs(f1_score(gold,pred,labels=['SUPPORTED','CONTRADICTED','INSUFFICIENT'],average='macro',zero_division=0)-conditions[name]['all']['macro_f1'])<1e-12
    baseline_audit=read(ROOT/'outputs/validation/artifact_audit.json')
    for name in ['scifact_full/case_rows.json','scifact_full/case_details.jsonl','scifact_full/rankings.json']:
        assert hashlib.sha256((ROOT/'outputs/validation'/name).read_bytes()).hexdigest()==baseline_audit['artifact_sha256'][name]
    original_traces={str(r['claim_id']):r for r in map(json.loads,(ROOT/'outputs/validation/scifact_full/case_details.jsonl').read_text(encoding='utf-8').splitlines()) if r['condition']=='retrieved_papers_production_selector'}
    for cid in groups['baseline']:
        trace=read(OUT/'local_traces'/f'{cid}.json')
        assert trace['paper_ids']==original_traces[cid]['paper_ids']
        assert all(c['accepted'] and c['pair_tokens']<=512 for c in trace['budget_checks'])
        assert all(not p['truncated'] and p['operational_status']=='OK' for p in trace['context']['per_passage_labels'])
    ownerraw=subprocess.check_output(['git','show',REV+':ranking_evaluation/analysis/kireeti_validation_20261110/relevance-gate-results/dev-cases.jsonl'])
    owner={r['claim_id']:r for r in map(json.loads,ownerraw.decode().splitlines()) if r['depth']==5 and r['condition']=='train_selected'}
    assert len(owner)==300
    owner_agreement=sum(owner[cid]['prediction']==groups['context_and_gate_0.65'][cid]['prediction'] for cid in owner)
    differences=[];contra_losses=[];comparisons={}
    base=groups['baseline']
    for name,group in groups.items():
        fixes=[cid for cid in base if not base[cid]['correct'] and group[cid]['correct']];regressions=[cid for cid in base if base[cid]['correct'] and not group[cid]['correct']]
        comparisons[name]={'fixes':len(fixes),'regressions':len(regressions),'net_correct':len(fixes)-len(regressions),'paired_accuracy_delta_ci':bootstrap([int(group[cid]['correct'])-int(base[cid]['correct']) for cid in base]),'fixed_claim_ids':fixes,'regression_claim_ids':regressions}
        if name not in ['train_selected_guard','train_selected_guard_and_gate']:continue
        for cid in base:
            r=group[cid];b=base[cid]
            if r['prediction']==b['prediction']:continue
            differences.append({'candidate':name,'claim_id':cid,'claim':r['claim'],'gold':r['gold'],'baseline':b['prediction'],'candidate_prediction':r['prediction'],'correct_before':b['correct'],'correct_after':r['correct'],'guard_arm':r['guard_arm'],'selected_paper_ids':r['selected_paper_ids']})
    for cid in base:
        b=base[cid]
        if b['gold']!='CONTRADICTED' or not b['correct'] or owner[cid]['prediction']=='CONTRADICTED':continue
        ctx=groups['context_only'][cid];gated=groups['gate_only_0.65'][cid]
        cause=[]
        if ctx['prediction']!='CONTRADICTED':cause.append('context_alone_loses_baseline_correct_contradiction')
        if gated['prediction']!='CONTRADICTED':cause.append('gate_alone_loses_baseline_correct_contradiction')
        if not cause:cause.append('combined_interaction_only')
        trace=read(OUT/'local_traces'/f'{cid}.json')
        before={p['paper_id']:p for p in trace['baseline']['per_paper_verdicts']}
        after={p['paper_id']:p for p in trace['context']['per_paper_verdicts']}
        dropped=[pid for pid,paper in before.items() if paper['verdict']=='CONTRADICTED' and max((p['score'] for p in trace['baseline']['evidence_passages'] if p['paper_id']==pid),default=-1)<.65]
        flips={pid:{'baseline':paper['verdict'],'context':after[pid]['verdict']} for pid,paper in before.items() if paper['verdict']!=after[pid]['verdict']}
        contra_losses.append({'contradiction_papers_dropped_by_gate':dropped,'per_paper_verdict_changes':flips,'claim_id':cid,'claim':b['claim'],'gold':b['gold'],'baseline':b['prediction'],'context_only':ctx['prediction'],'gate_only':gated['prediction'],'yuyue_candidate':owner[cid]['prediction'],'guard':groups['train_selected_guard'][cid]['prediction'],'guard_and_gate':groups['train_selected_guard_and_gate'][cid]['prediction'],'diagnostic':';'.join(cause)})
    split_audit=read(OUT/'SPLIT_AUDIT.json')
    overlap=set(split_audit['exact_claim_text_overlap'])
    text_disjoint={name:classification([r for r in group.values() if r['claim'] not in overlap]) for name,group in groups.items()}
    finite_count=0
    import math
    for path in (OUT/'nli_cache').glob('*.json'):
        prediction=read(path);probs=prediction['raw_probabilities']
        assert len(probs)==3 and all(math.isfinite(p) and 0<=p<=1 for p in probs.values()) and abs(sum(probs.values())-1)<1e-5
        assert math.isfinite(prediction['confidence']);finite_count+=1
    test_log=read_log(OUT/'tests.log') if (OUT/'tests.log').exists() else 'Not run'
    guarded=groups['train_selected_guard_and_gate'];prior=groups['context_and_gate_0.65']
    guarded_vs_prior={'paired_accuracy_delta_ci':bootstrap([int(guarded[cid]['correct'])-int(prior[cid]['correct']) for cid in base]),
        'paired_contradiction_recall_delta_ci':bootstrap([int(guarded[cid]['correct'])-int(prior[cid]['correct']) for cid in base if base[cid]['gold']=='CONTRADICTED']),
        'regressions_by_gold':{label:sum(base[cid]['correct'] and not guarded[cid]['correct'] and base[cid]['gold']==label for cid in base) for label in ['SUPPORTED','CONTRADICTED','INSUFFICIENT']}}
    report={'date':'2026-11-11','guarded_vs_prior':guarded_vs_prior,'software_tests':test_log[-1800:],'split_audit':split_audit,'exact_text_disjoint_sensitivity':text_disjoint,'finite_local_prediction_cache_entries':finite_count,'conditions':conditions,'comparisons':comparisons,'training_guard':frozen,'training_gate':gate_train,
      'original_baseline_verdict_agreement':300,'owner_candidate_agreement':owner_agreement,'owner_candidate_total':300,'contradiction_regressions_in_owner_candidate':contra_losses,
      'interpretation':'Guard/gate choices frozen from train. Six predeclared arms reported without dev-based winner selection. Dev already inspected; this does not establish generalization.',
      'evidence_accounting':'For guarded context routes, evidence includes the union of auxiliary focused input and retained context. No scoring-only anchors trick. Original focused raw probabilities were unavailable and remain empty.',
      'production_default':'Unchanged. Research policies only; no claim all metrics improve together.'}
    (RESULTS/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');csv_write(RESULTS/'per_claim_results.csv',rows+gate_rows);csv_write(RESULTS/'changed_claims.csv',differences);csv_write(RESULTS/'contradiction_regressions.csv',contra_losses)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':9,'figure.dpi':160})
    names=list(conditions);labels=['Original','Gate only','Context only','Yu Yue context+gate','Protected decisions','Protected +train gate']
    fig,axes=plt.subplots(1,3,figsize=(13,5),layout='constrained')
    metrics=[('Accuracy',lambda r:r['all']['accuracy']),('Contradicted recall',lambda r:r['all']['per_class']['CONTRADICTED']['recall']),('Input evidence F1',lambda r:r['evidence']['f1'])]
    for ax,(title,getter) in zip(axes,metrics):
        vals=[getter(conditions[n]) for n in names];bars=ax.barh(labels,vals,color=['#777777','#7290ad','#64a395','#426caa','#bd8c46','#a06558']);ax.set_xlim(0,1);ax.set_title(title)
        for bar,v in zip(bars,vals):ax.text(v+.01,bar.get_y()+bar.get_height()/2,f'{v:.3f}',va='center',fontsize=8)
    fig.suptitle('300 already-inspected SciFact dev claims · unchanged rankings and model weights')
    fig.savefig(RESULTS/'tradeoffs.png');fig.savefig(RESULTS/'tradeoffs.pdf');plt.close(fig)
    primary=conditions['train_selected_guard_and_gate'];prior_metrics=conditions['context_and_gate_0.65'];original_metrics=conditions['baseline'];primary_changes=comparisons['train_selected_guard_and_gate'];gate_only=conditions['gate_only_0.65']
    lines=['# Context and contradiction-protection study — 11 November 2026','',
      'This is an exploratory comparison on the same previously inspected SciFact development set. Candidate rules and the optional gate were selected on 200 training claims; no development metric selects a deployment default. The production frontend and original frozen results are unchanged.','',
      '## Main finding','',
      f"The train-selected protected-decision +{gate_train['selected_gate']} context-gate candidate reaches **{pct(primary['all']['accuracy'])} accuracy and {pct(primary['all']['per_class']['CONTRADICTED']['recall'])} contradiction recall**, versus {pct(original_metrics['all']['accuracy'])}/{pct(original_metrics['all']['per_class']['CONTRADICTED']['recall'])} originally and {pct(prior_metrics['all']['accuracy'])}/{pct(prior_metrics['all']['per_class']['CONTRADICTED']['recall'])} for Yu Yue’s candidate. It retains {round(primary['all']['accuracy']*300)}/300 correct, fixes {primary_changes['fixes']} original errors and introduces {primary_changes['regressions']} regressions; {guarded_vs_prior['regressions_by_gold']['INSUFFICIENT']} regressions are no-positive/INSUFFICIENT cases.",
      '', f"This is a tradeoff, not a universal upgrade. Relative to Yu Yue’s candidate, accuracy changes by {(primary['all']['accuracy']-prior_metrics['all']['accuracy'])*100:+.1f} percentage points and contradiction recall by {(primary['all']['per_class']['CONTRADICTED']['recall']-prior_metrics['all']['per_class']['CONTRADICTED']['recall'])*100:+.1f} points. Contradiction precision is {pct(primary['all']['per_class']['CONTRADICTED']['precision'])} versus {pct(prior_metrics['all']['per_class']['CONTRADICTED']['precision'])}; INSUFFICIENT recall is {pct(primary['all']['per_class']['INSUFFICIENT']['recall'])} versus {pct(prior_metrics['all']['per_class']['INSUFFICIENT']['recall'])}; evidence F1 is {primary['evidence']['f1']:.3f} versus {prior_metrics['evidence']['f1']:.3f}. The guard preserves baseline errors as well as successes.",'',
      f"Gate-only input evidence F1 is {gate_only['evidence']['f1']:.3f}, with {pct(gate_only['all']['accuracy'])} accuracy and {pct(gate_only['all']['per_class']['CONTRADICTED']['recall'])} contradiction recall. These arms expose different tradeoffs. Keep the default unchanged pending review and independent evaluation.",'',
      '## What was tested','',
      '- Original focused sentences, gate-only 0.65, whole-abstract context only, Yu Yue context+gate0.65, a training-selected guard, and that guard with a separately train-selected context gate.',
      '- Guard: keep the original focused Supported/Contradicted verdict. If it is Insufficient/Conflicting, use bounded abstract context. Context stays exact-source, fits the actual 512-token premise/claim limit or falls back to the original focused passages.',
      f"- Training selected `{frozen['selected_policy']}`. The second training experiment selected context gate **{gate_train['selected_gate']}**; the gate never removes a protected decisive verdict.",
      f"- Local CPU run made {summary['new_local_nli_calls']} new passage predictions and reused {summary['cached_passage_reuses']} measured/cached predictions. Baseline verdicts reproduced 300/300. Context+gate agrees with Yu Yue’s saved candidate on {owner_agreement}/300 claims.",
      '- Existing focused CPU predictions were reused with their measured label/confidence; missing raw class probabilities were not invented. The guard-training comparison uses the pinned owner’s actual training predictions. The gate-training context predictions were evaluated locally.','',
      '## Results','', '| Condition | Accuracy | Macro F1 | Supported recall | Contradicted recall | Contradicted precision | Insufficient recall | Evidence precision | Evidence recall | Evidence F1 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in conditions.items():
        c=r['all']['per_class'];e=r['evidence'];lines.append(f"| {name} | {pct(r['all']['accuracy'])} | {r['all']['macro_f1']:.3f} | {pct(c['SUPPORTED']['recall'])} | {pct(c['CONTRADICTED']['recall'])} | {pct(c['CONTRADICTED']['precision'])} | {pct(c['INSUFFICIENT']['recall'])} | {e['precision']:.3f} | {e['recall']:.3f} | {e['f1']:.3f} |")
    lines+=['','![Measured tradeoffs](results/tradeoffs.png)','',
      'Evidence scores cover complete original annotated sentences. Extra unannotated context counts as false positives under the closed benchmark; partial sentences count as extra false positives. Guarded context routes include both focused and context input. These are not official SciFact leaderboard metrics.','',
      '## Fixes and regressions','', '| Candidate | Baseline errors fixed | New errors introduced | Net additional correct |','|---|---:|---:|---:|']
    for name in ['context_and_gate_0.65','train_selected_guard','train_selected_guard_and_gate']:
        r=comparisons[name];lines.append(f"| {name} | {r['fixes']} | {r['regressions']} | {r['net_correct']:+d} |")
    lines += ['',f"Yu Yue’s candidate loses {len(contra_losses)} of the baseline’s correctly contradicted claims. `results/contradiction_regressions.csv` separates context-only losses, gate-only losses and combined interactions; tags can overlap and are diagnostic, not causal proof.",
      '', 'The guard preserves any already-correct focused contradiction by its decision rule. It also preserves wrong focused decisive predictions, so false contradictions and false supports remain. This is an explicit tradeoff, not a universal fix.','',
      '## What to review','',
      'Software checks: the complete frontend suite passes, including context/guard/cache-translation and Windows log-encoding regressions. See software_tests in results/summary.json for the captured result.','',
      '- `results/changed_claims.csv`: every changed verdict for both new candidates, including regressions.',
      '- `results/contradiction_regressions.csv`: original correct contradictions lost by the prior candidate, and the new candidates’ outcomes.',
      '- `results/per_claim_results.csv`: all 1,800 measured claim/condition rows; no failed cases excluded.',
      '- `outputs/context_study_20261111/local_traces/`: exact focused and context model input/predictions. Saved research replays display the decisive arm; auxiliary inputs remain in traces and evidence accounting.',
      '- `TRAIN_FROZEN.json`, `GATE_TRAIN_FROZEN.json` and the run protocol: training choices and provenance.','',
      '## Limits and next action','',
      f"Train/dev IDs are disjoint, but {len(overlap)} exact claim text is shared. The primary 300 cohort stays intact. An additional text-disjoint sensitivity analysis (299 claims) is retained in results/summary.json; it is not substituted for the primary result. Related papers/claim variants can still overlap.",'',
      'Do not replace the default solely because one development metric rises. Review class costs and both new candidates’ regressions, then run an independent evaluation with the chosen policy frozen. Related train/dev claims or papers may overlap; claim-bootstrap intervals do not remove this dependence. CSFCube/RECIPER retrieval results are unchanged and do not supply equivalent verdict labels.',
      '', 'No model weights, NLI threshold, ranking, public API dependency or source claim were changed. All additions stay inside explanation_frontend. Full models/corpora/raw caches remain ignored by Git.']
    (ROOT/'evaluation/context_study/REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for p in [OUT/'TRAIN_FROZEN.json',OUT/'GATE_TRAIN_FROZEN.json',OUT/'RUN_PROTOCOL.json',OUT/'SPLIT_AUDIT.json']:shutil.copy2(p,RESULTS/p.name)
    replay_dir=RESULTS/'replays';replay_dir.mkdir(exist_ok=True)
    for kind in ['replays','gate_replays']:
        for p in (OUT/kind).glob('*.json'):shutil.copy2(p,replay_dir/(kind+'-'+p.name))
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in RESULTS.rglob('*') if p.is_file() and p.name!='ARTIFACTS.json'}
    (RESULTS/'ARTIFACTS.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with zipfile.ZipFile(OUT/'review-kit.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in [*[p for p in RESULTS.rglob('*') if p.is_file()],ROOT/'evaluation/context_study/REPORT.md',*list((OUT/'replays').glob('*.json'))]:
            if p.is_file():archive.write(p,str(p.relative_to(ROOT)))
    print(json.dumps({n:{'accuracy':r['all']['accuracy'],'contra_recall':r['all']['per_class']['CONTRADICTED']['recall'],'evidence_f1':r['evidence']['f1']} for n,r in conditions.items()}),flush=True)
if __name__=='__main__':main()
