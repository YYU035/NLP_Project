"""Offline, gated SciFact IG pilot. Owner evaluation helper is preserved unchanged.
Run: python -m explanation_frontend.evaluation.xai_qc --help
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.model-cache/matplotlib'))


def mean(values):
    return statistics.mean(values) if values else None


def summarise(rows):
    eligible=[r for r in rows if r.get('numerical_status')=='passed' and 'error' not in r]
    def group(rs):
        return {'n':len(rs), **{key:mean([r[key] for r in rs]) for key in
            ('ap','ap_random','top1_hit','top1_random')},
            'faithfulness':{key:mean([r['faithfulness'][key] for r in rs]) for key in
            ('aopc_comprehensiveness','aopc_comprehensiveness_random','aopc_sufficiency','aopc_sufficiency_random')}}
    return {'attempted':len(rows),'stable':len(eligible),
            'unstable':sum(r.get('numerical_status')=='unstable' and 'error' not in r for r in rows),
            'failed':sum('error' in r for r in rows),
            'all_stable':group(eligible),
            'correct_stable':group([r for r in eligible if r['correct_raw']]),
            'incorrect_stable':group([r for r in eligible if not r['correct_raw']]),
            'by_label':{label:group([r for r in eligible if r['gold_label']==label]) for label in ('SUPPORT','CONTRADICT')}}


def sample_examples(examples, tokenizer, limit, max_tokens, seed):
    """Deterministic balanced pilot; explicit length eligibility, no outcome selection."""
    eligible=[e for e in examples if len(tokenizer.encode(' '.join(e['sentences']),e['claim'],truncation=False))<=max_tokens]
    rng=random.Random(seed); groups={label:[e for e in eligible if e['gold_label']==label] for label in ('SUPPORT','CONTRADICT')}
    for group in groups.values(): rng.shuffle(group)
    selected=[]
    while len(selected)<limit and any(groups.values()):
        for group in groups.values():
            if group and len(selected)<limit: selected.append(group.pop())
    return selected, len(eligible)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,default=ROOT/'outputs/xai/pilot')
    ap.add_argument('--limit',type=int,default=4)
    ap.add_argument('--max-tokens',type=int,default=192)
    ap.add_argument('--steps',type=int,nargs='+',default=[32,128,512])
    ap.add_argument('--baseline',choices=['pad','mask'],default='mask')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--claim-id',type=int,nargs='+',help='Explicit targeted diagnostic subset; not a representative sample.')
    ap.add_argument('--random-trials',type=int,default=10)
    ap.add_argument('--audit-all',action='store_true',help='Also score every full-abstract annotated pair; separate verifier audit, not end-to-end accuracy.')
    ap.add_argument('--audit-only',action='store_true',help='Run the full-abstract verifier audit without IG or XAI metrics.')
    args=ap.parse_args(argv)
    if args.audit_only: args.audit_all=True
    if args.limit<1 or not 1<=args.max_tokens<=512 or args.random_trials<2: ap.error('Invalid sample/token/trial limits')
    args.out_dir=args.out_dir.resolve()
    if not args.out_dir.is_relative_to(ROOT): ap.error('Outputs must stay inside explanation_frontend')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    from ..pipeline import local_verifier
    from ..semantic_integration import local_selector
    from ..xai_runtime import explain_checked
    from .._upstream.xai.eval_ig_dev import build_examples,average_precision,random_ap
    from .._upstream.claim_verifier.schemas import EvidencePassage,PassageLabel
    from ..pipeline_setup import NLI_ID,NLI_REVISION
    v=local_verifier(); examples=build_examples(str(args.data_dir))
    sampling_pool=[e for e in examples if not args.claim_id or e['claim_id'] in args.claim_id]
    selected,n_eligible=sample_examples(sampling_pool,v.tokenizer,args.limit,args.max_tokens,args.seed)
    if args.audit_only: selected=[]
    if not selected and not args.audit_all:
        raise ValueError('No length-eligible examples; increase --max-tokens (up to 512)')
    manifest={'dataset':'SciFact dev annotated claim-document pairs, full abstracts',
        'data_sha256':{name:hashlib.sha256((args.data_dir/name).read_bytes()).hexdigest() for name in ('corpus.jsonl','claims_dev.jsonl')},
        'n_annotated_pairs':len(examples),'n_length_eligible':n_eligible,
        'sample_ids':[[e['claim_id'],e['doc_id']] for e in selected],'requested_claim_ids':args.claim_id,
        'checkpoint':NLI_ID,'revision':NLI_REVISION,'baseline':args.baseline,'threshold':v.threshold,
        'model_dtype':'float32','model_mode':'eval','internal_batch_size':1,'data_git_revision':'7ed93fa4be69c661422b33b6c88b949dd2636e12',
        'steps':args.steps,'seed':args.seed,'max_tokens':args.max_tokens,'random_trials':args.random_trials,'sentence_random_trials':50,'rationale_gold':'union of annotated rationale sentence sets for the cited document',
        'target':'gold label for plausibility AND faithfulness; not explanation of a different predicted label',
        'perturbation':'delete evidence subword tokens only; claim and special tokens fixed; matched token budgets; ratios 1/5/10/20/50 percent',
        'run_mode':'audit_only' if args.audit_only else 'ig_pilot',
        'selection':('targeted claim IDs; exploratory diagnosis, not representative' if args.claim_id else 'balanced by gold label, seeded without replacement among length-eligible pairs; exploratory pilot, not held-out test'),
        'gate':'finite, nontruncated, relative <= .05 AND absolute delta <= .001',
        'tuning':'Baseline/step policy explored on a synthetic plant pair before this pilot; no scientific performance claim',
        'versions':{}}
    import importlib.metadata
    for name in ['torch','transformers','captum','numpy']: manifest['versions'][name]=importlib.metadata.version(name)
    (args.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    rows=[]; started=time.perf_counter()
    selector=local_selector() if selected else None
    with (args.out_dir/'per_example.jsonl').open('w',encoding='utf-8') as output:
        for index,ex in enumerate(selected):
            row={k:ex[k] for k in ('claim_id','doc_id','gold_label','rationale')}
            t=time.perf_counter()
            print(f'IG pair {index+1}/{len(selected)}: claim {ex["claim_id"]}, document {ex["doc_id"]}',flush=True)
            try:
                result,explainer,run=explain_checked(v,ex['claim'],' '.join(ex['sentences']),baseline=args.baseline,
                    steps=args.steps,target=PassageLabel(ex['gold_label']),sentences=ex['sentences'],return_run=True)
                row.update({k:result[k] for k in ('predicted_label','target_probability','baseline_probability','convergence_delta',
                    'relative_completeness_error','absolute_completeness_error','numerical_status','attempts')})
                row['correct_raw']=row['predicted_label']==ex['gold_label']
                # Investigative evidence coverage, not a causal conclusion about the classifier.
                selection=selector.select(ex['claim'],ex['doc_id'],ex['sentences'])
                from dataclasses import asdict
                row['selection_diagnostic']=[asdict(span) for span in selection]
                indices={i for span in selection for i in span.sentence_indices}
                row['selected_rationale_overlap']=bool(indices.intersection(ex['rationale']))
                row['error_category']='raw_full_abstract_prediction_correct' if row['correct_raw'] else 'raw_full_abstract_verifier_error'
                row['coverage_diagnostic']='gold_rationale_selected' if row['selected_rationale_overlap'] else 'gold_rationale_not_selected'
                # Classify actual selected spans separately; full-abstract errors cannot be blamed on selection.
                selected_predictions=[]
                for span in selection:
                    if len(v.tokenizer.encode(span.text,ex['claim'],truncation=False)) <= v.max_length:
                        pred=v._classify_passage(ex['claim'],EvidencePassage(passage_id=span.passage_id,
                            paper_id=ex['doc_id'],section='abstract',text=span.text))
                        selected_predictions.append(pred.model_dump(mode='json'))
                row['selected_passage_predictions']=selected_predictions
                all_neutral=bool(selected_predictions) and all(p['label']=='NEUTRAL' for p in selected_predictions)
                row['selected_stage_investigation']=('no_selected_passage' if not selected_predictions else
                    'rationale_present_but_all_selected_predictions_neutral' if all_neutral and row['selected_rationale_overlap'] else
                    'rationale_missing_and_all_selected_predictions_neutral' if all_neutral else 'inspect_selected_predictions')
                if result['numerical_status']=='passed':
                    scores=result['sentence_scores']; relevant=set(ex['rationale'])
                    rng=random.Random(args.seed+index)
                    row.update(ap=average_precision(scores,relevant),ap_random=random_ap(len(scores),relevant,rng,trials=50),
                        top1_hit=float(max(range(len(scores)),key=lambda i:scores[i]) in relevant),top1_random=len(relevant)/len(scores),
                        sentence_scores=scores,
                        faithfulness=explainer._faithfulness(run,segments=('evidence',),n_random=args.random_trials,seed=args.seed+index))
                    # Diagnostic artifact is explicit about target, not a validated explanation of the predicted class.
                (args.out_dir/f'pair-{index+1}-attribution.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            except Exception as exc:
                row['error']=f'{type(exc).__name__}: {exc}'
            row['seconds']=time.perf_counter()-t; rows.append(row)
            output.write(json.dumps(row,allow_nan=False)+'\n');output.flush()
            print(json.dumps({k:row.get(k) for k in ('claim_id','doc_id','numerical_status','error','seconds')}),flush=True)
    summary=summarise(rows); summary['seconds']=time.perf_counter()-started
    summary['limitations']=['Exploratory short-input balanced dev pilot; not representative or held-out.','Unstable/failed rows excluded from XAI quality means; coverage always reported.','Gold-target attribution does not explain a different predicted class.','Full-abstract classifier audit differs from deployed selected-passage workflow.','Deletion perturbations can create unnatural text; sentence length affects summed attributions.','Rationale overlap is a coverage diagnostic, not proof of error causation.']
    if args.audit_all:
        audit=[]
        for audit_index,ex in enumerate(examples,1):
            text=' '.join(ex['sentences']); p=EvidencePassage(passage_id=f"{ex['claim_id']}:{ex['doc_id']}",paper_id=ex['doc_id'],section='abstract',text=text)
            pred=v._classify_passage(ex['claim'],p)
            audit.append({'claim_id':ex['claim_id'],'doc_id':ex['doc_id'],'gold_label':ex['gold_label'],
                          'prediction':pred.label.value,'confidence':pred.confidence,
                          'operational_status':pred.operational_status.value})
            if audit_index % 25 == 0: print(f'Verifier audit {audit_index}/{len(examples)}',flush=True)
        (args.out_dir/'verifier-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
        usable=[r for r in audit if r['operational_status']=='OK']
        support=[r for r in usable if r['gold_label']=='SUPPORT']
        summary['full_abstract_audit']={'total_pairs':len(audit),'usable_nontruncated':len(usable),
            'excluded':len(audit)-len(usable),'support_pairs':len(support),
            'support_predicted_neutral':sum(r['prediction']=='NEUTRAL' for r in support),
            'confusion_matrix':{g:{p:sum(r['gold_label']==g and r['prediction']==p for r in usable) for p in ('SUPPORT','CONTRADICT','NEUTRAL')} for g in ('SUPPORT','CONTRADICT')},
            'note':'Thresholded native verifier, full gold-cited abstracts, not pipeline or original 116/216 experiment.'}
    summary['total_seconds_including_optional_audit']=time.perf_counter()-started
    (args.out_dir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    md=['# SciFact XAI exploratory pilot','',f"Attempted: {summary['attempted']}; stable: {summary['stable']}; unstable: {summary['unstable']}; failed: {summary['failed']}.",'',
        '| Group | n stable | AP | AP random | Top-1 | Top-1 random | Comp. | Comp. random | Suff. | Suff. random |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key in ('all_stable','correct_stable','incorrect_stable'):
        s=summary[key];fa=s['faithfulness']
        vals=[s[k] for k in ('ap','ap_random','top1_hit','top1_random')]+list(fa.values())
        md.append('| '+key+' | '+str(s['n'])+' | '+' | '.join('N/A' if x is None else f'{x:.4f}' for x in vals)+' |')
    md+=['','Higher comprehensiveness and lower sufficiency than random are the desired directions; negative values are retained.',*['- '+x for x in summary['limitations']]]
    (args.out_dir/'summary.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2,allow_nan=False),flush=True)
    return summary

if __name__=='__main__': main()
