"""Build an auditable review package from measured artifacts, never invented scores."""
import csv,json,hashlib,shutil,zipfile,statistics
from pathlib import Path
from .freeze import ROOT,OUT,freeze
from .metrics import bootstrap
RESULTS=ROOT/'evaluation/validation/results'

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows([{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])
def pct(n):return 'n/a' if n is None else f'{100*n:.1f}%'
def ci(c):return f"{pct(c['estimate'])} [{pct(c['low'])}, {pct(c['high'])}]"

def main():
    baseline=freeze();RESULTS.mkdir(exist_ok=True)
    replay_dir=RESULTS/'replays';replay_dir.mkdir(exist_ok=True)
    for p in list((OUT/'scifact_full/replays').glob('*.json'))+list((OUT/'pdf').glob('*replay.json')):shutil.copy2(p,replay_dir/p.name)
    science=read(OUT/'scifact_full/summary.json');rows=read(OUT/'scifact_full/case_rows.json');other=read(OUT/'other_datasets.json');stress=read(OUT/'stress.json');clean=read(OUT/'clean_setup.json');reliability=read(OUT/'reliability.json')
    main_name='retrieved_papers_production_selector';main_rows=[r for r in rows if r['condition']==main_name];main_summary=science['conditions'][main_name]
    controls={(r['claim_id'],r['condition']):r for r in rows}
    errors=[]
    for r in main_rows:
        if r['correct']:continue
        tags=[]
        if r['status']=='failed':tags.append('operational_failure')
        if r['gold']=='INSUFFICIENT':tags.append('non_insufficient_on_no_annotated_positive')
        else:
            if r.get('gold_documents_retrieved',0)<r.get('gold_documents',0):tags.append('gold_paper_missing_top 5')
            if r.get('complete_rationale_docs',0)<r.get('gold_rationale_docs',0):tags.append('complete_gold_rationale_not_covered')
            oracle=controls.get((r['claim_id'],'oracle_gold_rationale'),{})
            if not oracle.get('correct'):tags.append('oracle_verifier_also_wrong')
        if r['prediction']=='CONFLICTING':tags.append('extra_conflicting_evidence')
        if r.get('confidence',0)>=.9:tags.append('high_aggregate_confidence_error_not_calibrated')
        errors.append({'dataset':'SciFact dev','case_id':r['claim_id'],'claim':r['claim'],'gold':r['gold'],'prediction':r['prediction'],'status':r['status'],'diagnostic_tags':'; '.join(tags),'confidence':r.get('confidence'),'responsible_interfaces':'retrieval/selection/verifier/aggregation; tags are co-occurrence diagnostics, not causal proof'})
    for r in stress['synthetic_nli_cases']:
        if not r['passed']:errors.append({'dataset':'synthetic diagnostic','case_id':r['case_id'],'claim':r['claim'],'gold':r['expected'],'prediction':r['actual'],'diagnostic_tags':'population/generalisation: neutral vs contradiction','confidence':r['confidence'],'responsible_interfaces':'Chong Yao verifier; Naila XAI can explain the error, not correct it'})
    csv_write(RESULTS/'test_results.csv',rows);csv_write(RESULTS/'error_catalogue.csv',errors)
    paired=[]
    retrieval_rows=read(OUT/'scifact_full/retrieval_rows.json')
    with (OUT/'retrieval_per_query.csv').open(encoding='utf-8-sig',newline='') as f:raw=list(csv.DictReader(f))
    scopes=[('SciFact','dev',retrieval_rows,'claim_id')]+[(d,s,[r for r in raw if r['dataset']==d and r['split']==s],'query_id') for d,s in dict.fromkeys((r['dataset'],r['split']) for r in raw)]
    for dataset,split,rr,key in scopes:
        methods={m:{str(r[key]):r for r in rr if r['method']==m} for m in ('rrf','bm25')}
        ids=list(methods['rrf']);diff=[float(methods['rrf'][i]['Recall@10'])-float(methods['bm25'][i]['Recall@10']) for i in ids]
        clusters=[methods['rrf'][i].get('cluster',i) for i in ids]
        paired.append({'dataset':dataset,'split':split,'comparison':'RRF minus BM25 Recall@10; descriptive, not model selection',**bootstrap(diff,clusters)})
    owner_tests=read(OUT/'owner_tests.json') if (OUT/'owner_tests.json').exists() else {'results':[]}
    audit=read(OUT/'artifact_audit.json')
    live=read(ROOT/'outputs/integration/live-report.json') if (ROOT/'outputs/integration/live-report.json').exists() else {'status':'not_run'}
    live_current=(OUT/'live_smoke.log').exists() and '"source": "live"' in (OUT/'live_smoke.log').read_text(encoding='utf-8')
    if not live_current:live={'status':'failed_or_not_completed','details':'See live_smoke.log; previous integration artifacts are not accepted as current.'}
    report={'date':'2026-11-10','baseline_commit':baseline['git_commit'],'science':science,'other_datasets':other,'paired_retrieval_differences':paired,
        'software':{'owner_tests':owner_tests,'artifact_audit':audit,'clean_install':clean,'stress':{k:v for k,v in stress.items() if k!='synthetic_nli_cases'},'retrieval_repeat_checks':reliability['retrieval_top20_repeated'],'xai_checks':[{k:v for k,v in r.items() if k!='attribution'} for r in reliability['xai']],'live_functional_smoke':live},
        'error_rows':len(errors),'scope':'Frozen descriptive evaluation, no model/threshold tuning. Scientific mistakes are retained.'}
    (RESULTS/'validation-summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    # Standalone scientific plots; no fabricated target or interpolated metric.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'figure.dpi':160})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    labels=['Gold rationale','Gold papers + selector','Retrieved top 5 + selector']
    values=[science['conditions'][k]['positive_claims']['accuracy'] for k in science['conditions']]
    bars=axes[0].barh(labels,values,color=['#3d8c8c','#466ca8','#ad6548']);axes[0].set_xlim(0,1);axes[0].set_xlabel('Claim accuracy, same 188 annotated-positive claims')
    for bar,val in zip(bars,values):axes[0].text(val+.01,bar.get_y()+bar.get_height()/2,pct(val),va='center')
    axes[0].set_title('Controls locate failure stages; not causal attribution')
    matrix=main_summary['all']['confusion_matrix'];gold=list(matrix);pred=list(matrix[gold[0]]);arr=np.array([[matrix[g][p] for p in pred] for g in gold])
    axes[1].imshow(arr,cmap='Blues');axes[1].set_xticks(range(len(pred)),[p.replace('PROCESSING_FAILURE','FAILURE') for p in pred],rotation=40,ha='right',fontsize=8);axes[1].set_yticks(range(len(gold)),gold,fontsize=8);axes[1].set_ylabel('Gold');axes[1].set_title('Full offline route: all 300 SciFact dev claims')
    for y in range(len(gold)):
        for x in range(len(pred)):axes[1].text(x,y,str(arr[y,x]),ha='center',va='center',color='white' if arr[y,x]>arr.max()/2 else 'black')
    fig.savefig(RESULTS/'pipeline_accuracy.png');fig.savefig(RESULTS/'pipeline_accuracy.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    panels=[('SciFact dev',list(science['retrieval'].items())),('CSFCube test',[(r['method'],r) for r in other['summaries'] if r['dataset']=='CSFCube' and r['split']=='fold1_test']),('RECIPER',[(r['method'],r) for r in other['summaries'] if r['dataset']=='RECIPER'])]
    for ax,(title,rr) in zip(axes,panels):
        vals=[r['Recall@10'] for m,r in rr];cis=[r['recall10_ci'] for m,r in rr]
        ax.bar([m.upper() for m,r in rr],vals,color=['#466ca8','#3d8c8c','#ad6548'],yerr=[[v-c['low'] for v,c in zip(vals,cis)],[c['high']-v for v,c in zip(vals,cis)]],capsize=4);ax.set_ylim(0,1);ax.set_title(title);ax.set_ylabel('Recall@10;95% cluster-bootstrap interval')
    fig.suptitle('Different tasks, pools and judgments: compare methods within each dataset')
    fig.savefig(RESULTS/'retrieval_domains.png');fig.savefig(RESULTS/'retrieval_domains.pdf');plt.close(fig)
    calibration=science['calibration'];fig,ax=plt.subplots(figsize=(6.5,4.5),layout='constrained');ax.plot([0,1],[0,1],'--',color='grey');ax.scatter([b['mean_confidence'] for b in calibration['bins']],[b['accuracy'] for b in calibration['bins']],s=[max(20,b['n']*3) for b in calibration['bins']],color='#466ca8');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean raw passage confidence',ylabel='Accuracy',title='Oracle SUPPORT/CONTRADICT rationale pairs only\nBubble area reflects count; not aggregate calibration');fig.savefig(RESULTS/'oracle_reliability.png');plt.close(fig)
    lines=['# Full-pipeline validation — 10 November 2026','',
        f"Baseline `{baseline['git_commit']}`; pinned models, data hashes and frozen settings in `PROTOCOL.json` and `BASELINE.json`. No threshold or model was tuned on these results.",'',
        '## What was executed','',
        '- All 300 SciFact dev claims against all 5,183 corpus papers. BM25/SBERT top 100 branches, RRF(k=60), automatic top 5 papers, actual frozen semantic selector, 512-token budget, DeBERTa NLI, aggregation, grounded templates and replay checks. No gold-paper injection.',
        '- Privileged gold-rationale and gold-paper controls on 188 annotated-positive claims. 112 no-positive claims stay in the full-route denominator as closed-benchmark INSUFFICIENT.',
        '- CSFCube 24 dev /26 test query-facet tasks and RECIPER 1,024 questions, independently rerun for BM25, SBERT and RRF. CSFCube and RECIPER have retrieval judgments, not NLI or sentence-evidence gold labels.',
        '- Separate clean Windows Python environment, exact dependency installation, pip check, full test suite, offline actual-model smoke, fresh public-provider smoke, PDF rejection/real-PDF tests, synthetic language tests and actual repeated retrieval/NLI/XAI calls.','',
        '## Scientific results','',
        '| Condition | Claims | Accuracy | Macro F1 | Evidence micro F1 on positive claims | Processing failures |','|---|---:|---:|---:|---:|---:|']
    for name,c in science['conditions'].items():lines.append(f"| {name} | {c['all']['n']} | {pct(c['all']['accuracy'])} | {c['all']['macro_f1']:.3f} | {c['evidence_micro_positive_only']['f1']:.3f} | {c['operational_failures']} |")
    lines += ['',f"Full-route accuracy 95% bootstrap interval: **{ci(main_summary['accuracy_ci'])}**. Comparing controls: use the same 188 positive claims; full-route positive-only accuracy is {pct(main_summary['positive_claims']['accuracy'])}.",
        '', 'Macro F1 uses two gold-present classes in controls and three in the full cohort; these columns are not directly comparable. Oracle evidence F1=1 is by construction, not learned selector performance.','', '![Stage controls and full confusion matrix](results/pipeline_accuracy.png)','',
        '| Dataset / split | Method | Queries | Recall@10 | MRR@10 | nDCG@10 |','|---|---|---:|---:|---:|---:|']
    for m,r in science['retrieval'].items():lines.append(f"| SciFact dev | {m} | {r['eligible']} | {r['Recall@10']:.3f} | {r['MRR@10']:.3f} | {r['nDCG@10']:.3f} |")
    for r in other['summaries']:lines.append(f"| {r['dataset']} / {r['split']} | {r['method']} | {r['queries']} | {r['Recall@10']:.3f} | {r['MRR@10']:.3f} | {r['nDCG@10']:.3f} |")
    lines += ['', 'CSFCube graded nDCG uses exponential gains 2^grade−1 and appears separately in the JSON. Binary relevance is grade ≥2. RECIPER is adapted source-paper retrieval over original titles/abstracts; generated answers are excluded. Its single source-paper target is not exhaustive relevance.','',
        '![Three domain retrieval results](results/retrieval_domains.png)','',
        '## Reliability and limitations','',
        f"- Clean dependency install exit {clean['dependency_install_exit_code']}; pip check exit {clean['pip_check_exit_code']}; final clean suite exit {clean['tests_exit_code']}; offline actual-model smoke exit {clean['offline_model_smoke_exit_code']}. Initial run had 162 passes and one 3-second UI test timeout under concurrent benchmark load; AppTest allowance raised to 15s, then 163/163 passed. After adding the checkout-newline regression check, the final clean suite passed 164/164. No application decision logic changed.",
        f"- Live functional smoke: {live.get('status')}; {live.get('candidate_count','n/a')} candidates / {live.get('verified_papers','n/a')} verified papers. Public-service availability and rate limits remain external dependencies. A working live response is not an accuracy measurement.",
        f"- PDF checks: {sum(r['passed'] for r in stress['pdf_cases'])}/{len(stress['pdf_cases'])}. Includes two-column text, image-only rejection, encryption, malformed input, 201-page rejection, 20MiB limit and an original scientific PDF with exact page/span export/replay. Real PDF result: {stress['pdf_cases'][-1].get('verdict','unavailable')}; functionality does not imply scientific correctness or perfect reading order.",
        f"- Synthetic language expectations: {stress['synthetic_nli_passed']}/{stress['synthetic_nli_total']}. Failures are retained: changes in population/generalisation can incorrectly turn unknown evidence into a confident contradiction.",
        f"- NLI repeatability: {sum(r['passed'] for r in stress['repeatability'])}/{len(stress['repeatability'])} exact probability repeats. Retrieval: {sum(r['passed'] for r in reliability['retrieval_top20_repeated'])}/{len(reliability['retrieval_top20_repeated'])} repeated top 20 lists. These are bounded checks on this machine/software stack.",
        '- Two new synthetic word-MASK occlusion cases have repeatability and exact-offset checks, including a wrong verifier prediction. They are diagnostic demonstrations, not additional SciFact plausibility evidence. Prior four-pair occlusion pilot remains exploratory; IG remains experimental after failed convergence gates.',
        f"- Raw oracle rationale-pair ECE(10 bins): {calibration['ece_10_bins']:.3f}, n={calibration['n']}. Gold labels cover SUPPORT/CONTRADICT only. Do not present this as three-class or aggregate confidence calibration. High model confidence does not guarantee a correct verdict.",
        '- SciFact dev and previously published CSFCube results were already inspected. These are frozen descriptive audits, not unseen holdout performance. CSFCube test has only 26 tasks; dev/test share query-paper identities. RECIPER uses the entire released QA set with frozen zero-shot settings. Intervals are descriptive and do not eliminate dataset dependence.',
        '- Production selector splits abstract strings; sentence evidence credit requires full coverage of the original annotated sentence, with fragments counted as extra false positives. This intentionally conservative measure is not official SciFact leaderboard parity. Closed-world judgments can mark unannotated but plausible passages as false positives.',
        '- SciFact uses production retrieval encoder revision2a7… with Porter BM25. CSFCube/RECIPER use published protocol encoder1110… with simple BM25; results are not one identical configuration across domains. The pinned lexical retriever is custom NumPy BM25, not rank-bm25.',
        '- Desktop validation found CRLF-only source differences between checkouts. Source checks now retain raw hashes and accept matching LF-normalized hashes; actual source edits still fail the frozen-state gate. Measured artifacts/replays preserve their exact bytes in Git.','- Offline full-corpus automatic top 5 selection is an evaluation wrapper. The public-web and interactive user-selection journey is tested separately; this is not a benchmark of public-web discovery recall.','',
        '## Error catalogue and next work by owner','',
        f"Pinned owner suites rerun today: {sum(r.get('passed',0) for r in owner_tests['results'])} passed across {len(owner_tests['results'])} owners. Isolated copies, external network blocked; verifier initialization uses the actual pinned local model. Artifact accounting/independent sklearn checks: {audit['passed']}/{audit['total']}.",'',
        f"`results/error_catalogue.csv` retains {len(errors)} failed full-route/synthetic cases. Tags can overlap and identify where to investigate, not proven causes.",
        '', '| Owner | Working and tested | Open work |','|---|---|---|',
        '| Raghavi | Fresh public discovery handoff to real selector/verifier; source provenance and provider-error tests | [PENDING] Test provider recall on a shared query set; rate limits and changing indexes cannot be guaranteed by local tests. |',
        '| Naila | BM25/SBERT three-domain runs; deterministic retrieval subset; word occlusion repeatability | [PENDING] Review retrieval misses and population-error explanations. IG convergence remains unresolved; larger unbiased XAI validation is pending. |',
        '| Yu Yue | RRF, actual semantic selection, whole-span token budgets, three-domain protocols | [PENDING] Inspect complete-rationale misses/partial sentence fragments. Any splitter/selector change needs a new version and separate development evaluation. |',
        '| Chong Yao | Real local NLI, cached aggregation, source-linked outputs and repeatability | [PENDING] Review oracle errors, NEUTRAL/CONTRADICT confusion, population shift and confidence reliability. Any threshold/calibration work needs separate development data. |',
        '| Kireeti | Integrated templates, claim/subset state, PDF safeguards, replay, clean setup and review exports | [PENDING] Review real PDF reading order and representative good/bad replays with teammates; human usefulness/citation judgments remain pending. |','',
        '## What you need to review','',
        '1. Open `results/pipeline_accuracy.png` and `results/retrieval_domains.png`; check that these are the results you are willing to present, including weak metrics.',
        '2. Open `results/error_catalogue.csv` and compare good and bad SciFact replays in `outputs/validation/scifact_full/replays` using Replay/import. Keep the same claim when inspecting evidence and subset verdicts.',
        '3. Inspect `outputs/validation/pdf/real-arxiv-replay.json` for source-page correctness and reading order; the returned verdict can still be scientifically wrong.',
        '4. Run a personal claim and a personal text PDF. Record human feedback separately; no human scores have been filled in.',
        '5. Use `outputs/validation/review-kit.zip` to share measured reports/replays. Models, private data and downloaded corpora are excluded.','',
        '## Reproduce','',
        'From explanation_frontend, run `validate.cmd`. Existing completed frozen study artifacts are reused, not silently rerun. For another SciFact run, use `python -m explanation_frontend.evaluation.validation.scifact --output scifact_repeat` from the repository root; preserve the original outputs. Add `-CleanSetup` to validate.cmd for another isolated Python environment check. If dependencies/models are absent, run the existing `setup.ps1` followed by `setup-pipeline.ps1` (which includes discovery and attribution setup). PowerShell can launch these with `-ExecutionPolicy Bypass` for that process. All writes stay under this folder.','',
        'Full per-condition results: `results/test_results.csv`; machine-readable metrics: `results/validation-summary.json`; source/data/model hashes: `outputs/validation/BASELINE.json`. The report distinguishes measured results, fixture acceptance and pending human validation.']
    (ROOT/'evaluation/validation/FINAL_TEST_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    shutil.copy2(OUT/'BASELINE.json',RESULTS/'BASELINE.json')
    manifests={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in RESULTS.rglob('*') if p.is_file() and p.name!='ARTIFACTS.json'}
    (RESULTS/'ARTIFACTS.json').write_text(json.dumps(manifests,indent=2),encoding='utf-8')
    files=[p for p in RESULTS.rglob('*') if p.is_file()]+[ROOT/'evaluation/validation/FINAL_TEST_REPORT.md',ROOT/'evaluation/validation/PROTOCOL.json']+list((OUT/'scifact_full/replays').glob('*.json'))+list((OUT/'pdf').glob('*replay.json'))
    with zipfile.ZipFile(OUT/'review-kit.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in files:archive.write(p,str(p.relative_to(ROOT)))
    print(json.dumps({'full_route_accuracy':main_summary['all']['accuracy'],'macro_f1':main_summary['all']['macro_f1'],'errors':len(errors),'package':str(OUT/'review-kit.zip')}),flush=True)
    return report
if __name__=='__main__':main()
