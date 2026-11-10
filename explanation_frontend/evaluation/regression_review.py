"""Diagnose frozen regressions and package recorded evidence; never run or tune NLI."""
import csv,json,hashlib,html,shutil,zipfile
from pathlib import Path
from collections import Counter
from .independent_audit import ROOT,OUT,load
DEST=ROOT/'evaluation/regression_review_20261111'

def select_review_cases(regressions,all_rows):
    selected=[]
    for transition in sorted({(r['gold'],r['prediction']) for r in regressions}):
        group=[r for r in regressions if (r['gold'],r['prediction'])==transition]
        selected.append((min(group,key=lambda r:int(r['claim_id']))['claim_id'],'One representative of '+transition[0]+' → '+transition[1]))
    for label in ['SUPPORTED','INSUFFICIENT']:
        group=[r for r in all_rows if r['condition']=='yuyue_frozen' and r['gold']==label and r['correct'] and not r['baseline_correct']]
        if group:selected.append((min(group,key=lambda r:int(r['claim_id']))['claim_id'],'Recovered '+label+' case'))
    group=[r for r in all_rows if r['condition']=='yuyue_frozen' and r['gold']=='CONTRADICTED' and not r['correct'] and not r['baseline_correct']]
    if group:selected.append((min(group,key=lambda r:int(r['claim_id']))['claim_id'],'Persistent contradiction error'))
    assert len({cid for cid,_ in selected})==len(selected)
    return selected

def reaggregate_scope(response,kept):
    from ..pipeline import safe_aggregate
    from .._upstream.claim_verifier.verifier import ClaimVerifier
    return safe_aggregate(object.__new__(ClaimVerifier),response,sorted(kept)).aggregated_verdict.value

def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader()
        w.writerows([{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])

def main():
    from .._upstream.claim_verifier.schemas import VerificationResponse
    from ..person4 import Person4Adapter
    from ..models import Paper,RunSnapshot,make_context
    from ..templates import build_template
    from ..validation import validate_explanation
    from ..export import export_snapshot,import_snapshot
    from ..pipeline import safe_aggregate
    from .._upstream.claim_verifier.verifier import ClaimVerifier
    DEST.mkdir(exist_ok=True)
    form=DEST/'REVIEW_FORM.csv'
    if form.exists():
        with form.open(encoding='utf-8-sig',newline='') as f:existing=list(csv.DictReader(f))
        fields=['evidence_supports_verdict','citation_correct','context_helpful','reviewer_name','issue_notes']
        if any(row.get(field,'').strip() for row in existing for field in fields):raise ValueError('Filled review form exists; refusing to overwrite human input')
    cases=DEST/'cases';cases.mkdir(exist_ok=True);replays=DEST/'replays';replays.mkdir(exist_ok=True)
    protocol=load(OUT/'PROTOCOL.json');source_summary=ROOT/'evaluation/independent_results_20261111/summary.json'
    before_hash=hashlib.sha256(source_summary.read_bytes()).hexdigest()
    cohort={str(c['id']):c for c in load(OUT/'cohort.json')}
    corpus={str(d['doc_id']):d for d in map(json.loads,(ROOT/'outputs/xai/data/corpus.jsonl').read_text(encoding='utf-8').splitlines())}
    rows=[r for shard in range(3) for r in load(OUT/f'shard-{shard}/rows.json')]
    groups={name:{r['claim_id']:r for r in rows if r['condition']==name} for name in ['baseline','yuyue_frozen']}
    for r in rows:r['baseline_correct']=groups['baseline'][r['claim_id']]['correct']
    reg=[r for r in groups['yuyue_frozen'].values() if r['baseline_correct'] and not r['correct']];assert len(reg)==21
    chosen=select_review_cases(reg,rows);ids=set(r['claim_id'] for r in reg)|{cid for cid,_ in chosen};diagnostics=[];pack=[];full_details=[]
    css='body{font:16px/1.55 system-ui,sans-serif;margin:0;background:#f4f6f8;color:#172838}main{max-width:1100px;margin:auto;padding:32px}article,.card{background:white;border:1px solid #d4dce3;border-radius:10px;padding:22px;margin:20px 0}h1,h2{line-height:1.2}a{color:#145b92}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{padding:10px;border:1px solid #d4dce3;text-align:left;vertical-align:top}.gold{background:#fff1bf}blockquote{border-left:4px solid #527f97;padding-left:16px;white-space:pre-wrap;overflow-wrap:anywhere}.muted{color:#526574}details{padding:10px 0}code{overflow-wrap:anywhere}nav{display:flex;gap:20px;flex-wrap:wrap}@media print{body{background:white}main{padding:0}article{break-inside:avoid}nav{display:none}}'
    def page(title,body):return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+html.escape(title)+'</title><style>'+css+'</style><main>'+body+'</main></html>'
    for cid in sorted(ids,key=int):
        bpath=next(OUT.glob('shard-*/'+cid+'-baseline-trace.json'));b=load(bpath);c=load(bpath.with_name(cid+'-yuyue_frozen-trace.json'))
        assert b['paper_ids']==c['paper_ids'];claim=cohort[cid]
        bn=VerificationResponse.model_validate(b['response']);cn=VerificationResponse.model_validate(c['response'])
        br=groups['baseline'][cid];cr=groups['yuyue_frozen'][cid]
        candidate_ids={p.paper_id for p in cn.evidence_passages};baseline_ids={p.paper_id for p in bn.evidence_passages}
        scope=reaggregate_scope(bn,candidate_ids);gate_sufficient=scope!=cr['gold']
        bp={p.paper_id:p for p in bn.per_paper_verdicts};cp={p.paper_id:p for p in cn.per_paper_verdicts}
        gold_docs=set(claim.get('evidence',{}));dropped=sorted(baseline_ids-candidate_ids)
        source_checks=0
        for response in [bn,cn]:
            for p in response.evidence_passages:
                parts=p.passage_id.split(':');a,z=int(parts[-3]),int(parts[-2]);assert ' '.join(corpus[p.paper_id]['abstract'])[a:z]==p.text;source_checks+=1
        detail={'claim_id':cid,'claim':claim['claim'],'gold':cr['gold'],'baseline':br['prediction'],'candidate':cr['prediction'],
          'is_regression':br['correct'] and not cr['correct'],'scope_only_counterfactual':scope,
          'diagnosis':('gate_sufficient_to_lose_focused_correctness' if gate_sufficient else 'retained_context_predictions_needed_for_change') if br['correct'] and not cr['correct'] else ('recovered_baseline_error' if cr['correct'] else 'persistent_error_not_a_regression'),
          'candidate_retained_papers':sorted(candidate_ids),'dropped_papers':dropped,'dropped_gold_papers':sorted(set(dropped)&gold_docs),
          'gold_papers_retrieved':sorted(gold_docs&set(b['paper_ids'])),'gold_papers_total':len(gold_docs),
          'baseline_gold_sentence_tp':br['evidence']['tp'],'candidate_gold_sentence_tp':cr['evidence']['tp'],
          'paper_changes':{pid:{'baseline':bp[pid].verdict.value,'candidate':cp[pid].verdict.value,
            'baseline_max_cosine':max((p.score for p in bn.evidence_passages if p.paper_id==pid),default=None),
            'candidate_passages':sum(p.paper_id==pid for p in cn.evidence_passages)} for pid in b['paper_ids'] if bp[pid].verdict!=cp[pid].verdict or pid in dropped},
          'source_span_checks':source_checks,'notes':'Scope-only reuses original recorded predictions on candidate-retained papers. It is a diagnostic counterfactual, not a new benchmark result or proof of sole cause. No new NLI calls.'}
        if detail['is_regression']:diagnostics.append(detail)
        full_details.append(detail)
        body='<nav><a href="../index.html">Review guide</a><a href="../ALL_REGRESSIONS.md">All21 regression summary</a></nav><h1>Claim '+cid+'</h1><p>'+html.escape(claim['claim'])+'</p>'
        body+='<table><tr><th>Benchmark reference</th><th>Original focused verdict</th><th>Primary candidate verdict</th><th>Focused predictions, candidate paper scope</th></tr><tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in [cr['gold'],br['prediction'],cr['prediction'],scope])+'</tr></table>'
        if cr['gold']=='INSUFFICIENT':body+='<p><strong>No-positive benchmark case:</strong> absence of an annotated rationale is not proof that every retrieved passage is scientifically irrelevant. Human semantic judgment remains open.</p>'
        body+='<p class="muted">Same frozen top-five papers in both versions. Source text and predictions below are recorded actual-model outputs. Model confidence is uncalibrated. This review is diagnostic and intentionally shows benchmark labels.</p><p><b>Diagnostic:</b> '+html.escape(detail['diagnosis'])+'</p>'
        for name,native,raw in [('baseline',bn,b),('yuyue_frozen',cn,c)]:
            evaluator=object.__new__(ClaimVerifier);adapter=Person4Adapter(lambda response,subset_of_paper_ids:safe_aggregate(evaluator,response,subset_of_paper_ids))
            empty=[pid for pid in raw['paper_ids'] if not any(p.paper_id==pid for p in native.evidence_passages)]
            snap=adapter.snapshot(native,[Paper(paper_id=pid,title=corpus[pid]['title'],source_status='unverified',source_version=protocol['data_sha256']['corpus.jsonl']) for pid in raw['paper_ids']],mode='cached',abstained_paper_ids=empty)
            data=snap.model_dump(mode='json');data['filters'].update(abstract_policy='baseline' if name=='baseline' else 'yuyue_primary',evaluation_scope='Frozen209 internal audit; full-corpus RRFtop5; not fresh web search',claim_id=cid,replay_packaging='Timestamp records packaging; predictions are reused from prior audit, no new inference.')
            data['filters']['context_configuration_json']=json.dumps({'policy':'bounded_abstract' if name=='yuyue_frozen' else 'semantic_focused','min_paper_score':.65 if name=='yuyue_frozen' else None,'base_selection':{'top_k':3,'min_score':.5,'context_radius':0},'max_pair_tokens':512,'scope':'Original frozen audit configuration; not fresh inference'})
            snap=RunSnapshot.model_validate(data);ctx=make_context(snap,raw['paper_ids'],snap.provisional);ex=build_template(ctx);assert validate_explanation(ex,ctx).displayable
            bundle=export_snapshot(ctx,ex);assert import_snapshot(bundle).snapshot.fingerprint==snap.fingerprint
            filename=cid+'-'+name+'.json';(replays/filename).write_text(bundle,encoding='utf-8')
            body+='<p><a href="../replays/'+filename+'">Download '+name+' replay</a></p>'
        for pid in b['paper_ids']:
            doc=corpus[pid];gold_sent={i for e in claim.get('evidence',{}).get(pid,[]) for i in e['sentences']}
            body+='<article id="paper-'+pid+'"><h2>Corpus document '+pid+': '+html.escape(doc['title'])+'</h2><p>Paper verdict: '+bp[pid].verdict.value+' → '+cp[pid].verdict.value+'. '+('Gate removed all candidate passages.' if pid in dropped else '')+'</p>'
            body+='<details><summary>Original abstract; yellow marks benchmark rationale sentences</summary><ol>'
            for i,text in enumerate(doc['abstract']):body+='<li'+(' class="gold"' if i in gold_sent else '')+'>'+html.escape(text)+'</li>'
            body+='</ol></details>'
            for label,response in [('Focused baseline',bn),('Primary context',cn)]:
                labels={p.passage_id:p for p in response.per_passage_labels}
                ps=[p for p in response.evidence_passages if p.paper_id==pid];body+='<h3>'+label+'</h3>'
                if not ps:body+='<p>No accepted passage.</p>'
                for p in ps:
                    pred=labels[p.passage_id];body+='<details><summary>'+pred.label.value+' · model confidence '+f'{pred.confidence:.3f}'+' · cosine '+f'{p.score:.3f}'+'</summary><code>'+html.escape(p.passage_id)+'</code><blockquote>'+html.escape(p.text)+'</blockquote></details>'
            body+='</article>'
        (cases/(cid+'.html')).write_text(page('Claim '+cid,body),encoding='utf-8')
    counts=Counter(d['diagnosis'] for d in diagnostics);transitions=Counter((d['gold'],d['candidate']) for d in diagnostics)
    assert len(diagnostics)==21 and counts['gate_sufficient_to_lose_focused_correctness']==10
    for order,(cid,reason) in enumerate(chosen,1):
        r=groups['yuyue_frozen'][cid];pack.append({'review_order':order,'case_id':cid,'claim':r['claim'],'reference_label':r['gold'],'baseline':groups['baseline'][cid]['prediction'],'candidate':r['prediction'],'selection_reason':reason,'case_page':'cases/'+cid+'.html',
          'evidence_supports_verdict':'','citation_correct':'','context_helpful':'','reviewer_name':'','issue_notes':''})
    pdf=ROOT/'evaluation/validation/results/replays/real-arxiv-replay.json'
    if pdf.exists():
        shutil.copy2(pdf,replays/'pdf-ig-paper.json');pdf_bundle=import_snapshot(pdf.read_bytes())
        body='<nav><a href="../index.html">Review guide</a></nav><h1>PDF source check</h1><p>'+html.escape(pdf_bundle.snapshot.claim)+'</p><p>Reference verdict: not adjudicated. This is an extraction, page, citation and replay review, not an accuracy-scored example.</p><p><a href="https://arxiv.org/pdf/1703.01365">Open original scientific PDF</a> · <a href="../replays/pdf-ig-paper.json">Download saved PDF replay</a></p>'
        for passage in pdf_bundle.snapshot.verification.evidence_passages:body+='<article><h2>Recorded page '+str(passage.page)+'</h2><p>Check the complete passage in the replay against this page of the original PDF. Check reading order and whether the quotation supports the displayed verdict.</p></article>'
        (cases/'pdf.html').write_text(page('PDF review',body),encoding='utf-8')
        pack.append({'review_order':len(pack)+1,'case_id':'pdf-ig-paper','claim':pdf_bundle.snapshot.claim,'reference_label':'NOT_ADJUDICATED','baseline':'recorded existing PDF policy','candidate':pdf_bundle.snapshot.provisional.aggregated_verdict,'selection_reason':'Real PDF page/citation/reading-order inspection; PDF selector unchanged','case_page':'cases/pdf.html','evidence_supports_verdict':'','citation_correct':'','context_helpful':'','reviewer_name':'','issue_notes':''})
    write_csv(DEST/'REGRESSIONS.csv',diagnostics);write_csv(DEST/'REVIEW_FORM.csv',pack)
    (DEST/'case_diagnostics.json').write_text(json.dumps(full_details,indent=2),encoding='utf-8')
    summary={'date':'2026-11-11','regressions':21,'transition_counts':{a+' → '+b:n for (a,b),n in transitions.items()},'diagnostic_counts':dict(counts),
      'retrieval_order_agreement':21,'cases_missing_annotated_gold_papers':sum(d['gold_papers_total']>0 and not d['gold_papers_retrieved'] for d in diagnostics),
      'regressions_with_dropped_gold_papers':sum(bool(d['dropped_gold_papers']) for d in diagnostics),
      'baseline_correct_without_any_gold_sentence':sum(d['gold_papers_total']>0 and d['baseline_gold_sentence_tp']==0 for d in diagnostics),
      'review_cases':len(pack),'review_selection':'One minimum-ID case per regression transition, one recovered SUPPORT, one recovered INSUFFICIENT, one persistent CONTRADICTED error, and one unadjudicated PDF. Diagnostic sample, not representative performance estimate.',
      'human_ratings_completed':0,'new_model_calls':0,'policy_or_threshold_changes':False,'frozen_audit_summary_sha256':before_hash,
      'limitations':['Scope-only counterfactual isolates paper removal on recorded baseline predictions, not a sole-cause proof.','Context diagnosis does not identify causative words or establish that benchmark-unannotated evidence is false.','The reserved audit is now inspected; future policy work must use separate development data.']}
    (DEST/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    md=['# Regression investigation — 11 November 2026','', 'The primary policy and frozen audit are unchanged. This report explains all21 regressions using recorded native predictions; **zero new model calls and zero threshold/model changes**.','',
      '## Findings','', '- **10/21: paper removal is sufficient to lose focused correctness.** Reaggregating the original predictions using only papers retained by the new candidate changes a correct verdict into an incorrect one. Context can also contribute; this does not establish the gate as the sole cause.',
      '- **11/21: retained-context predictions are needed for the changed result.** Reaggregating focused predictions on the candidate paper scope remains correct; the actual context-based predictions change the result.',
      '- **Retrieval order is identical in21/21.** No retrieval change explains these paired regressions. Retrieval quality can still limit both versions.',
      f"- {summary['regressions_with_dropped_gold_papers']} regressions lose at least one previously accepted annotated-gold paper. {summary['cases_missing_annotated_gold_papers']} have no annotated gold paper in either top-five scope. {summary['baseline_correct_without_any_gold_sentence']} positive cases had a correct baseline label without selecting any complete annotated gold sentence.",
      '', '| Benchmark → primary verdict | Cases |','|---|---:|']
    for (a,b),n in sorted(transitions.items()):md.append(f'| {a} → {b} | {n} |')
    md+=['','Four no-positive cases become Supported. These are benchmark errors, but absent annotations do not establish that the retrieved evidence is scientifically wrong; human review must inspect the sources.','',
      '## Review first','', 'Open **index.html** for the9-case guide, then fill **REVIEW_FORM.csv**. The first eight are deterministic diagnostic claim cases; the last is a real PDF source check. All21 regressions have complete source pages and baseline/primary replay JSONs. No human ratings are filled in.','',
      '## Responsibilities','', '| Owner | Check | Status |','|---|---|---|',
      '| Yu Yue | Inspect gate-discarded annotated papers and whether whole-abstract context hides qualifying sentences; propose experiments only on development data. | PENDING human review |',
      '| Chong Yao | Inspect retained-paper NLI changes, especially Contradicted → Supported/Conflicting, and confidence despite wrong benchmark outcomes. | PENDING human review |',
      '| Naila | Inspect sentence relevance and explain model sensitivity on development examples; rankings did not change in these regressions. | PENDING human review |',
      '| Raghavi | Test fresh discovery/provenance/provider failures separately; this offline regression comparison does not evaluate web recall. | PENDING live testing |',
      '| Kireeti | Compare exact source text, displayed scope, quotations, export/replay and the PDF pages. Automated source/replay checks pass; semantic ratings remain blank. | PENDING human review |','',
      '## Interpretation','', 'No confirmed frontend correctness defect was found in these traces: recorded source spans match the corpus and replay/grounding checks pass. This does not validate the scientific meaning. These losses arise under the configured gate/context policy; do not patch or tune that policy using this inspected audit.',
      '', 'The guard experiment is retained as a comparison, not selectively substituted for failed claims. Overall audit remains45 fixes /21 regressions; no improved accuracy is claimed from this diagnostic report.']
    md=[s.replace('all21','all 21').replace('in21/21','in 21/21').replace('the9-case','the 9-case').replace('All21','All 21').replace('remains45','remains 45').replace('/21','/21') for s in md]
    (DEST/'ALL_REGRESSIONS.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    index='<h1>Review9 recorded outputs</h1><p>Start here; allow about30–45 minutes. These are deliberately chosen diagnostic cases, not a representative sample or proof of human usefulness.</p><nav><a href="ALL_REGRESSIONS.md">All21 findings and owner checklist</a><a href="REVIEW_FORM.csv">Blank review form</a><a href="HOW_TO_REVIEW.md">Rating instructions</a><a href="REGRESSIONS.csv">All21 case diagnostics</a></nav><p>For each case: compare the claim and benchmark reference, open source abstracts and both passage sets, then judge the primary verdict. Check the reference label critically, especially no-positive cases. Import replay JSONs in the app if you want to inspect scoped verdicts.</p>'
    for r in pack:index+='<article><h2>'+str(r['review_order'])+'. '+html.escape(r['case_id'])+'</h2><p>'+html.escape(r['claim'])+'</p><p>'+html.escape(r['selection_reason'])+'</p><p><a href="'+r['case_page']+'">Open source comparison</a></p></article>'
    index+='<h2>Every regression</h2><p>'+ ' · '.join('<a href="cases/'+r['claim_id']+'.html">'+r['claim_id']+'</a>' for r in sorted(diagnostics,key=lambda d:int(d['claim_id'])))+'</p><p>No settings changed. Human ratings completed:0. All source text is escaped; the pages load without external scripts, fonts or API requests. The PDF link opens the original source only when clicked.</p>'
    (DEST/'index.html').write_text(page('Regression review guide',index.replace('Review9','Review 9').replace('about30','about 30').replace('All21','All 21')),encoding='utf-8')
    assert hashlib.sha256(source_summary.read_bytes()).hexdigest()==before_hash
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.rglob('*') if p.is_file() and p.name!='ARTIFACTS.json'}
    (DEST/'ARTIFACTS.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    output=ROOT/'outputs/regression_review_20261111';output.mkdir(exist_ok=True)
    with zipfile.ZipFile(output/'review-kit.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in DEST.rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(DEST)))
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
