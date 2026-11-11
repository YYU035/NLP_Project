"""Full5183-paper SciFact experiment through real selector/NLI/aggregation, no gold injection."""
import argparse,csv,json,time,hashlib,statistics
from pathlib import Path
from dataclasses import asdict
from evaluation.validation.freeze import ROOT, OUT, freeze
from evaluation.validation.retrieval import rank_dataset
from evaluation.validation.metrics import ranking_metrics, classification, evidence_score, bootstrap, reliability


def gold_label(claim):
    labels={e['label'] for group in claim.get('evidence',{}).values() for e in group}
    if not labels:return 'INSUFFICIENT'
    if len(labels)!=1:raise ValueError('Mixed claim gold labels require a separate protocol')
    return {'SUPPORT':'SUPPORTED','CONTRADICT':'CONTRADICTED'}[next(iter(labels))]


def sentence_recovery(sentences,spans):
    """Whole original sentences only; partial fragments do not receive full-sentence credit."""
    full=set();partial=set();offset=0
    intervals=sorted((s.start,s.end) for s in spans)
    for index,sentence in enumerate(sentences):
        start=offset+len(sentence)-len(sentence.lstrip());end=offset+len(sentence.rstrip());offset+=len(sentence)+1
        cursor=start;overlap=False
        for a,b in intervals:
            if a<end and b>start:overlap=True
            if a<=cursor:cursor=max(cursor,b)
        if end>start and cursor>=end:full.add(index)
        elif overlap:partial.add(index)
    return full,partial


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=300);ap.add_argument('--output',default='scifact_full');args=ap.parse_args(argv)
    if not 1<=args.limit<=300:ap.error('limit must be1..300')
    baseline=freeze();directory=(OUT/args.output).resolve()
    if not directory.is_relative_to(OUT.resolve()):raise ValueError('Output path escape')
    directory.mkdir(parents=True,exist_ok=True)
    from research_verifier.pipeline import local_encoder, local_verifier, safe_aggregate
    from research_verifier.evidence.semantic import local_selector
    from research_verifier.evidence.budget import enforce_verifier_budget
    from research_verifier.evidence.spans import EvidenceSpan
    from research_verifier.evidence.selector import person4_passages
    from research_verifier.verification.schemas import EvidencePassage
    from research_verifier.verification.adapter import VerifierAdapter
    from research_verifier.contracts.models import Paper, RunSnapshot, make_context
    from research_verifier.explanations.templates import build_template
    from research_verifier.explanations.grounding import validate_explanation
    from research_verifier.explanations.replay import export_snapshot, import_snapshot
    import torch
    torch.set_num_threads(6)
    corpus_rows=[json.loads(line) for line in (ROOT/'outputs/xai/data/corpus.jsonl').read_text(encoding='utf-8').splitlines()]
    corpus={str(d['doc_id']):d for d in corpus_rows}
    claims=[json.loads(line) for line in (ROOT/'outputs/xai/data/claims_dev.jsonl').read_text(encoding='utf-8').splitlines()][:args.limit]
    queries={str(c['id']):{'text':c['claim'],'candidate_ids':list(corpus)} for c in claims}
    ranking_path=directory/'rankings.json'
    if ranking_path.exists():rankings=json.loads(ranking_path.read_text(encoding='utf-8'))
    else:
        print('Index/retrieve',len(corpus),'papers;',len(claims),'claims',flush=True)
        rankings=rank_dataset(corpus,queries,local_encoder(),stemmer='porter',model_key='production-MiniLM2a7-seq256',branch_depth=100,
            progress=lambda n,total:print('Retrieval',n,'/',total,flush=True))
        rankings={m:{qid:order[:100] for qid,order in rows.items()} for m,rows in rankings.items()}
        ranking_path.write_text(json.dumps(rankings),encoding='utf-8')
    retrieval_rows=[]
    for claim in claims:
        for method in rankings:
            values=ranking_metrics(rankings[method][str(claim['id'])],claim.get('evidence',{}))
            if values is not None:retrieval_rows.append({'claim_id':claim['id'],'method':method,**values})
    v=local_verifier();selector=local_selector();cache={};cache_hits=[0];neural_calls=[0]
    original=v._classify_passage
    def cached(claim,passage):
        key=(claim,passage.text)
        if key not in cache:cache[key]=original(claim,passage);neural_calls[0]+=1
        else:cache_hits[0]+=1
        return cache[key].model_copy(update={'passage_id':passage.passage_id,'paper_id':passage.paper_id})
    v._classify_passage=cached
    rows=[];calibration=[];start=time.perf_counter();replays=directory/'replays';replays.mkdir(exist_ok=True)
    def condition(claim,pids,oracle):
        passages={};spans_by_doc={};checks=[];abstained=[];rationale_map={}
        for pid in pids:
            doc=corpus[pid];source=' '.join(doc['abstract'])
            if oracle:
                spans=[]
                for index,ev in enumerate(claim['evidence'][pid]):
                    inds=tuple(ev['sentences']);text=' '.join(doc['abstract'][i] for i in inds)
                    passage_id=f'{pid}:oracle-rationale:{index}'
                    spans.append(EvidenceSpan(pid,inds,0,len(text),text,1.,passage_id));rationale_map[passage_id]=ev['label']
            else:
                spans=selector.select_abstract(claim['claim'],pid,source)
                if any(source[s.start:s.end]!=s.text for s in spans):raise ValueError('Source-span mismatch')
            decision=enforce_verifier_budget(claim['claim'],spans,v.tokenizer)
            checks.extend({'paper_id':pid,**check} for check in decision.checked)
            accepted=list(decision.accepted);spans_by_doc[pid]=accepted
            if not spans and not decision.rejected:abstained.append(pid)
            passages[pid]=[EvidencePassage.model_validate(p) for p in person4_passages(accepted)]
        native=v.verify(claim['claim'],[{'paper_id':pid,'title':corpus[pid]['title']} for pid in pids],passages)
        metadata=[Paper(paper_id=pid,title=corpus[pid]['title'],source_status='unverified',source_version=baseline['input_sha256']['corpus.jsonl']) for pid in pids]
        adapter=VerifierAdapter(lambda response,subset_of_paper_ids:safe_aggregate(v,response,subset_of_paper_ids))
        snapshot=adapter.snapshot(native,metadata,mode='cached',abstained_paper_ids=abstained)
        if any(not check['accepted'] for check in checks):
            data=snapshot.model_dump(mode='json')
            for pid in sorted({check['paper_id'] for check in checks if not check['accepted']}):
                data['verification']['processing_failures'].append({'stage':'passage_selection','code':'VERIFIER_INPUT_LIMIT','paper_id':pid})
            if data['provisional']['status']!='failed':data['provisional']['status']='partial'
            snapshot=RunSnapshot.model_validate(data)
        gold={(str(pid),i) for pid,evs in claim.get('evidence',{}).items() for ev in evs for i in ev['sentences']}
        recovered=set();partials=set()
        for pid,spans in spans_by_doc.items():
            if oracle:recovered|={(pid,i) for s in spans for i in s.sentence_indices}
            else:
                whole,fragment=sentence_recovery(corpus[pid]['abstract'],spans)
                recovered|={(pid,i) for i in whole};partials|={(pid,i) for i in fragment}
        ev_score=evidence_score(len(recovered&gold),len(recovered-gold)+len(partials),len(gold-recovered))
        complete=[]
        for pid,evs in claim.get('evidence',{}).items():
            complete.append(any({(str(pid),i) for i in ev['sentences']}<=recovered for ev in evs))
        result={'prediction':snapshot.provisional.aggregated_verdict,'status':snapshot.provisional.status,
            'confidence':snapshot.provisional.global_confidence,'evidence':ev_score,'gold_sentences':len(gold),
            'recovered_gold_sentences':len(recovered&gold),'partial_sentence_fragments':len(partials),
            'selected_sentences':len(recovered),'passages':len(native.evidence_passages),'rejected_spans':sum(not x['accepted'] for x in checks),
            'gold_documents':len(claim.get('evidence',{})),'gold_documents_retrieved':len(set(pids)&set(claim.get('evidence',{}))),
            'complete_rationale_docs':sum(complete),'gold_rationale_docs':len(complete),
            'processing_failures':[f.model_dump(mode='json') for f in snapshot.verification.processing_failures]}
        if oracle:
            reverse={v.model.config.id2label[i]:label.value for i,label in v.label_mapping.items()}
            for pred in native.per_passage_labels:
                if pred.operational_status.value!='OK' or pred.passage_id not in rationale_map:continue
                raw=reverse[max(pred.raw_probabilities,key=pred.raw_probabilities.get)]
                calibration.append({'claim_id':claim['id'],'gold':rationale_map[pred.passage_id],'predicted':raw,
                    'correct':raw==rationale_map[pred.passage_id],'confidence':max(pred.raw_probabilities.values())})
        return result,snapshot,adapter,checks
    with (directory/'case_details.jsonl').open('w',encoding='utf-8') as details:
        for index,claim in enumerate(claims):
            row={'claim_id':claim['id'],'claim':claim['claim'],'gold':gold_label(claim)}
            for name in ('oracle_gold_rationale','gold_papers_production_selector','retrieved_papers_production_selector'):
                if name!='retrieved_papers_production_selector' and not claim.get('evidence'):
                    continue
                pids=list(claim['evidence']) if name!='retrieved_papers_production_selector' else rankings['rrf'][str(claim['id'])][:5]
                t=time.perf_counter()
                try:
                    result,snapshot,adapter,checks=condition(claim,pids,name=='oracle_gold_rationale')
                    entry={**row,'condition':name,**result,'seconds':time.perf_counter()-t}
                    entry['correct']=entry['prediction']==entry['gold']
                    context=make_context(snapshot,pids,snapshot.provisional)
                    explanation=build_template(context);entry['grounding_passed']=validate_explanation(explanation,context).displayable
                    if not entry['grounding_passed']:raise ValueError('Grounding gate failed')
                    # Inspectable replays only for the actual retrieved-paper route, never oracle concatenations.
                    if name=='retrieved_papers_production_selector' and (index<5 or (not entry['correct'] and index%10==0)):
                        data=snapshot.model_dump(mode='json');data['search_complete']=True
                        data['stage_statuses']['passages']='person3_semantic_bounded_train_frozen'
                        data['filters']['evaluation_scope']='Full5183-paper corpus, top100 per branch, RRFtop5; no gold injection; not live web discovery'
                        snapshot=RunSnapshot.model_validate(data);context=make_context(snapshot,pids,snapshot.provisional)
                        bundle=export_snapshot(context,build_template(context));assert import_snapshot(bundle).snapshot.fingerprint==snapshot.fingerprint
                        (replays/f'claim-{claim["id"]}.json').write_text(bundle,encoding='utf-8')
                    details.write(json.dumps({'claim_id':claim['id'],'condition':name,'paper_ids':pids,'budget_checks':checks,'response':snapshot.verification.model_dump(mode='json')})+'\n');details.flush()
                except Exception as exc:
                    entry={**row,'condition':name,'prediction':None,'status':'failed','error':f'{type(exc).__name__}: {exc}','seconds':time.perf_counter()-t,'correct':False}
                rows.append(entry)
                (directory/'rows_progress.json').write_text(json.dumps(rows),encoding='utf-8')
            if index%10==0 or index+1==len(claims):print('Verification',index+1,'/',len(claims),'unique NLI',neural_calls[0],'cache hits',cache_hits[0],flush=True)
    v._classify_passage=original
    summary={'claims':len(claims),'corpus':len(corpus),'settings':baseline['science_protocol'],'no_tuning':True,'elapsed_seconds':time.perf_counter()-start,
        'unique_nli_calls':neural_calls[0],'exact_input_cache_hits':cache_hits[0],'conditions':{},'calibration':reliability(calibration),
        'limitations':['SciFact dev already inspected; this is descriptive frozen evaluation, not untouched holdout.','Closed-corpus claim labels: absence of annotated evidence is benchmark INSUFFICIENT, not proof about all literature.',
        'Evidence F1 gives credit only to complete original sentences covered by production string spans; partial fragments count as extra FP. Not official SciFact leaderboard parity.',
        'Oracle rationales can concatenate noncontiguous annotated sentences; this is a privileged control, not an ordinary source passage.',
        'Top5 automatic selection is an offline evaluation wrapper; UI users can select different candidate scopes.','Exact model-input cache reuse affects runtime, not predictions; no latency comparison between conditions.']}
    for name in ('oracle_gold_rationale','gold_papers_production_selector','retrieved_papers_production_selector'):
        group=[r for r in rows if r['condition']==name];positive=[r for r in group if r['gold']!='INSUFFICIENT']
        ev=[r['evidence'] for r in positive if 'evidence' in r]
        summary['conditions'][name]={'all':classification(group),'positive_claims':classification(positive),'operational_failures':sum(r['status']=='failed' for r in group),
            'accuracy_ci':bootstrap([int(r['correct']) for r in group]),
            'evidence_micro_positive_only':evidence_score(sum(e['tp'] for e in ev),sum(e['fp'] for e in ev),sum(e['fn'] for e in ev)),
            'selected_sentences_on_no_positive_claims':sum(r.get('selected_sentences',0) for r in group if r['gold']=='INSUFFICIENT'),
            'complete_rationale_document_recall':sum(r.get('complete_rationale_docs',0) for r in positive)/sum(r.get('gold_rationale_docs',0) for r in positive) if positive else None}
    summary['retrieval']={}
    for method in rankings:
        rr=[r for r in retrieval_rows if r['method']==method]
        summary['retrieval'][method]={'eligible':len(rr),'excluded_no_positive':len(claims)-len(rr),
            **({key:statistics.mean(r[key] for r in rr) for key in rr[0] if '@' in key} if rr else {}),'recall10_ci':bootstrap([r['Recall@10'] for r in rr])}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (directory/'case_rows.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    (directory/'calibration_rows.json').write_text(json.dumps(calibration),encoding='utf-8')
    (directory/'retrieval_rows.json').write_text(json.dumps(retrieval_rows),encoding='utf-8')
    print(json.dumps({'complete':len(claims),'main_accuracy':summary['conditions']['retrieved_papers_production_selector']['all']['accuracy']}),flush=True)
    return summary
if __name__=='__main__':main()
