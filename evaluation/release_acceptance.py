"""Actual-model fixed-candidate acceptance, not a retrieval-quality experiment."""
import json,hashlib,random,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]


def discovery_for(claim,documents):
    from research_verifier.discovery.service import DiscoverySnapshot
    papers=[];records=[]
    for rank,doc in enumerate(documents,1):
        pid='scifact:'+str(doc['doc_id'])
        paper=dict(title=doc['title'],abstract=' '.join(doc['abstract']),authors=[],source='scifact-benchmark',source_id=str(doc['doc_id']))
        papers.append(dict(identity=pid,identity_aliases=[pid],original_claim=claim,paper=paper,
            provenance=[dict(provider='scifact-benchmark',source_id=str(doc['doc_id']),query_id='fixed',rank=rank)],source_records=[paper]))
        records.append(dict(paper=paper,query_id='fixed',rank=rank))
    return DiscoverySnapshot.model_validate(dict(mode='cached',imported_at=datetime.now(timezone.utc).isoformat(),
        source_reference='SciFact dev fixed candidate acceptance: known annotated paper plus2 seeded distractors; not live discovery',
        result=dict(original_claim=claim,top_k=len(documents),papers=papers,providers={'scifact-benchmark':dict(provider='scifact-benchmark',status='success',
            queries=[dict(id='fixed',provider='scifact-benchmark',strategy='fixed candidate engineering test',stage='evaluation',query=claim,status='success')],records=records)}),
        selected_paper_ids=[p['identity'] for p in papers]))


def main():
    from research_verifier.pipeline import LocalPipeline, local_verifier
    from research_verifier.contracts.models import make_context
    from research_verifier.explanations.templates import build_template
    from research_verifier.explanations.grounding import validate_explanation
    from research_verifier.explanations.replay import export_snapshot, import_snapshot
    from research_verifier.interpretability.metrics import build_examples
    from evaluation.xai_qc import sample_examples
    data=ROOT/'outputs/xai/data'
    corpus={str(d['doc_id']):d for d in [json.loads(line) for line in (data/'corpus.jsonl').read_text(encoding='utf-8').splitlines()]}
    examples=build_examples(str(data)); selected,_=sample_examples(examples,local_verifier().tokenizer,8,512,20261109)
    rng=random.Random(20261109);out=ROOT/'outputs/release';out.mkdir(parents=True,exist_ok=True)
    rows=[];start=time.perf_counter()
    for index,ex in enumerate(selected):
        t=time.perf_counter();row={'claim_id':ex['claim_id'],'gold_document_label':ex['gold_label']}
        try:
            distractors=rng.sample([d for key,d in corpus.items() if key!=ex['doc_id']],2)
            source=discovery_for(ex['claim'],[corpus[ex['doc_id']],*distractors])
            engine=LocalPipeline();snapshot=engine.run_from_discovery(source,source.selected_paper_ids)
            context=make_context(snapshot,source.selected_paper_ids,snapshot.provisional)
            explanation=build_template(context);gate=validate_explanation(explanation,context)
            if not gate.displayable:raise ValueError('Grounding failed')
            bundle=export_snapshot(context,explanation)
            restored=import_snapshot(bundle)
            if restored.snapshot.fingerprint!=snapshot.fingerprint:raise ValueError('Replay mismatch')
            for subset in (source.selected_paper_ids[:1],source.selected_paper_ids[1:],[]):
                aggregate=engine.aggregate(snapshot,subset)
                subset_context=make_context(snapshot,subset,aggregate)
                if not validate_explanation(build_template(subset_context),subset_context).displayable:raise ValueError('Subset grounding failed')
            path=out/f'scifact-{ex["claim_id"]}.json';path.write_text(bundle,encoding='utf-8')
            row.update(status='passed',verdict=snapshot.provisional.aggregated_verdict,run_status=snapshot.provisional.status,
                passages=len(snapshot.verification.evidence_passages),export_path=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),claim=ex['claim'])
        except Exception as exc:row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        row['seconds']=time.perf_counter()-t;rows.append(row);print(json.dumps(row),flush=True)
    summary={'date':'2026-11-09','protocol':'8 seeded balanced SciFact dev claim/document examples, known annotated paper plus2 random distractors each. Actual retrieval/ranking/semantic/NLI/template/subset/replay; imported corpus boundary replaces external discovery.',
        'not_a_quality_score':'Annotated paper is deliberately included; not retrieval accuracy or representative claim verification accuracy. Every case retained.',
        'attempted':len(rows),'passed':sum(r['status']=='passed' for r in rows),'cases':rows,'seconds':time.perf_counter()-start}
    (out/'acceptance.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':summary['passed'],'attempted':summary['attempted']}),flush=True)
    return summary
if __name__=='__main__':main()
