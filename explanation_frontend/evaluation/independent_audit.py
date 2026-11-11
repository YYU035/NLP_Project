"""Frozen unused-claim, cited-paper-disjoint internal audit; not official test data."""
import json,hashlib,time,argparse
from pathlib import Path
from .context_study.freeze_train import ROOT
from .validation.scifact import gold_label
from .context_study.run_local import evidence_metrics
OUT=ROOT/'outputs/independent_audit_20261111'

def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'PROTOCOL.json').exists():raise ValueError('Cohort already frozen; do not overwrite')
    train=list(map(json.loads,(ROOT/'outputs/context_study_20261111/claims_train.jsonl').read_text(encoding='utf-8').splitlines()))
    dev=list(map(json.loads,(ROOT/'outputs/xai/data/claims_dev.jsonl').read_text(encoding='utf-8').splitlines()))
    used=set(load(ROOT/'evaluation/context_study/TRAIN_FROZEN.json')['train_ids'])|{str(c['id']) for c in train[:200]}
    seen=[c for c in train if str(c['id']) in used]+dev
    def papers(c):return {str(p) for p in [*c.get('evidence',{}),*c.get('cited_doc_ids',[])]}
    seen_papers=set().union(*(papers(c) for c in seen));texts={c['claim'].strip().casefold() for c in seen}
    cohort=sorted([c for c in train if str(c['id']) not in used and c['claim'].strip().casefold() not in texts and not papers(c)&seen_papers],key=lambda c:hashlib.sha256(str(c['id']).encode()).hexdigest())
    from collections import Counter
    record={'date':'2026-11-11','scope':'Previously unused internal holdout from published SciFact train; NOT official test or external-domain validation. Related statements and model pretraining exposure cannot be ruled out.',
      'sampling':'All eligible claims, no label-dependent selection. Exclude prior SHA200 training IDs, conservatively also first200 raw train IDs, all inspected dev exact normalized texts, and ALL their cited/evidence paper IDs.',
      'claim_ids':[str(c['id']) for c in cohort],'n':len(cohort),'labels':dict(Counter(gold_label(c) for c in cohort)),
      'data_sha256':{n:hashlib.sha256((ROOT/'outputs/xai/data'/n).read_bytes()).hexdigest() for n in ['corpus.jsonl','claims_dev.jsonl']},
      'train_sha256':hashlib.sha256((ROOT/'outputs/context_study_20261111/claims_train.jsonl').read_bytes()).hexdigest(),
      'candidate':'Yu Yue bounded exact abstract, .65 paper cosine gate, original anchors .5/top3, actual512-token pair budget; frozen before this cohort. No model/threshold changes.',
      'retrieval':'Full5183 corpus, production encoder2a7, Porter BM25, RRF60, top100 branches, verifytop5; no gold injection.',
      'assessment':'Compare accuracy paired95% bootstrap and macroF1; report ALL per-class precision/recall and input evidence tradeoffs. No policy selection or tuning from this audit.',
      'promotion_rule':'Overall paired accuracy interval must exclude zero, macroF1 must improve, and zero operational/source/budget failures. Any independent contradiction-recall loss remains an explicit limitation; no universal-reliability claim.',
      'shards':3,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'PROTOCOL.json').write_text(json.dumps(record,indent=2),encoding='utf-8');(OUT/'cohort.json').write_text(json.dumps(cohort),encoding='utf-8')
    print('Frozen cohort',record['n'],record['labels'],flush=True)
    from .validation.retrieval import rank_dataset
    from ..pipeline import local_encoder
    docs={str(d['doc_id']):d for d in map(json.loads,(ROOT/'outputs/xai/data/corpus.jsonl').read_text(encoding='utf-8').splitlines())}
    queries={str(c['id']):{'text':c['claim'],'candidate_ids':list(docs)} for c in cohort}
    ranks=rank_dataset(docs,queries,local_encoder(),stemmer='porter',model_key='production-MiniLM2a7-seq256',branch_depth=100)
    (OUT/'rankings.json').write_text(json.dumps({q:ids[:5] for q,ids in ranks['rrf'].items()}),encoding='utf-8')

def run(shard):
    import torch
    torch.set_num_threads(4)
    from ..pipeline import local_verifier,safe_aggregate
    from ..semantic_integration import local_selector
    from .context_study.context_adapter import BoundedAbstractSelector
    from .._upstream.person3.verifier_budget import enforce_verifier_budget
    from .._upstream.person3.semantic_evidence import person4_passages
    from .._upstream.claim_verifier.schemas import EvidencePassage,PassageVerdict
    from ..person4 import Person4Adapter
    from ..models import Paper,make_context
    from ..templates import build_template
    from ..validation import validate_explanation
    from ..export import export_snapshot,import_snapshot
    cohort=load(OUT/'cohort.json');ranks=load(OUT/'rankings.json');docs={str(d['doc_id']):d for d in map(json.loads,(ROOT/'outputs/xai/data/corpus.jsonl').read_text(encoding='utf-8').splitlines())}
    v=local_verifier();base=local_selector();context=BoundedAbstractSelector(base,v.tokenizer,min_paper_score=.65)
    directory=OUT/f'shard-{shard}';directory.mkdir(exist_ok=True);cache=directory/'nli';cache.mkdir(exist_ok=True);rows=[];calls=0;hits=0;original=v._classify_passage
    def cached(claim,passage):
        nonlocal calls,hits
        key=hashlib.sha256(json.dumps([claim,passage.text],ensure_ascii=False).encode()).hexdigest();path=cache/(key+'.json')
        if path.exists():result=PassageVerdict.model_validate_json(path.read_text(encoding='utf-8'));hits+=1
        else:
            result=original(claim,passage);calls+=1
            if result.truncated or result.operational_status.value!='OK':raise ValueError('Unusable audit model prediction')
            path.write_text(result.model_dump_json(),encoding='utf-8')
        return result.model_copy(update={'passage_id':passage.passage_id,'paper_id':passage.paper_id})
    v._classify_passage=cached
    try:
        for index,claim in enumerate(cohort):
            if index%3!=shard:continue
            cid=str(claim['id']);pids=ranks[cid]
            for name,selector in [('baseline',base),('yuyue_frozen',context)]:
                passages={};spans={};abstained=[];checks=[]
                for pid in pids:
                    source=' '.join(docs[pid]['abstract']);chosen=selector.select_abstract(claim['claim'],pid,source)
                    assert all(source[s.start:s.end]==s.text for s in chosen)
                    budget=enforce_verifier_budget(claim['claim'],chosen,v.tokenizer)
                    if budget.rejected:raise ValueError('Audit budget rejection')
                    checks.extend(budget.checked);spans[pid]=list(budget.accepted)
                    if not chosen:abstained.append(pid)
                    passages[pid]=[EvidencePassage.model_validate(p) for p in person4_passages(budget.accepted)]
                response=v.verify(claim['claim'],[{'paper_id':pid,'title':docs[pid]['title']} for pid in pids],passages)
                adapter=Person4Adapter(lambda response,subset_of_paper_ids:safe_aggregate(v,response,subset_of_paper_ids))
                snapshot=adapter.snapshot(response,[Paper(paper_id=pid,title=docs[pid]['title'],source_status='unverified') for pid in pids],mode='cached',abstained_paper_ids=abstained)
                if snapshot.verification.processing_failures:raise ValueError('Audit processing failure')
                ctx=make_context(snapshot,pids,snapshot.provisional);ex=build_template(ctx)
                if not validate_explanation(ex,ctx).displayable:raise ValueError('Grounding failed')
                prediction=snapshot.provisional.aggregated_verdict
                row={'claim_id':cid,'claim':claim['claim'],'gold':gold_label(claim),'condition':name,'prediction':prediction,'correct':prediction==gold_label(claim),'status':snapshot.provisional.status,**evidence_metrics(claim,docs,spans)}
                rows.append(row)
                (directory/f'{cid}-{name}-trace.json').write_text(json.dumps({'claim_id':cid,'paper_ids':pids,'budget':checks,'response':response.model_dump(mode='json')}),encoding='utf-8')
                if index<15:
                    bundle=export_snapshot(ctx,ex);assert import_snapshot(bundle).snapshot.fingerprint==snapshot.fingerprint
                    (directory/f'{cid}-{name}-replay.json').write_text(bundle,encoding='utf-8')
            (directory/'rows.json').write_text(json.dumps(rows),encoding='utf-8')
            if len(rows)%20==0:print('shard',shard,'claims',len(rows)//2,'new NLI',calls,'cache',hits,flush=True)
    finally:v._classify_passage=original
    (directory/'complete.json').write_text(json.dumps({'rows':len(rows),'new_nli':calls,'cache_hits':hits}),encoding='utf-8');print('COMPLETE',shard,len(rows),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run']);ap.add_argument('--shard',type=int,choices=[0,1,2],default=0);a=ap.parse_args()
    prepare() if a.stage=='prepare' else run(a.shard)
