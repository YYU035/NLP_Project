"""Independent checks on completed experiment accounting and metric calculations."""
import json,csv,hashlib
from collections import Counter
from sklearn.metrics import accuracy_score,f1_score,ndcg_score
import numpy as np
from .freeze import ROOT,OUT,freeze

def main():
    freeze();rows=json.loads((OUT/'scifact_full/case_rows.json').read_text(encoding='utf-8'));summary=json.loads((OUT/'scifact_full/summary.json').read_text(encoding='utf-8'))
    checks=[]
    def check(name,passed):
        checks.append({'check':name,'passed':bool(passed)})
        if not passed:raise ValueError('Artifact audit failed: '+name)
    counts=Counter(r['condition'] for r in rows)
    check('Complete300 main +188 oracle +188 gold-paper condition rows',dict(counts)=={'oracle_gold_rationale':188,'gold_papers_production_selector':188,'retrieved_papers_production_selector':300})
    check('Unique claim/condition IDs',len({(r['claim_id'],r['condition']) for r in rows})==len(rows))
    check('All676 grounded template gates',all(r.get('grounding_passed') for r in rows))
    check('No processing failures',all(r['status']!='failed' and not r.get('processing_failures') for r in rows))
    for condition in counts:
        group=[r for r in rows if r['condition']==condition];gold=[r['gold'] for r in group];pred=[r['prediction'] for r in group]
        actual=summary['conditions'][condition]['all'];labels=actual['macro_classes']
        check(condition+' accuracy matches sklearn',abs(accuracy_score(gold,pred)-actual['accuracy'])<1e-12)
        check(condition+' macroF1 matches sklearn',abs(f1_score(gold,pred,labels=labels,average='macro',zero_division=0)-actual['macro_f1'])<1e-12)
    from .dataset_adapters.csfcube import load as cs_load
    from .dataset_adapters.reciper import load as rec_load
    from .metrics import ranking_metrics
    datasets=[cs_load(OUT/'data/csfcube',split=s) for s in ['fold1_dev','fold1_test']]+[rec_load(OUT/'data/reciper')]
    for dataset in datasets:
        split=dataset['protocol'].get('split','entire_release_zero_shot');ranked=json.loads((OUT/f"{dataset['dataset']}-{split}-rankings.json").read_text(encoding='utf-8'))
        for method,queries in ranked.items():
            check(dataset['dataset']+' '+split[:10]+' '+method+' complete query IDs',set(queries)==set(dataset['queries']))
            check(dataset['dataset']+' '+method+' unique rankings use exact declared pools',all(len(order)==len(set(order)) and set(order)==set(dataset['queries'][qid]['candidate_ids']) for qid,order in queries.items()))
            for qid in list(queries)[:5]:
                order=queries[qid];grades=dataset.get('grades',{}).get(qid)
                relevant={p for p,g in grades.items() if g>=2} if grades is not None else set(dataset['qrels'][qid])
                metrics=ranking_metrics(order,relevant,grades)
                if metrics is None:continue
                scores=np.arange(len(order),0,-1,dtype=float)[None,:]
                binary=np.array([[int(p in relevant) for p in order]])
                check(dataset['dataset']+' '+method+' '+qid+' nDCG sklearn',abs(ndcg_score(binary,scores,k=10)-metrics['nDCG@10'])<1e-12)
                if grades is not None:
                    gain=np.array([[2**grades.get(p,0)-1 for p in order]])
                    check('Graded nDCG sklearn '+method+' '+qid,abs(ndcg_score(gain,scores,k=10)-metrics['graded_nDCG@10'])<1e-12)
    files=['scifact_full/case_rows.json','scifact_full/case_details.jsonl','scifact_full/rankings.json','scifact_full/calibration_rows.json','other_datasets.json','retrieval_per_query.csv','stress.json','reliability.json','clean_setup.json']
    report={'checks':checks,'passed':sum(r['passed'] for r in checks),'total':len(checks),'artifact_sha256':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in files}}
    (OUT/'artifact_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({'audit_passed':report['passed'],'total':report['total']}),flush=True)
if __name__=='__main__':main()
