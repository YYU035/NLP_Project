"""Independent rerun of CSFCube dev/test and RECIPER, frozen three-method route."""
import sys,json,csv,time,statistics
from pathlib import Path
from .freeze import ROOT,OUT,freeze
from .retrieval import rank_dataset
from .metrics import ranking_metrics,bootstrap

def main():
    baseline=freeze()
    from .dataset_adapters.csfcube import load as cs_load
    from .dataset_adapters.reciper import load as rec_load
    from ...semantic_integration import local_selector
    import torch
    torch.set_num_threads(6);encoder=local_selector().encoder
    studies=[cs_load(OUT/'data/csfcube',split=s) for s in ('fold1_dev','fold1_test')]+[rec_load(OUT/'data/reciper')]
    summaries=[];per_query=[]
    for dataset in studies:
        name=dataset['dataset'];split=dataset['protocol'].get('split','entire_release_zero_shot')
        print('Rerun',name,split,len(dataset['queries']),flush=True);start=time.perf_counter()
        ranked=rank_dataset(dataset['documents'],dataset['queries'],encoder,stemmer='simple',model_key='MiniLM1110-seq256',progress=lambda n,total:print(name,split,n,'/',total,flush=True))
        clusters={qid:q.get('paper_id',qid) for qid,q in dataset['queries'].items()}
        if name=='RECIPER':clusters={qid:next(iter(dataset['qrels'][qid])) for qid in dataset['queries']}
        for method,queries in ranked.items():
            rows=[]
            for qid,order in queries.items():
                grades=dataset.get('grades',{}).get(qid)
                relevant=[pid for pid,g in grades.items() if g>=2] if grades is not None else dataset['qrels'][qid]
                values=ranking_metrics(order,relevant,grades)
                if values is None:continue
                row={'dataset':name,'split':split,'method':method,'query_id':qid,'cluster':clusters[qid],**values}
                rows.append(row);per_query.append(row)
            metrics={k:statistics.mean(r[k] for r in rows) for k in rows[0] if '@' in k}
            summaries.append({'dataset':name,'split':split,'method':method,'queries':len(rows),'collection':len(dataset['documents']),**metrics,
                'recall10_ci':bootstrap([r['Recall@10'] for r in rows],[r['cluster'] for r in rows])})
        (OUT/f'{name}-{split}-rankings.json').write_text(json.dumps(ranked),encoding='utf-8')
        (OUT/f'{name}-{split}-provenance.json').write_text(json.dumps({'dataset_version':dataset['version'],'manifest':dataset['manifest'],'protocol':dataset['protocol'],'encoder_revision':'1110a243fdf4706b3f48f1d95db1a4f5529b4d41','encoder_max_tokens':256,'bm25_stemmer':'simple','rrf_k':60,'branch_depth':'entire candidate pool','seconds':time.perf_counter()-start},indent=2),encoding='utf-8')
    report={'scope':'Fresh independent3-method CPU reruns of fixed owner protocols; no cross-encoder loaded because default app uses RRF.',
        'settings_frozen':True,'summaries':summaries,'comparability':'Same declared text/pools/encoder revision; current installed package versions/CPU can differ from owner MPS environment. No tuning or equality assumed.'}
    (OUT/'other_datasets.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    with (OUT/'retrieval_per_query.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in per_query for k in row)));writer.writeheader();writer.writerows(per_query)
    print('Completed',len(per_query),'query/method rows',flush=True)
if __name__=='__main__':main()
