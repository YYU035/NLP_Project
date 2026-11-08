"""Shared retrieval-results table with explicit method ownership/run provenance."""
import argparse
import csv
import json
from pathlib import Path


def build(comparison,protocol_file,output):
    output.mkdir(parents=True,exist_ok=False)
    summary=json.loads((comparison/'summary.json').read_text())
    protocol=json.loads(protocol_file.read_text())
    rows=[]
    for row in summary:
        person2=row['method'] in ('bm25','sbert')
        rows.append({'dataset':'SciFact','split':'dev','corpus_documents':5183,
            'method':row['method'],'rerank_depth':row['top_n'],'rrf_k':row['rrf_k'],
            'method_owner':'Naila (Person 2)' if person2 else 'Yu Yue (Person 3)',
            'experiment_runner':'Yu Yue (Person 3)','owner_confirmation':'pending',
            'method_source':'2bc7d81292313498fd70dd13656024d00cf3dda4' if person2 else 'Person 3 frozen local source snapshot; see protocol.json',
            'run_protocol_git_head':protocol['git_head'],
            'evaluated_claims':row['evaluated_claims'],'excluded_no_relevant':row['excluded_no_relevant'],
            'MRR@10':row['MRR@10'],'Recall@5':row['Recall@5'],'Recall@10':row['Recall@10'],
            'Recall@20':row['Recall@20'],'nDCG@10':row['nDCG@10'],
            'latency_scope':row['latency_stage'],'latency_mean_seconds':row['latency_mean_seconds'],
            'latency_p95_seconds':row['latency_p95_seconds'],
            'qualification':'one-run development result, not independent owner rerun or stable end-to-end latency'})
    with (output/'retrieval-results.csv').open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'retrieval-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    template=[{'owner':owner,'dataset':'','split':'','method':'','run_command':'','source_commit':'',
               'data_hashes':'','model_revision':'','Recall@10':'','MRR@10':'','nDCG@10':'',
               'latency_scope':'','results_artifact':'','confirmation':''}
              for owner in ['Raghavi (Person 1)','Naila (Person 2)','Yu Yue (Person 3)','Chong Yao (Person 4)','Kireeti (Person 5)']]
    with (output/'owner-submissions.csv').open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(template[0]));writer.writeheader();writer.writerows(template)
    (output/'README.md').write_text('''# Shared retrieval-results table

retrieval-results.csv contains measured results from Yu Yue's execution of the
pinned BM25/SBERT/RRF/reranker pipeline. Method ownership and experiment runner
are separate. These rows do not claim Naila independently ran or approved them.

Each owner should run their own experiments and add an artifact/command to
owner-submissions.csv, then confirm the matching results. Person 4/5 may link
verification/explanation results separately; do not put Accuracy/F1 into retrieval
Recall columns. Person 1 live API samples and SciFact fixed-corpus runs are
separate evaluation settings. Unknown values remain blank.

No messages were sent to teammates and no owner signatures were fabricated.
Coordination artifact is prepared; owner reruns/confirmations remain pending.
''')
    print(output/'retrieval-results.csv')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--comparison',type=Path,required=True);ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args();build(a.comparison,a.protocol,a.output_dir)
