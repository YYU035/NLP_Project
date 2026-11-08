"""Compact RRF-constant ablations from one immutable branch export."""
import argparse
import json
from pathlib import Path

from ranking_evaluation.metrics import evaluate_rankings
from ranking_evaluation.rrf import fuse_ordered_ids
from ranking_evaluation.scifact import RankingExport, load_scifact, sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rankings',type=Path,required=True)
    ap.add_argument('--corpus',type=Path,required=True)
    ap.add_argument('--claims',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    export=RankingExport.model_validate_json(args.rankings.read_bytes())
    docs,claims=load_scifact(args.corpus,args.claims,split='dev')
    if export.status!='complete' or export.corpus_sha256!=sha256(args.corpus) or export.claims_sha256!=sha256(args.claims):
        raise ValueError('Incomplete export or dataset hash mismatch')
    records={r.claim_id:r for r in export.records}
    if len(records)!=len(export.records) or set(records)!=set(claims):raise ValueError('Claim coverage mismatch')
    for record in export.records:
        if set(record.bm25_ranking+record.sbert_ranking)-docs.keys():raise ValueError('Unknown corpus identity')
    qrels={c:list(row.evidence) for c,row in claims.items()}
    args.output_dir.mkdir(parents=True,exist_ok=False)
    rows=[]
    for constant in [20,60,100]:
        ranks={c:[doc for doc,_ in fuse_ordered_ids(r.bm25_ranking,r.sbert_ranking,rrf_k=constant)] for c,r in records.items()}
        metrics=evaluate_rankings(ranks,qrels)
        (args.output_dir/f'rrf-{constant}.json').write_text(json.dumps({'rrf_k':constant,'rankings':ranks,'metrics':metrics},indent=2)+'\n')
        rows.append({'rrf_k':constant,**metrics['macro']})
    (args.output_dir/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    (args.output_dir/'manifest.json').write_text(json.dumps({'status':'complete','protocol':'fixed-input development ablation; no best-constant selection',
        'corpus_sha256':sha256(args.corpus),'claims_sha256':sha256(args.claims),'rankings_sha256':sha256(args.rankings),
        'source_sha256':{p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('rrf.py')]},'constants':[20,60,100]},indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
