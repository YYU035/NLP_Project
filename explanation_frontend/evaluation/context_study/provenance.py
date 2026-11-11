"""Capture experiment provenance and audit train/dev identities without altering choices."""
import argparse,json,hashlib,importlib.metadata
from pathlib import Path
from .freeze_train import ROOT,OUT
from ..validation.scifact import gold_label

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['start','audit']);args=ap.parse_args()
    if args.mode=='start':
        target=OUT/'RUN_PROTOCOL.json'
        if target.exists():print('Retaining run provenance');return
        baseline=json.loads((ROOT/'outputs/validation/BASELINE.json').read_text(encoding='utf-8'))
        record={'date':'2026-11-11','captured':'before evaluation stages','source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
          'baseline_sha256':hashlib.sha256((ROOT/'outputs/validation/BASELINE.json').read_bytes()).hexdigest(),'model_files_sha256':baseline['model_files_sha256'],
          'training_choices_sha256':hashlib.sha256((OUT/'TRAIN_FROZEN.json').read_bytes()).hexdigest(),
          'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','sentence-transformers','numpy','scikit-learn']},
          'execution':'Pinned local CPU NLI float32, six Torch threads; no latency or throughput inference.',
          'scope':'Original defaults and baseline retained. Already-inspected dev; not an untouched test.'}
        target.write_text(json.dumps(record,indent=2),encoding='utf-8')
    else:
        train={str(c['id']):c for c in map(json.loads,(OUT/'claims_train.jsonl').read_text(encoding='utf-8').splitlines())}
        dev=list(map(json.loads,(ROOT/'outputs/xai/data/claims_dev.jsonl').read_text(encoding='utf-8').splitlines()))
        owner=[json.loads(s) for s in (OUT/'owner_train_cases.jsonl').read_text(encoding='utf-8').splitlines()];focused=[x for x in owner if x['policy']=='baseline']
        assert len(focused)==200 and all(x['claim']==train[x['claim_id']]['claim'] and x['gold']==gold_label(train[x['claim_id']]) for x in focused)
        ids={x['claim_id'] for x in focused};dev_ids={str(c['id']) for c in dev}
        record={'train_ids':len(ids),'dev_ids':len(dev_ids),'id_overlap':sorted(ids&dev_ids),'exact_claim_text_overlap':sorted({train[c]['claim'] for c in ids}&{c['claim'] for c in dev}),
          'focused_cached_claim_text_agreement':len(focused),'focused_cached_gold_agreement':len(focused),'training_source_sha256':hashlib.sha256((OUT/'claims_train.jsonl').read_bytes()).hexdigest(),
          'scope':'Disjoint IDs and exact-text checks do not exclude related claims or shared papers.'}
        assert not record['id_overlap'];(OUT/'SPLIT_AUDIT.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(json.dumps(record),flush=True)
if __name__=='__main__':main()
