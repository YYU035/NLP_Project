"""Validate real frontend artifacts, preserving legitimate provider failures."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def validate(directory):
    result={}
    for mode in ('recorded','live'):
        report=json.loads((directory/f'{mode}-report.json').read_text())
        raw=directory/f'{mode}-run.json'
        bundle=json.loads(raw.read_text());snapshot=bundle['snapshot']
        if (report['candidate_count']<=0 or report['verified_papers']<=0 or report['passages']<=0
                or report['status'] not in ('ok','partial') or not report['displayable']):
            raise ValueError('No valid model-backed result for '+mode)
        if snapshot['stage_statuses']['passages'] not in ('person3_semantic_train_frozen','person3_semantic_bounded_train_frozen'):
            raise ValueError('Person 3 semantic selector is not enabled')
        if snapshot['stage_statuses']['passages']=='person3_semantic_bounded_train_frozen':
            budgets=json.loads(snapshot['filters']['verifier_budget_json'])
            if any(check['pair_tokens']>512 for value in budgets.values() for check in value['checks'] if check['accepted']):
                raise ValueError('Accepted evidence exceeded Person 4 input budget')
        source=snapshot['discovery_context']['result']
        papers={p['identity']:p for p in source['papers']}
        if source['original_claim']!=snapshot['claim']:
            raise ValueError('Original claim mismatch')
        for passage in snapshot['verification']['evidence_passages']:
            match=re.search(r':abstract:(\d+):(\d+):[0-9a-f]{12}$',passage['passage_id'])
            if not match or passage['paper_id'] not in papers:
                raise ValueError('Missing evidence lineage')
            start,end=map(int,match.groups())
            abstract=papers[passage['paper_id']]['paper']['abstract']
            if abstract[start:end]!=passage['text']:
                raise ValueError('Evidence quotation differs from source')
        result[mode]={'functional_validation':'passed','application_status':report['status'],
            'candidate_count':report['candidate_count'],'verified_papers':report['verified_papers'],
            'passages':report['passages'],'exact_source_spans':True,
            'provider_statuses':{name:{'status':p['status'],'error':p.get('error')} for name,p in source['providers'].items()},
            'artifact_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
            'scientific_quality':'not established by smoke checks'}
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifacts',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('Output exists')
    result=validate(args.artifacts)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(result,indent=2))
