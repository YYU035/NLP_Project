"""Freeze the pre-study state before examining new complete-pipeline results."""
import hashlib,json,platform,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/validation'
OWNER='90aaca109585c12c877367b1a08fe4dfee4a4de3'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'BASELINE.json'
    if target.exists():
        baseline=json.loads(target.read_text(encoding='utf-8'))
        for name,digest in baseline['source_sha256'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Frozen production source changed: '+name)
        for name,digest in baseline['input_sha256'].items():
            if sha(ROOT/'outputs/xai/data'/name)!=digest:raise ValueError('Frozen study input changed: '+name)
        return baseline
    from ...pipeline_setup import NLI_ID,NLI_REVISION,NLI_PATH
    paths=[ROOT/'pipeline.py',ROOT/'semantic_integration.py',ROOT/'person4.py',ROOT/'_upstream/person3/semantic-train-frozen.json',ROOT/'_upstream/retrieval/retriever.py',ROOT/'_upstream/retrieval/bm25.py',ROOT/'_upstream/claim_verifier/verifier.py']
    models={}
    for name in ['claim-verifier','all-MiniLM-L6-v2','person3-semantic']:
        base=ROOT/'.models'/name
        files=[p for p in base.rglob('*') if p.is_file() and (p.suffix=='.safetensors' or p.name in ['pytorch_model.bin','config.json','tokenizer.json','pinned_revision.txt','download_manifest.json']) and '.cache' not in p.parts]
        models[name]={str(p.relative_to(base)):sha(p) for p in files}
    data={name:sha(ROOT/'outputs/xai/data'/name) for name in ['corpus.jsonl','claims_dev.jsonl']}
    config={'date':'2026-11-10','created_at':datetime.now(timezone.utc).isoformat(),
       'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'python':platform.python_version(),'owner_source':OWNER,'input_sha256':data,'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},
       'model_files_sha256':models,'nli':{'id':NLI_ID,'revision':NLI_REVISION,'threshold':.5,'pair_limit':512,'dtype':'float32'},
       'science_protocol':{'scifact_claims':300,'candidate_universe':'all5183 corpus abstracts; never inject gold papers',
          'methods':['bm25','sbert','rrf'],'fusion_branch_depth':100,'rrf_k':60,'verify_paper_depth':5,
          'selector':'production exact-string split and frozen semantic policy',
          'controls':['oracle_gold_rationale','gold_papers_production_selector','retrieved_papers_production_selector'],
          'confidence_intervals':'seeded percentile bootstrap,2000 repeats; claim units for SciFact; related claims can remain dependent',
          'no_positive_policy':'112 no-positive claims included as closed-benchmark INSUFFICIENT; evidence F1 denominator excludes these and reports their selections separately'},
       'splits':{'SciFact':'dev already inspected; locked descriptive evaluation, not untouched final test',
          'CSFCube':'official fold1_dev24 and fold1_test26 tasks, seven query-paper IDs overlap; no tuning here',
          'RECIPER':'entire1024-question release; frozen zero-shot source-paper ranking, not a held-out split'},
       'selection_policy':'Freeze settings and report every case. No threshold/model selection from these results.',
       'protocol_caveat':'Offline corpus benchmark replaces web discovery; functional live-provider checks are separate. Top100 branches/top5 papers define an automated evaluation wrapper, not a claim that the UI automatically selects papers.',
       'documentation_finding':'Pinned BM25 is a custom NumPy implementation, not a call to rank-bm25; correct dependency descriptions separately.'}
    target.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    (ROOT/'evaluation/validation/PROTOCOL.json').write_text(json.dumps({k:v for k,v in config.items() if k not in ['model_files_sha256']},indent=2)+'\n',encoding='utf-8')
    print('Frozen baseline',config['git_commit'],flush=True)
    return config
if __name__=='__main__':freeze()
