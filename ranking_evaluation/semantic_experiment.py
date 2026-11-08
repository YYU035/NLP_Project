"""Train-only selector configuration search, then frozen SciFact-dev diagnostics."""

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import sys
from time import perf_counter

from ranking_evaluation.evaluate_evidence import load_rankings, score_selection, summarize
from ranking_evaluation.export_naila import validate_source, stable_top_k
from ranking_evaluation.scifact import load_scifact, sha256
from ranking_evaluation.semantic_evidence import LocalSemanticSelector, SelectionConfig, select_scored_sentences


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runtime', type=Path, required=True)
    ap.add_argument('--comparison', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--retrieval-root', type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    corpus = args.runtime/'scifact/corpus.jsonl'
    train_file, dev_file = [args.runtime/f'scifact/claims_{s}.jsonl' for s in ['train','dev']]
    docs, train_all = load_scifact(corpus, train_file, split='train')
    _, dev = load_scifact(corpus, dev_file, split='dev')
    if set(train_all) & set(dev):
        raise ValueError('Train/dev claim IDs overlap')
    train_ids = sorted(train_all, key=lambda c: hashlib.sha256(c.encode()).hexdigest())[:200]
    train = {key:train_all[key] for key in train_ids}
    model_path = args.runtime/'sbert-model'
    model_manifest = json.loads((model_path/'download_manifest.json').read_text())
    for name, expected in model_manifest['files_sha256'].items():
        if sha256(model_path/name) != expected:
            raise ValueError('Model hash mismatch: '+name)
    upstream = validate_source(args.retrieval_root)
    config_grid = [SelectionConfig(k, threshold, radius) for k, threshold, radius
                   in itertools.product([1,2,3], [.2,.3,.4,.5], [0,1])]
    manifest = {'status':'running','training_claim_ids':train_ids,
        'training_sampling':'first 200 SHA-256 ordered claim IDs, independent of labels',
        'corpus_sha256':sha256(corpus),'train_sha256':sha256(train_file),'dev_sha256':sha256(dev_file),
        'grid':[asdict(c) for c in config_grid],
        'objective':'maximize train-sample retrieved-top5 micro sentence F1; ties prefer precision, less context, fewer passages, higher threshold',
        'protocol':'train-only configuration search; all dev results reported as development diagnostics, not untouched test',
        'model':model_manifest,'retrieval_source':upstream,
        'max_sequence_length':256,'model_training':'none; pretrained weights unchanged',
        'packages':{dist.metadata['Name']:dist.version for dist in importlib.metadata.distributions()},
        'source_sha256':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}}
    manifest_path = args.output_dir/'manifest.json'
    def save():manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        selector = LocalSemanticSelector(model_path)
        encoder = selector.encoder
        sentence_texts = list(dict.fromkeys(sentence for doc in docs.values() for sentence in doc.abstract))
        lookup = {s:i for i,s in enumerate(sentence_texts)}
        cache_key = hashlib.sha256(json.dumps({'corpus':sha256(corpus),'model':model_manifest,
            'length':256,'sentence_transformers':importlib.metadata.version('sentence-transformers')},sort_keys=True).encode()).hexdigest()[:20]
        cache = args.runtime/'sentence-cache'/cache_key
        cache.mkdir(parents=True,exist_ok=True)
        array_path = cache/'vectors.npy'
        if array_path.exists():
            vectors = np.load(array_path)
        else:
            vectors = encoder.encode(sentence_texts,batch_size=64,normalize_embeddings=True,
                                     convert_to_numpy=True,show_progress_bar=True)
            np.save(array_path,vectors)
        if vectors.shape[0]!=len(sentence_texts) or not np.isfinite(vectors).all():
            raise ValueError('Invalid cached sentence embeddings')
        manifest['sentence_vectors_sha256']=sha256(array_path)
        queries = {**train,**dev}
        query_ids = list(queries)
        query_vectors = encoder.encode([queries[c].claim for c in query_ids],batch_size=64,
            normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
        qvec = dict(zip(query_ids,query_vectors))
        doc_vectors = {d:vectors[[lookup[s] for s in doc.abstract]] for d,doc in docs.items()}
        score_cache = {}
        def spans(cid, doc, config):
            key=(cid,doc)
            if key not in score_cache:score_cache[key]=doc_vectors[doc] @ qvec[cid]
            return select_scored_sentences(doc,docs[doc].abstract,score_cache[key],config=config)
        sys.path.insert(0,str(args.retrieval_root.resolve()))
        from retrieval.data import load_scifact_corpus
        from retrieval.retriever import build_all, rrf_fuse
        papers = load_scifact_corpus(corpus)
        retrievers = build_all(papers,encoder=encoder,sbert_model=model_manifest['model_id'],cache_dir=None)
        paper_ids = [p.paper_id for p in papers]
        train_ranks = {}
        for cid,claim in train.items():
            bm25 = stable_top_k(retrievers['bm25'].bm25.score_all(claim.claim),paper_ids,100)
            sbert = stable_top_k(retrievers['sbert'].sbert.score_all(claim.claim),paper_ids,100)
            train_ranks[cid]=[d for d,_ in rrf_fuse({'bm25':bm25,'sbert':sbert})][:5]
        (args.output_dir/'train-rankings.json').write_text(json.dumps(train_ranks)+'\n')
        def measure(claims, ranks, config, *, keep=False):
            records=[]
            for cid,claim in claims.items():
                selected = list(claim.evidence) if ranks is None else ranks[cid]
                predicted=set();passages=[]
                for doc in selected:
                    for span in spans(cid,doc,config):
                        predicted.update((doc,i) for i in span.sentence_indices)
                        if keep:passages.append(asdict(span))
                row={'claim_id':cid,**score_selection(predicted,claim.evidence)}
                if keep:row.update(passages=passages,predicted_sentence_ids=sorted(predicted),selected_papers=selected)
                records.append(row)
            return summarize(records),records
        tuning=[]
        for config in config_grid:
            result,_=measure(train,train_ranks,config)
            tuning.append({'configuration':asdict(config),'metrics':result})
        def objective(row):
            c=row['configuration'];m=row['metrics']['micro']
            return (m['f1'],m['precision'],-c['context_radius'],-c['top_k'],c['min_score'])
        best=max(tuning,key=objective)
        frozen=SelectionConfig(**best['configuration'])
        (args.output_dir/'train-search.json').write_text(json.dumps(tuning,indent=2)+'\n')
        (args.output_dir/'frozen-config.json').write_text(json.dumps({'selection_config':asdict(frozen),
            'model_id':model_manifest['model_id'],'model_revision':model_manifest['revision'],
            'max_sequence_length':256,'tuning_split':'train','tuning_claims':len(train)},indent=2)+'\n')
        manifest.update(frozen_configuration=asdict(frozen),training_best_micro=best['metrics']['micro'])
        save()  # Configuration is frozen before any dev evidence labels are scored.
        print('Frozen train-selected configuration:',asdict(frozen),flush=True)
        version=f'corpus:{sha256(corpus)};claims:{sha256(dev_file)}'
        rankings,hashes=load_rankings([args.comparison/'repeat-1-rrf-60.json',
            args.comparison/'repeat-1-rrf-60-rerank-20.json'],dataset_version=version,claims=dev,docs=docs)
        manifest['dev_ranking_report_sha256']=hashes
        summaries=[]
        with (args.output_dir/'dev-per-claim.jsonl').open('x') as stream:
            for policy,radius in [('semantic_sentence',0),('semantic_context',1)]:
                config=replace(frozen,context_radius=radius)
                jobs=[('oracle_gold_papers',None,None)]
                jobs += [(method,k,{c:ids[:k] for c,ids in ranks.items()})
                         for method,ranks in rankings.items() for k in [5,10]]
                for method,k,ranks in jobs:
                    start=perf_counter()
                    result,records=measure(dev,ranks,config,keep=True)
                    summaries.append({'policy':policy,'ranking_method':method,'paper_depth':k,
                        'configuration':asdict(config),'selection_seconds':perf_counter()-start,**result})
                    for record in records:
                        stream.write(json.dumps({'policy':policy,'ranking_method':method,'paper_depth':k,**record},ensure_ascii=False)+'\n')
                    print('Dev:',policy,method,k,result['micro'],flush=True)
        (args.output_dir/'dev-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
        manifest['status']='complete'
    except Exception as exc:
        manifest.update(status='failed',error=str(exc));raise
    finally:save()


if __name__=='__main__':main()
