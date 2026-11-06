"""Explicit setup and finite-score QC for Yu Yue's optional local scorer."""
import hashlib
import importlib.metadata
import json
import math
import subprocess
from time import perf_counter
from pathlib import Path
from urllib.request import urlopen

from ..ranking import runtime

ROOT = Path(__file__).resolve().parents[1]
MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
CHECKSUM = "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"


def main():
    target = ROOT / ".models" / "qc-msmarco"
    target.mkdir(parents=True, exist_ok=True)
    for name in ["README.md", "config.json", "tokenizer.json", "tokenizer_config.json",
                 "vocab.txt", "special_tokens_map.json", "model.safetensors"]:
        path = target / name
        if not path.is_file():
            with urlopen(f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}?download=true", timeout=60) as response:
                temporary = path.with_suffix(path.suffix + '.part')
                with temporary.open('wb') as handle:
                    while chunk := response.read(4 * 1024 * 1024):
                        handle.write(chunk)
                temporary.replace(path)
    with (target / 'model.safetensors').open('rb') as handle:
        if hashlib.file_digest(handle, 'sha256').hexdigest() != CHECKSUM:
            raise ValueError('Cross-encoder checksum mismatch')
    card = (target / 'README.md').read_text(encoding='utf-8')
    if 'license: apache-2.0' not in card.lower():
        raise ValueError('Expected open-source Apache-2.0 license declaration')
    Input, _, rank_claim = runtime()
    from ranking_evaluation.reranker import LocalCrossEncoder
    import torch
    torch.set_num_threads(6)
    pairs = [('What is the capital of France?', 'Paris is the capital of France.'),
             ('What is the capital of France?', 'Bananas are a fruit.'),
             ('Physical exercise improves memory.', 'Aerobic exercise improved memory scores in adults.')]
    rows = []
    chosen = None
    for use64 in [False, True]:
        try:
            scorer = LocalCrossEncoder(target, cpu_float64=use64, batch_size=3)
            scores = [float(x) for x in scorer.score_pairs(pairs)]
            if not use64:
                chosen = scorer
            rows.append({'dtype': 'float64' if use64 else 'default', 'finite': all(map(math.isfinite, scores)),
                         'parameter_dtype': str(next(scorer.model.model.parameters()).dtype),
                         'scores': [x if math.isfinite(x) else str(x) for x in scores]})
        except Exception as error:
            rows.append({'dtype': 'float64' if use64 else 'default', 'finite': False,
                         'error_type': type(error).__name__})
    result = {'model': MODEL, 'revision': REVISION, 'sha256': CHECKSUM, 'license': 'Apache-2.0',
              'packages': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'sentence-transformers']},
              'pair_count': len(pairs), 'results': rows,
              'note': 'Numerical smoke only; scientific ranking quality not measured; production RRF-only unchanged.'}
    development = ROOT / 'outputs' / 'development_qc' / 'retrieval_records.json'
    if chosen is not None and rows[0]['finite'] and development.exists():
        from ranking_evaluation.reranker import rerank
        def load(name):
            raw = subprocess.check_output(['git', 'show',
                f'1ac06b675e3d0719a63cb8efe98eda4f2c5fd576:claim_verifier/data/{name}'], cwd=ROOT.parent)
            return [json.loads(line) for line in raw.decode().splitlines()]
        corpus = {str(p['doc_id']): p for p in load('corpus.jsonl')}
        claims = {str(p['id']): p for p in load('claims_dev.jsonl')}
        methods = json.loads(development.read_text(encoding='utf-8'))
        benchmarks = []
        for cid in sorted({r['claim_id'] for r in methods}):
            lists = {r['method']: r['ranked_ids'] for r in methods if r['claim_id'] == cid}
            ids = sorted(set(lists['bm25']) | set(lists['sbert']))
            entries = [{'identity': pid, 'identity_aliases': [pid], 'original_claim': claims[cid]['claim'],
                'paper': {'title': corpus[pid]['title'], 'abstract': ' '.join(corpus[pid]['abstract']),
                          'source': 'scifact', 'source_id': pid}, 'provenance': [], 'source_records': []} for pid in ids]
            inputs = Input.model_validate({'claim_id': cid, 'bm25_ranking': lists['bm25'], 'sbert_ranking': lists['sbert'],
                'candidates': {'original_claim': claims[cid]['claim'], 'top_k': 100, 'papers': entries}})
            fused = rank_claim(inputs).papers
            started = perf_counter()
            ranked = rerank(fused, claim=claims[cid]['claim'], scorer=chosen, top_n=20)
            elapsed = perf_counter() - started
            actual = [r.entry.identity for r in ranked][:100]
            gold = {pid for pid, evidence in claims[cid]['evidence'].items() if evidence}
            dcg = sum(1 / math.log2(i + 2) for i, pid in enumerate(actual[:10]) if pid in gold)
            idcg = sum(1 / math.log2(i + 2) for i in range(min(10, len(gold))))
            benchmarks.append({'mrr_at_100': next((1 / (i+1) for i, pid in enumerate(actual) if pid in gold), 0),
                'recall_at_10': len(set(actual[:10]) & gold) / len(gold), 'ndcg_at_10': dcg / idcg,
                'rerank_seconds': elapsed})
            print(f'Rerank development claim {len(benchmarks)}/22', flush=True)
        result['development_top20_rerank'] = {'claims': len(benchmarks),
            **{key: sum(row[key] for row in benchmarks) / len(benchmarks) for key in benchmarks[0]},
            'note': 'Same frozen small dev sample; not untouched test performance; no tuning or production enablement.'}
        result['note'] = 'Finite-score and frozen development checks completed; production remains RRF-only pending team review.'
    output = ROOT / 'outputs' / 'team_qc'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'reranker-summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
