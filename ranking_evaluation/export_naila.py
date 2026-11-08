"""Run a pinned local Person 2 snapshot and export all SciFact split claims.

The upstream scoring implementations are unchanged. The integration layer sorts
all scores before truncating, enforcing the upstream documented cutoff tie rule.
Gold evidence is read only for validation, never passed to retrieval scorers.
"""

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from time import perf_counter

from ranking_evaluation.scifact import (
    RankingExport, RankingExportRecord, load_scifact, sha256,
)


def stable_top_k(scores, paper_ids: list[str], k: int) -> list[str]:
    import numpy as np

    values = np.asarray(scores)
    if values.ndim != 1 or len(values) != len(paper_ids) or not np.isfinite(values).all():
        raise ValueError('Expected one finite retrieval score per corpus document')
    if type(k) is not int or k <= 0:
        raise ValueError('k must be a positive integer')
    order = np.lexsort((np.arange(len(values)), -values))[:k]
    return [paper_ids[int(index)] for index in order]


def validate_source(root: Path) -> dict:
    manifest = json.loads((root / 'source_manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Source manifest path escapes snapshot')
        if sha256(path) != expected:
            raise ValueError(f'Upstream source checksum mismatch: {name}')
    return manifest


def export_rankings(*, papers, claims, bm25, sbert, k, synchronize=lambda: None):
    """Score every claim independently, without using its relevance labels."""
    ids = [paper.paper_id for paper in papers]
    if bm25.paper_ids != ids or sbert.paper_ids != ids:
        raise ValueError('Both indexes must use the same corpus document order')
    records = []
    # One warm-up per branch, including query encoding. No gold-based selection.
    first = next(iter(claims.values())).claim
    bm25.score_all(first)
    sbert.score_all(first)
    synchronize()
    for claim_id, claim in claims.items():
        branches = []
        for scorer in (bm25, sbert):
            synchronize()
            start = perf_counter()
            ranking = stable_top_k(scorer.score_all(claim.claim), ids, k)
            synchronize()
            branches.append((ranking, perf_counter() - start))
        records.append(RankingExportRecord(
            claim_id=claim_id, bm25_ranking=branches[0][0], sbert_ranking=branches[1][0],
            bm25_seconds=branches[0][1], sbert_seconds=branches[1][1],
        ))
        if len(records) % 25 == 0:
            print(f'Ranked {len(records)}/{len(claims)} claims', flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retrieval-root', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--claims', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'dev'], required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--model-revision', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new path')
    if args.k <= 0 or args.batch_size <= 0:
        parser.error('k and batch size must be positive')
    model_manifest = json.loads((args.model_path / "download_manifest.json").read_text())
    if (model_manifest["model_id"] != args.model_id
            or model_manifest["revision"] != args.model_revision):
        raise ValueError("SBERT manifest does not match declared model/revision")
    for name, expected in model_manifest["files_sha256"].items():
        path = args.model_path / name
        if not path.resolve().is_relative_to(args.model_path.resolve()) or sha256(path) != expected:
            raise ValueError(f"SBERT model checksum mismatch: {name}")
    source = validate_source(args.retrieval_root)
    # Explicit source snapshot takes precedence over the empty repo placeholder.
    sys.path.insert(0, str(args.retrieval_root.resolve()))
    from retrieval.bm25 import BM25Retriever
    from retrieval.data import load_scifact_corpus
    from retrieval.sbert import SBERTRetriever
    from sentence_transformers import SentenceTransformer
    import torch
    import nltk  # Require real Porter stemming, never silently use the fallback.

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable')
    synchronize = (lambda: torch.cuda.synchronize()) if args.device == 'cuda' else lambda: None
    docs, claims = load_scifact(args.corpus, args.claims, split=args.split)
    papers = load_scifact_corpus(args.corpus)
    if [p.paper_id for p in papers] != list(docs):
        raise ValueError('Upstream corpus IDs/order disagree with benchmark adapter')
    synchronize()
    start = perf_counter()
    encoder = SentenceTransformer(str(args.model_path.resolve()), device=args.device,
                                  local_files_only=True, trust_remote_code=False)
    synchronize()
    load_seconds = perf_counter() - start
    bm25 = BM25Retriever()
    start = perf_counter()
    bm25.index(papers)
    bm25_index_seconds = perf_counter() - start
    sbert = SBERTRetriever(args.model_id, batch_size=args.batch_size, cache_dir=None,
                          device=args.device, encoder=encoder)
    synchronize()
    start = perf_counter()
    sbert.index(papers, use_cache=False)
    synchronize()
    sbert_index_seconds = perf_counter() - start
    records = export_rankings(papers=papers, claims=claims, bm25=bm25, sbert=sbert,
                              k=args.k, synchronize=synchronize)
    config = {
        'upstream_commit': source['commit'],
        'upstream_files_sha256': json.dumps(source['files_sha256'], sort_keys=True),
        'bm25': json.dumps(bm25.settings(), sort_keys=True),
        'sbert': json.dumps(sbert.settings(), sort_keys=True),
        'sbert_revision': args.model_revision,
        'sbert_files_sha256': json.dumps({p.relative_to(args.model_path).as_posix(): sha256(p)
            for p in sorted(args.model_path.rglob('*'))
            if p.is_file() and '.cache' not in p.relative_to(args.model_path).parts}, sort_keys=True),
        'device': args.device, 'branch_depth': str(args.k),
        'tie_policy': 'integration full sort: score descending, corpus index ascending; fixes upstream argpartition cutoff ties',
        'claim_policy': 'every split claim including no-evidence claims; no gold candidate filtering',
        'text_policy': 'upstream Paper.index_text title + abstract; original SciFact sentence list retained upstream',
        'timing_scope': 'warm per-query scoring+full sorting+ID conversion; CUDA synchronized; excludes model loading/index construction',
        'model_load_seconds': str(load_seconds), 'bm25_index_seconds': str(bm25_index_seconds),
        'sbert_index_seconds': str(sbert_index_seconds),
        'packages': json.dumps({name: importlib.metadata.version(name) for name in (
            'torch', 'sentence-transformers', 'transformers', 'numpy', 'nltk')}, sort_keys=True),
    }
    result = RankingExport(status='complete', corpus_sha256=sha256(args.corpus),
                           claims_sha256=sha256(args.claims), retrieval_config=config, records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(result.model_dump_json(indent=2) + '\n')
    print(f'Exported {len(records)} claims to {args.output}', flush=True)


if __name__ == '__main__':
    main()
