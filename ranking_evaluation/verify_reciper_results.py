"""Independently recompute single-source metrics from saved complete rankings.

No model inference and no use of ranking_evaluation.metrics. This checks result
arithmetic/export integrity, not expert relevance or scientific evidence truth.
"""

import argparse
import json
from math import isclose, log2
from pathlib import Path

from shared.datasets.reciper import load
from shared.datasets.setup import file_hash

METHODS = {'bm25', 'sbert', 'rrf', 'rrf_cross_encoder'}


def verify(dataset, report):
    if report['version'] != dataset['version'] or set(report['rankings']) != METHODS:
        raise ValueError('Dataset version or four-method coverage mismatch')
    questions, corpus = set(dataset['queries']), set(dataset['documents'])
    ranks = {}
    for method, rankings in report['rankings'].items():
        if set(rankings) != questions:
            raise ValueError('Every question must be ranked')
        positions = {}
        for qid, ranking in rankings.items():
            if len(ranking) != len(set(ranking)) or set(ranking) != corpus:
                raise ValueError('Full corpus rankings must be unique and complete')
            targets = dataset['qrels'][qid]
            if len(targets) != 1:
                raise ValueError('Independent formula requires one source target')
            positions[qid] = ranking.index(targets[0]) + 1
        ranks[method] = positions
        observed = report['methods'][method]['binary_metrics']['macro']
        independently_computed = {'MRR': sum(1 / rank for rank in positions.values()) / len(questions)}
        for k in (1, 5, 10, 20):
            independently_computed.update({
                f'Recall@{k}': sum(rank <= k for rank in positions.values()) / len(questions),
                f'MRR@{k}': sum(1 / rank for rank in positions.values() if rank <= k) / len(questions),
                f'nDCG@{k}': sum(1 / log2(rank + 1) for rank in positions.values() if rank <= k) / len(questions)})
        if any(not isclose(value, observed[key], rel_tol=0, abs_tol=1e-12)
               for key, value in independently_computed.items()):
            raise ValueError('Saved metrics differ from independent single-source arithmetic')
    top_n = report['configuration']['rerank_top_n']
    for qid in questions:
        before, after = report['rankings']['rrf'][qid], report['rankings']['rrf_cross_encoder'][qid]
        if set(before[:top_n]) != set(after[:top_n]) or before[top_n:] != after[top_n:]:
            raise ValueError('Reranking changed candidates outside its declared prefix')
    before, after = ranks['rrf'], ranks['rrf_cross_encoder']
    effects = {
        'source_rank_improved': sum(after[qid] < before[qid] for qid in questions),
        'source_rank_worsened': sum(after[qid] > before[qid] for qid in questions),
        'source_rank_unchanged': sum(after[qid] == before[qid] for qid in questions),
        'source_top1_recovered': sum(before[qid] != 1 and after[qid] == 1 for qid in questions),
        'source_top1_lost': sum(before[qid] == 1 and after[qid] != 1 for qid in questions),
        'source_outside_rrf_prefix': sum(before[qid] > top_n for qid in questions),
        'maximum_possible_recall_in_reranked_prefix': sum(before[qid] <= top_n for qid in questions) / len(questions),
        'note': 'Known-source recovery only; other retrieved papers may also be relevant.'}
    return {'independent_arithmetic_verified': True, 'complete_rankings_verified': True,
        'rerank_prefix_integrity_verified': True, 'questions': len(questions),
        'corpus_size': len(corpus), 'methods': sorted(METHODS), 'source_rank_effects': effects,
        'limitation': 'Implementation/result arithmetic audit, not expert relevance/evidence validation or an independent model rerun.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/reciper'))
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new path')
    report = verify(load(args.data), json.loads(args.comparison.read_text()))
    report['provenance'] = {'comparison_sha256': file_hash(args.comparison),
                            'verifier_sha256': file_hash(Path(__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
