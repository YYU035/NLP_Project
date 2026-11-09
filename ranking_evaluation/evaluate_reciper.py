"""Evaluate full-collection RECIPER source-paper recovery, not sentence evidence."""

from ranking_evaluation.metrics import evaluate_rankings


def evaluate(dataset, export):
    if dataset['dataset'] != 'RECIPER':
        raise ValueError('Expected RECIPER dataset')
    if export.get('status') != 'complete' or export.get('dataset_version') != dataset['version']:
        raise ValueError('Completed rankings must match the fixed dataset/protocol')
    if not isinstance(export.get('method'), str) or not export['method'].strip():
        raise ValueError('Method provenance is required')
    rankings = export['rankings']
    if set(rankings) != set(dataset['queries']):
        raise ValueError('Rankings must cover every question exactly once')
    corpus = set(dataset['documents'])
    for ranking in rankings.values():
        if not isinstance(ranking, list) or len(ranking) != len(set(ranking)) or set(ranking) != corpus:
            raise ValueError('Every paper in the full corpus must be ranked exactly once')
    return {'dataset': 'RECIPER', 'version': dataset['version'], 'method': export['method'],
        'protocol': dataset['protocol'], 'query_count': len(rankings), 'corpus_size': len(corpus),
        'binary_metrics': evaluate_rankings(rankings, dataset['qrels'], ks=(1, 5, 10, 20)),
        'graded_macro': {},
        'metric_note': 'Single-source paper recovery only. Other papers may also be relevant; no sentence-evidence or verification gold. Adapted title/abstract retrieval, not official paragraph/recipe retrieval parity.'}
