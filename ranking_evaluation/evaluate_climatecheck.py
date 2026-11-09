"""Supplemental gold abstract-pool ranking metrics for ClimateCheck."""

from ranking_evaluation.metrics import evaluate_rankings


def evaluate(dataset, export):
    if dataset['dataset'] != 'ClimateCheck':
        raise ValueError('Expected ClimateCheck dataset')
    if export.get('status') != 'complete' or export.get('dataset_version') != dataset['version']:
        raise ValueError('Completed rankings must match the dataset protocol/version')
    if not isinstance(export.get('method'), str) or not export['method'].strip():
        raise ValueError('Method provenance is required')
    rankings = export['rankings']
    if set(rankings) != set(dataset['queries']):
        raise ValueError('Every selected exact claim must be scored')
    for qid, ranking in rankings.items():
        pool = set(dataset['queries'][qid]['candidate_ids'])
        if not isinstance(ranking, list) or len(ranking) != len(set(ranking)):
            raise ValueError('Expected unique ranking IDs')
        if set(ranking) != pool:
            raise ValueError('Every judged-pool paper must be ranked exactly once')
    qrels = {qid: [pid for pid, grade in grades.items() if grade == 2]
             for qid, grades in dataset['grades'].items()}
    metrics = evaluate_rankings(rankings, qrels, ks=(1, 2, 5, 10))
    return {'dataset': 'ClimateCheck', 'version': dataset['version'],
        'method': export['method'], 'protocol': dataset['protocol'],
        'query_count': len(rankings), 'binary_metrics': metrics, 'graded_macro': {},
        'metric_note': 'Supplemental closed human-judged pool metrics; not official full-corpus Recall/Bpref or verdict F1. Supports and Refutes both count as evidentiary positives; no sentence gold.',
        'pool_sizes': {qid: len(row['candidate_ids']) for qid, row in dataset['queries'].items()}}
