"""Audit measured benchmarks, replay ranking ablations and summarize uncertainty.

No model inference or hyperparameter selection. Replay uses exported rankings;
bootstrap intervals describe sampled query clusters, not label correctness.
"""

import argparse
import csv
import importlib.metadata
import json
from math import ceil, isclose, isfinite
from pathlib import Path

from ranking_evaluation.evaluate_csfcube import evaluate as evaluate_cs
from ranking_evaluation.evaluate_reciper import evaluate as evaluate_rec
from ranking_evaluation.rrf import fuse_ordered_ids
from shared.datasets.csfcube import load as load_cs
from shared.datasets.reciper import load as load_rec
from shared.datasets.setup import file_hash

METHODS = ('bm25', 'sbert', 'rrf', 'rrf_cross_encoder')
METRICS = ('Recall@10', 'MRR@10', 'nDCG@10')


def cluster_intervals(scores, groups, *, repeats=2000, seed=20261109):
    """Resample whole clusters; concatenate all their tasks, including repeats."""
    import numpy as np
    values = np.asarray(scores, dtype=float)
    if (values.ndim != 2 or values.shape[1] == 0 or len(values) != len(groups) or not len(groups)
            or not np.isfinite(values).all() or repeats < 100):
        raise ValueError('Expected finite task/metric matrix, matching groups and >=100 repeats')
    names = sorted(set(groups))
    sums = np.asarray([values[[g == name for g in groups]].sum(axis=0) for name in names])
    sizes = np.asarray([groups.count(name) for name in names])
    weights = np.random.default_rng(seed).multinomial(len(names), [1 / len(names)] * len(names), size=repeats)
    bootstrap = (weights @ sums) / (weights @ sizes)[:, None]
    return values.mean(axis=0), bootstrap, len(names)


def replay_prefix(before, after, original_depth, depth):
    """Counterfactual smaller prefix using its order in the cached larger prefix."""
    if (not 0 <= depth <= original_depth or set(before[:original_depth]) != set(after[:original_depth])
            or before[original_depth:] != after[original_depth:]):
        raise ValueError('Invalid cached rerank prefix or requested depth')
    keep = set(before[:depth])
    return [pid for pid in after[:original_depth] if pid in keep] + before[depth:]


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def audit_task(dataset, report, *, split, evaluator, repeats, seed):
    if set(report['rankings']) != set(METHODS) or report['version'] != dataset['version']:
        raise ValueError('Expected four complete methods for the exact dataset protocol')
    evaluated = {}
    primary = []
    for method in METHODS:
        rankings = report['rankings'][method]
        for qid, row in dataset['queries'].items():
            if len(rankings[qid]) != len(set(rankings[qid])) or set(rankings[qid]) != set(row['candidate_ids']):
                raise ValueError('Every candidate must be ranked once in every method')
        fresh = evaluator(dataset, {'status': 'complete', 'dataset_version': dataset['version'],
                                    'method': method, 'rankings': rankings})
        saved = report['methods'][method]
        for key, value in fresh['binary_metrics']['macro'].items():
            other = saved['binary_metrics']['macro'][key]
            if value is None and other is None:
                continue
            if value is None or other is None or not isclose(value, other, abs_tol=1e-12, rel_tol=0):
                raise ValueError('Saved metrics failed recomputation')
        for key, value in fresh.get('graded_macro', {}).items():
            other = saved.get('graded_macro', {}).get(key)
            if value is None and other is None:
                continue
            if value is None or other is None or not isclose(value, other, abs_tol=1e-12, rel_tol=0):
                raise ValueError('Saved graded metrics failed recomputation')
        evaluated[method] = fresh
        metrics = fresh['binary_metrics']
        primary.append({'dataset': dataset['dataset'], 'split': split, 'method': method,
            'rrf_k': report['configuration']['rrf_k'] if method.startswith('rrf') else None,
            'rerank_depth': report['configuration']['rerank_top_n'] if method == 'rrf_cross_encoder' else None,
            'total_queries': len(dataset['queries']), 'evaluated_queries': metrics['evaluated_claims'],
            'excluded_no_positive': metrics['excluded_no_relevant'], **metrics['macro'], **fresh.get('graded_macro', {})})
    base_k, base_n = report['configuration']['rrf_k'], report['configuration']['rerank_top_n']
    if base_k != 60 or base_n != 20:
        raise ValueError('This declared analysis expects frozen RRF=60, cached CE prefix=20')
    for qid, query in dataset['queries'].items():
        if report['per_query'][qid]['rerank_scored_count'] != min(base_n, len(query['candidate_ids'])):
            raise ValueError('Prefix replay requires every cached prefix candidate to have been scored')
    ablations = []
    for constant in (20, 60, 100):
        rankings = {qid: [pid for pid, _ in fuse_ordered_ids(report['rankings']['bm25'][qid],
                    report['rankings']['sbert'][qid], rrf_k=constant)] for qid in dataset['queries']}
        if constant == 60 and rankings != report['rankings']['rrf']:
            raise ValueError('RRF replay differs from the measured runtime ordering')
        scored = evaluator(dataset, {'status': 'complete', 'dataset_version': dataset['version'],
                           'method': f'rrf_k_{constant}_replay', 'rankings': rankings})
        ablations.append({'dataset': dataset['dataset'], 'split': split, 'family': 'rrf_constant',
            'rrf_k': constant, 'rerank_depth': 0, 'protocol': 'exact rank-only RRF replay; no new inference',
            **scored['binary_metrics']['macro']})
    for depth in (0, 5, 10, 20):
        rankings = {qid: replay_prefix(report['rankings']['rrf'][qid], report['rankings']['rrf_cross_encoder'][qid],
                                      base_n, depth) for qid in dataset['queries']}
        scored = evaluator(dataset, {'status': 'complete', 'dataset_version': dataset['version'],
                           'method': f'cached_ce_prefix_{depth}', 'rankings': rankings})
        ablations.append({'dataset': dataset['dataset'], 'split': split, 'family': 'rerank_prefix',
            'rrf_k': 60, 'rerank_depth': depth,
            'protocol': 'cached CE20 order restriction for depths 5/10; actual baseline/depth20 exports for 0/20; no new latency claim',
            **scored['binary_metrics']['macro']})
    eligible = [qid for qid in dataset['queries'] if evaluated['bm25']['binary_metrics']['per_claim'][qid]['MRR'] is not None]
    groups = [dataset['queries'][qid]['paper_id'] if dataset['dataset'] == 'CSFCube' else dataset['qrels'][qid][0]
              for qid in eligible]
    keys = [(method, metric) for method in METHODS for metric in METRICS]
    scores = [[evaluated[method]['binary_metrics']['per_claim'][qid][metric] for method, metric in keys] for qid in eligible]
    point, draws, clusters = cluster_intervals(scores, groups, repeats=repeats, seed=seed)
    import numpy as np
    intervals = []
    for index, (method, metric) in enumerate(keys):
        low, high = np.quantile(draws[:, index], [0.025, 0.975])
        intervals.append({'dataset': dataset['dataset'], 'split': split, 'comparison': method,
            'metric': metric, 'estimate': float(point[index]), 'ci_low': float(low), 'ci_high': float(high),
            'eligible_queries': len(eligible), 'clusters': clusters, 'bootstrap_repeats': repeats,
            'seed': seed, 'resampling_unit': 'query-paper ID' if dataset['dataset'] == 'CSFCube' else 'source-paper ID'})
    for left, right in [('rrf', 'bm25'), ('rrf', 'sbert'), ('rrf_cross_encoder', 'rrf')]:
        for metric in METRICS:
            a, b = keys.index((left, metric)), keys.index((right, metric))
            low, high = np.quantile(draws[:, a] - draws[:, b], [0.025, 0.975])
            intervals.append({'dataset': dataset['dataset'], 'split': split, 'comparison': f'{left} minus {right}',
                'metric': metric, 'estimate': float(point[a] - point[b]), 'ci_low': float(low), 'ci_high': float(high),
                'eligible_queries': len(eligible), 'clusters': clusters, 'bootstrap_repeats': repeats,
                'seed': seed, 'resampling_unit': 'query-paper ID' if dataset['dataset'] == 'CSFCube' else 'source-paper ID'})
    latency = []
    for stage in ('bm25_index_seconds', 'bm25_seconds', 'sbert_seconds', 'rrf_seconds', 'rerank_seconds'):
        values = []
        for row in report['per_query'].values():
            value = row.get(stage) if stage in ('bm25_index_seconds', 'rerank_seconds') else row['timings'].get(stage)
            if value is not None:
                if not isfinite(value) or value < 0:
                    raise ValueError('Invalid stage latency')
                values.append(value)
        values.sort()
        if values:
            latency.append({'dataset': dataset['dataset'], 'split': split, 'stage': stage,
                'observations': len(values), 'mean_seconds': sum(values) / len(values),
                'p95_seconds': values[ceil(0.95 * len(values)) - 1],
                'scope': 'one shared-workstation run; per-stage observations, not full-app throughput',
                'device': report.get('environment', {}).get('device') or report['configuration']['reranker']['device']})
    return primary, ablations, intervals, latency, {'dataset': dataset['dataset'], 'split': split,
        'saved_metrics_recomputed': True, 'complete_candidate_rankings': True,
        'rrf60_replay_matches_runtime': True, 'cached_rerank_prefix_integrity': True,
        'eligible_queries': len(eligible), 'clusters': clusters}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cs-dev', type=Path, default=Path('outputs/datasets/csfcube-fold1-dev-four-methods/comparison.json'))
    parser.add_argument('--cs-test', type=Path, default=Path('outputs/datasets/csfcube-fold1-test-four-methods/comparison.json'))
    parser.add_argument('--reciper', type=Path, default=Path('outputs/datasets/reciper-full-343-papers-1024-questions-mps-four-methods/comparison.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap-repeats', type=int, default=2000)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output exists; choose a new directory')
    primary, ablations, intervals, latency, checks, sources = [], [], [], [], [], {}
    tasks = [(load_cs(Path('data/csfcube'), split='fold1_dev'), args.cs_dev, 'fold1_dev', evaluate_cs),
             (load_cs(Path('data/csfcube'), split='fold1_test'), args.cs_test, 'fold1_test', evaluate_cs),
             (load_rec(Path('data/reciper')), args.reciper, 'entire_release_zero_shot', evaluate_rec)]
    for index, (dataset, path, split, evaluator) in enumerate(tasks):
        result = audit_task(dataset, json.loads(path.read_text()), split=split, evaluator=evaluator,
                            repeats=args.bootstrap_repeats, seed=20261109 + index)
        for destination, values in zip([primary, ablations, intervals, latency], result[:4]):
            destination.extend(values)
        checks.append(result[4]); sources[str(path)] = file_hash(path)
    sci_path = Path('ranking_evaluation/results/person3-v2/retrieval-results.csv')
    evidence_path = Path('ranking_evaluation/results/person3-v2/semantic-dev-summary.json')
    with sci_path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if int(row['evaluated_claims']) + int(row['excluded_no_relevant']) != 300:
                raise ValueError('SciFact ranking denominator mismatch')
            for metric in ('Recall@5', 'Recall@10', 'Recall@20', 'MRR@10', 'nDCG@10'):
                if not 0 <= float(row[metric]) <= 1:
                    raise ValueError('SciFact summary metric out of range')
            primary.append({'dataset': 'SciFact', 'split': 'dev', 'method': row['method'],
                'rrf_k': row['rrf_k'], 'rerank_depth': row['rerank_depth'], 'total_queries': 300,
                'evaluated_queries': int(row['evaluated_claims']),
                'excluded_no_positive': int(row['excluded_no_relevant']),
                **{metric: float(row[metric]) for metric in ('Recall@5','Recall@10','Recall@20','MRR@10','nDCG@10')}})
            latency.append({'dataset': 'SciFact', 'split': 'dev', 'stage': row['latency_scope'],
                'method': row['method'], 'rerank_depth': row['rerank_depth'],
                'mean_seconds': float(row['latency_mean_seconds']), 'p95_seconds': float(row['latency_p95_seconds']),
                'scope': row['qualification'], 'device': 'original GPU run; see original manifest'})
    evidence = []
    for row in json.loads(evidence_path.read_text()):
        counts = row['counts'];tp,pred,gold = counts['tp'],counts['predicted'],counts['gold']
        p,recall = tp / pred if pred else 0.0, tp / gold if gold else 0.0
        f1 = 2*p*recall/(p+recall) if p+recall else 0.0
        for key,value in [('precision',p),('recall',recall),('f1',f1)]:
            if not isclose(value,row['micro'][key],rel_tol=0,abs_tol=1e-12):
                raise ValueError('SciFact evidence arithmetic mismatch')
        if (row['evaluated_claims'] + row['excluded_no_gold'] != row['total_claims']
                or not isclose(counts['complete_gold_documents'] / counts['gold_documents'],
                               row['complete_rationale_document_recall'], abs_tol=1e-12, rel_tol=0)):
            raise ValueError('SciFact evidence denominator/rationale arithmetic mismatch')
        evidence.append({'dataset':'SciFact','split':'dev','policy':row['policy'],
            'ranking_method':row['ranking_method'],'paper_depth':row['paper_depth'],
            'evaluated_claims':row['evaluated_claims'],'excluded_no_gold':row['excluded_no_gold'],
            **row['micro'],'complete_rationale_document_recall':row['complete_rationale_document_recall'],
            'tp':tp,'predicted_sentences':pred,'gold_sentences':gold})
        evidence[-1]['no_gold_selected_sentences'] = row['no_gold_claims_predicted_sentences']
    cs_dev,cs_test = tasks[0][0],tasks[1][0]
    split_audit = {'query_id_overlap':len(cs_dev['queries'].keys() & cs_test['queries'].keys()),
        'query_paper_id_overlap':len({q['paper_id']for q in cs_dev['queries'].values()} & {q['paper_id']for q in cs_test['queries'].values()}),
        'qualification':'Official query/facet split; not a paper-disjoint independent test.'}
    if split_audit['query_id_overlap']:
        raise ValueError('CSFCube official task splits overlap')
    checks.append({'dataset':'SciFact','summary_metrics_in_range':True,'evidence_micro_arithmetic_verified':True,
        'qualification':'Original measured summaries reused; no new inference or query-level confidence intervals because full original raw rankings are not in this checkout.'})
    sources.update({str(sci_path):file_hash(sci_path),str(evidence_path):file_hash(evidence_path)})
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for name,rows in [('ranking_summary',primary),('ablations',ablations),('uncertainty',intervals),('latency',latency),('evidence_summary',evidence)]:
        write_csv(args.output_dir/f'{name}.csv',rows)
    manifest={'status':'complete','scope':'Person 3 retrieval/ranking and available gold evidence evaluation; not three-domain final verdict evaluation',
        'active_datasets':['SciFact','CSFCube','RECIPER'],'sources_sha256':sources,
        'analysis_sha256':file_hash(Path(__file__)),'checks':checks,'csfcube_split_audit':split_audit,
        'bootstrap_repeats':args.bootstrap_repeats,'bootstrap_seed_base':20261109,
        'numpy_version':importlib.metadata.version('numpy'),
        'uncertainty_policy':'Exploratory paired percentile cluster bootstrap, 2000 draws by default; no multiplicity correction, no confirmatory significance claim or tuning on test results.',
        'replay_policy':'RRF constants replay exact exported branches. Smaller CE prefixes restrict cached CE20 order; no new inference or latency reduction is claimed.',
        'output_sha256':{p.name:file_hash(p)for p in sorted(args.output_dir.glob('*.csv'))}}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(args.output_dir)


if __name__ == '__main__':
    main()
