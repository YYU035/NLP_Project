"""Gold pool boundaries, source claim-ID collisions and ranking metric checks."""

import pytest

from shared.datasets.climatecheck import from_rows
from ranking_evaluation.evaluate_climatecheck import evaluate


def rows():
    return [dict(claim='Claim A', abstract='Relevant text.', claim_id=1, abstract_id=1, annotation='Supports'),
            dict(claim='Claim A', abstract='Uninformative text.', claim_id=1, abstract_id=2, annotation='Not Enough Information'),
            dict(claim='Different claim with same ID', abstract='Contradicting text.', claim_id=1, abstract_id=3, annotation='Refutes')]


def test_exact_text_splits_reused_claim_id_without_merging_gold():
    dataset = from_rows(rows(), split='test', manifest={'revision': 'fixture'})
    assert len(dataset['queries']) == 2
    assert dataset['protocol']['source_claim_ids_with_multiple_texts'] == 1
    assert sorted(len(query['candidate_ids']) for query in dataset['queries'].values()) == [1, 2]
    assert 'pair_labels' not in next(iter(dataset['queries'].values()))


def test_metric_hand_calculation_and_exact_pool_validation():
    dataset = from_rows(rows(), split='test', manifest={})
    rankings = {qid: list(reversed(query['candidate_ids'])) for qid, query in dataset['queries'].items()}
    export = {'status': 'complete', 'dataset_version': dataset['version'], 'method': 'fixture', 'rankings': rankings}
    result = evaluate(dataset, export)
    assert result['binary_metrics']['macro']['MRR@2'] == pytest.approx(0.75)
    assert result['binary_metrics']['macro']['Recall@1'] == pytest.approx(0.5)
    assert result['binary_metrics']['macro']['Recall@2'] == 1.0
    next(iter(rankings.values())).append('foreign-paper')
    with pytest.raises(ValueError, match='judged-pool'):
        evaluate(dataset, export)


def test_duplicate_pair_and_conflicting_abstract_rejected():
    data = rows()
    with pytest.raises(ValueError, match='Duplicate'):
        from_rows([*data, data[0]], split='test', manifest={})
    data[2]['abstract_id'] = 1
    with pytest.raises(ValueError, match='conflicting'):
        from_rows(data, split='test', manifest={})


def test_no_evidentiary_gold_is_excluded_not_scored_as_success():
    dataset = from_rows([rows()[1]], split='test', manifest={})
    rankings = {qid: query['candidate_ids'] for qid, query in dataset['queries'].items()}
    result = evaluate(dataset, {'status': 'complete', 'dataset_version': dataset['version'],
        'method': 'fixture', 'rankings': rankings})
    assert result['binary_metrics']['excluded_no_relevant'] == 1
    assert result['binary_metrics']['macro']['Recall@2'] is None
