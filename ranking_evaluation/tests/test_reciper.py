"""Source-label separation and full-corpus team ranking integration."""

import json

import pytest

from ranking_evaluation.evaluate_reciper import evaluate
from ranking_evaluation.pipeline import ModelDescription
from shared.datasets.reciper import from_rows, prepare


def fixture():
    papers = [{'paper_id': f'p{i}', 'title': f'Materials study {i}',
        'abstract': f'Original battery measurements {i}.', 'recipes': ['generated recipe'],
        'sections': [], 'metadata': {}} for i in range(3)]
    questions = [{'question': f'How does temperature affect material {i}?',
        'paper_id': f'p{i}', 'answer': 'Generated answer MUST NOT become scoring text',
        'paper_title': 'Gold title MUST NOT be appended'} for i in range(2)]
    return papers, questions


def test_full_candidate_scope_and_gold_separation(tmp_path):
    papers, questions = fixture()
    dataset = from_rows(papers, questions, manifest={'revision': 'test'})
    assert len(dataset['queries']) == 2
    assert all(row['candidate_ids'] == ['p0', 'p1', 'p2'] for row in dataset['queries'].values())
    assert [row['text'] for row in dataset['queries'].values()] == [row['question'] for row in questions]
    prepare(dataset, tmp_path / 'export')
    text = (tmp_path / 'export/retrieval-input.json').read_text()
    assert 'Generated answer' not in text and 'Gold title' not in text
    payload = json.loads(text)
    assert 'qrels' not in payload
    assert all('recipes' not in document for document in payload['documents'].values())
    assert json.loads((tmp_path / 'export/judgments.json').read_text())['qrels'] == dataset['qrels']
    with pytest.raises(FileExistsError):
        prepare(dataset, tmp_path / 'export')


def test_invalid_references_and_duplicate_texts_rejected():
    papers, questions = fixture()
    questions[0]['paper_id'] = 'not-in-corpus'
    with pytest.raises(ValueError, match='outside the corpus'):
        from_rows(papers, questions, manifest={})
    papers, questions = fixture()
    questions[1]['question'] = questions[0]['question']
    with pytest.raises(ValueError, match='unique'):
        from_rows(papers, questions, manifest={})


def test_metric_hand_calculation_and_complete_collection():
    dataset = from_rows(*fixture(), manifest={})
    a, b = dataset['queries']
    export = {'status': 'complete', 'dataset_version': dataset['version'], 'method': 'fixture',
              'rankings': {a: ['p1', 'p0', 'p2'], b: ['p1', 'p0', 'p2']}}
    result = evaluate(dataset, export)
    assert result['binary_metrics']['macro']['MRR'] == pytest.approx(0.75)
    assert result['binary_metrics']['macro']['Recall@1'] == pytest.approx(0.5)
    assert result['binary_metrics']['macro']['Recall@5'] == 1.0
    export['rankings'][a] = ['p0']
    with pytest.raises(ValueError, match='full corpus'):
        evaluate(dataset, export)


def test_real_team_modules_score_every_paper_and_report_progress():
    np = pytest.importorskip("numpy")
    from ranking_evaluation.run_csfcube import run
    class Encoder:
        def encode(self, texts, **kwargs):
            return np.asarray([[1.0, len(text), sum(map(ord, text)) % 13] for text in texts])
    class Scorer:
        def score_pairs(self, pairs):
            return list(range(len(pairs)))
    dataset = from_rows(*fixture(), manifest={})
    progress = []
    model = ModelDescription(model_id='fixture', revision='fixture', device='cpu',
        dtype='float64', max_length=512, batch_size=2)
    report = run(dataset, encoder=Encoder(), model_name='fixture', scorer=Scorer(),
        model_description=model, top_n=2, evaluator=evaluate,
        progress=lambda *values: progress.append(values))
    assert report['dataset'] == 'RECIPER'
    assert [row[0] for row in progress] == [1, 2]
    assert all(row['rerank_scored_count'] == 2 for row in report['per_query'].values())
    for method in report['rankings'].values():
        assert all(set(ranking) == {'p0', 'p1', 'p2'} for ranking in method.values())


def test_independent_saved_result_arithmetic_detects_tampering():
    from copy import deepcopy
    from ranking_evaluation.verify_reciper_results import verify
    dataset = from_rows(*fixture(), manifest={})
    a, b = dataset['queries']
    rankings = {a: ['p1', 'p0', 'p2'], b: ['p1', 'p0', 'p2']}
    methods = ['bm25', 'sbert', 'rrf', 'rrf_cross_encoder']
    report = {'version': dataset['version'], 'configuration': {'rerank_top_n': 2},
              'rankings': {name: deepcopy(rankings) for name in methods},
              'methods': {name: evaluate(dataset, {'status': 'complete',
                  'dataset_version': dataset['version'], 'method': name, 'rankings': rankings}) for name in methods}}
    result = verify(dataset, report)
    assert result['independent_arithmetic_verified'] is True
    assert result['source_rank_effects']['source_rank_unchanged'] == 2
    report['methods']['bm25']['binary_metrics']['macro']['MRR'] = 1.0
    with pytest.raises(ValueError, match='independent single-source arithmetic'):
        verify(dataset, report)
