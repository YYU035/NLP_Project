"""Historical candidate safety and source integrity across real module boundaries."""

from datetime import date

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("httpx")

from paper_search.metadata import ScientificPaper
from paper_search.retrieval_models import RetrievedPaper, RetrievalResult
from ranking_evaluation.pipeline import ModelDescription
from ranking_evaluation.run_matter_retrieval import process_case, select_cases
from ranking_evaluation.semantic_evidence import LocalSemanticSelector, SelectionConfig


class Encoder:
    def encode(self, texts, **kwargs):
        return np.asarray([[1.0, 0.5] for text in texts])


class Scorer:
    def score_pairs(self, pairs):
        return [0.8] * len(pairs)


def fixture():
    case = {'case_id': 'opaque-1', 'claim_text': 'Batteries retain capacity.',
            'evidence_not_after': '2022-12-31'}
    paper = ScientificPaper(source='openalex', source_id='W123', title='Battery study',
        abstract='Batteries retain capacity. Measurements show stable cycling.',
        publication_date=date(2021, 1, 1))
    entry = RetrievedPaper(identity='openalex:W123', identity_aliases=['openalex:W123'],
        original_claim=case['claim_text'], paper=paper, source_records=[paper], provenance=[])
    candidates = RetrievalResult(original_claim=case['claim_text'], top_k=10, papers=[entry])
    discovery = {'case_id': case['case_id'], 'evidence_not_after': case['evidence_not_after'],
                 'candidates': candidates.model_dump(mode='json')}
    selector = object.__new__(LocalSemanticSelector)
    selector.encoder = Encoder()
    selector.config = SelectionConfig(top_k=3, min_score=0.5)
    model = ModelDescription(model_id='test', revision='test', device='cpu',
        dtype='float32', max_length=512, batch_size=16)
    return case, discovery, selector, model


def test_rank_extract_exact_sources_and_empty_candidates():
    case, discovery, selector, model = fixture()
    result = process_case(case, discovery, selector=selector, scorer=Scorer(), model=model, encoder_id='test')
    assert result['operational']['candidate_count'] == 1
    assert result['operational']['selected_passages'] == 2
    abstract = discovery['candidates']['papers'][0]['paper']['abstract']
    for span in result['evidence'][0]['source_spans']:
        assert abstract[span['start']:span['end']] == span['text']
    assert result['verification_status'] == 'not_run'
    discovery['candidates']['papers'] = []
    empty = process_case(case, discovery, selector=selector, scorer=Scorer(), model=model, encoder_id='test')
    assert empty['operational']['selected_passages'] == 0
    assert empty['evidence'] == []


def test_reject_future_source_alternative_and_claim_mismatch():
    case, discovery, selector, model = fixture()
    discovery['candidates']['papers'][0]['source_records'][0]['publication_date'] = '2023-01-01'
    with pytest.raises(ValueError, match='publication date'):
        process_case(case, discovery, selector=selector, scorer=Scorer(), model=model, encoder_id='test')
    discovery['candidates']['original_claim'] = 'Changed claim'
    with pytest.raises(ValueError, match='claim mismatch'):
        process_case(case, discovery, selector=selector, scorer=Scorer(), model=model, encoder_id='test')


def test_leakage_safe_deterministic_selection():
    case, *_ = fixture()
    payload = {'dataset': 'Matter-of-Fact', 'version': 'v1', 'cases': [case]}
    assert select_cases(payload, 1) == [case]
    case['gold_label'] = True
    with pytest.raises(ValueError, match='leakage-safe'):
        select_cases(payload, 1)
