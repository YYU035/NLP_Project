"""Source integrity and evidence metric regressions on constructed examples."""

from types import SimpleNamespace

import pytest

from ranking_evaluation.evidence_selection import select_evidence
from ranking_evaluation.evaluate_evidence import evaluate, load_rankings, score_selection, summarize


def rationale(*indices):
    return SimpleNamespace(sentences=list(indices))


def test_original_sentences_and_offsets_survive_abbreviations_and_unicode():
    sentences = ['TNF-α is measured, e.g. in blood.', 'Treatment reduces risk by 10%.', 'Other topic.']
    spans = select_evidence('Treatment reduces risk', 'paper-a', sentences, window_size=1, top_k=1)
    assert spans[0].sentence_indices == (1,)
    assert spans[0].text == sentences[1]
    assert ' '.join(sentences)[spans[0].start:spans[0].end] == spans[0].text
    assert spans == select_evidence('Treatment reduces risk', 'paper-a', sentences, top_k=1)


def test_windows_preserve_all_original_sentences_without_rewriting():
    sentences = ['First.', ' Second sentence. ', 'Third.']
    spans = select_evidence('Second', 'p', sentences, window_size=2, top_k=3)
    for span in spans:
        assert span.text == ' '.join(sentences)[span.start:span.end]
        assert span.text == ' '.join(sentences[i] for i in span.sentence_indices)
    with pytest.raises(ValueError, match='sequence'):
        select_evidence('claim', 'p', 'Do not split this string.')
    assert select_evidence('claim', 'p', []) == []


def test_alternative_rationale_completion_and_union_recall_are_distinct():
    row = score_selection({('p', 0)}, {'p': [rationale(0), rationale(2, 3)]})
    assert row['precision'] == 1
    assert row['recall'] == pytest.approx(1 / 3)
    assert row['complete_gold_documents'] == 1
    assert row['any_complete_rationale'] is True


def test_document_ids_prevent_false_sentence_matches_and_missing_gold_is_zero():
    row = score_selection({('wrong-paper', 0)}, {'gold-paper': [rationale(0)]})
    assert (row['tp'], row['predicted'], row['gold']) == (0, 1, 1)
    assert row['recall'] == row['f1'] == 0
    assert score_selection(set(), {'p': [rationale(0)]})['recall'] == 0


def test_no_gold_is_excluded_and_selections_are_still_counted():
    good = score_selection({('p', 0)}, {'p': [rationale(0)]})
    nei = score_selection({('other', 0)}, {})
    result = summarize([good, nei])
    assert result['evaluated_claims'] == 1
    assert result['excluded_no_gold'] == 1
    assert result['micro']['f1'] == 1
    assert result['no_gold_claims_predicted_sentences'] == 1
    assert summarize([nei])['micro'] is None


def test_oracle_vs_retrieved_counts_wrong_papers_and_deduplicates_overlaps():
    docs = {p: SimpleNamespace(abstract=['Claim term.', 'Another sentence.']) for p in ['gold', 'wrong']}
    claims = {'1': SimpleNamespace(claim='Claim term', evidence={'gold': [rationale(0)]}),
              '2': SimpleNamespace(claim='No evidence', evidence={})}
    rows, records = evaluate(docs, claims, {'rrf': {'1': ['wrong'], '2': ['wrong']}}, depths=(1,))
    oracle = next(r for r in rows if r['policy'] == 'context2_top3' and r['ranking_method'] == 'oracle_gold_papers')
    retrieved = next(r for r in rows if r['policy'] == 'context2_top3' and r['ranking_method'] == 'rrf')
    assert oracle['counts']['predicted'] == 2  # overlapping windows are not double-counted
    assert oracle['micro']['recall'] == 1
    assert retrieved['micro']['recall'] == 0
    assert retrieved['counts']['predicted'] == 2  # irrelevant retrieved paper is not filtered using gold
    assert len(records) == 12


def test_report_loader_rejects_wrong_dataset_and_duplicate_ranked_ids(tmp_path):
    import json
    path = tmp_path / 'report.json'
    report = {'dataset': {'dataset_version': 'v1', 'synthetic': False},
              'configuration': {'top_n': 20}, 'rankings': {'rrf': {'1': ['p', 'p']}}}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='dataset'):
        load_rankings([path], dataset_version='v2', claims={'1': None}, docs={'p': None})
    with pytest.raises(ValueError, match='Duplicate'):
        load_rankings([path], dataset_version='v1', claims={'1': None}, docs={'p': None})


def test_character_coverage_merges_spans_and_does_not_credit_partial_sentences():
    from ranking_evaluation.evaluate_evidence import covered_sentence_indices
    assert covered_sentence_indices(['ABC DEF', 'Next.'], [(0, 3), (4, 7)]) == ([0], [])
    assert covered_sentence_indices(['ABC DEF', 'Next.'], [(0, 3)]) == ([], [0])


def test_frontend_adapter_validates_actual_source_text():
    from ranking_evaluation.evaluate_evidence import frontend_spans
    def selector(claim, doc, body, **kwargs):
        return [SimpleNamespace(passage_id=doc+':abstract:0:3:123456789abc',
                                paper_id=doc, text='ABC', score=1)]
    spans = frontend_spans(selector, 'claim', 'p', ['ABC DEF'])
    assert spans[0].sentence_indices == ()
    with pytest.raises(ValueError, match='source text'):
        frontend_spans(selector, 'claim', 'p', ['Different'])
