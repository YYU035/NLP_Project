"""Source preservation and selective-neighbor behavior for the experiment policy."""
from ranking_evaluation.analysis.kireeti_validation_20261110.train_context import segments, select
from ranking_evaluation.semantic_evidence import SelectionConfig, select_scored_sentences


def test_baseline_matches_existing_selection_with_original_whitespace():
    text = 'First sentence.   Second sentence!\nThird sentence? Fourth sentence.'
    bounds = segments(text)
    scores = [.7, .55, .1, .6]
    expected = select_scored_sentences('p', [text[a:b] for a,b in bounds], scores, config=SelectionConfig(3,.5,0))
    actual = select('p', text, bounds, scores, None)
    assert [s.text for s in actual] == [s.text for s in expected]
    assert all(text[s.start:s.end] == s.text for s in actual)


def test_neighbor_gate_preserves_anchor_and_excludes_low_relevance_context():
    text = 'Low context. Anchor sentence. Relevant context.'
    spans = select('p',text,segments(text),[.1,.8,.45],.4)
    assert len(spans) == 1
    assert spans[0].text == 'Anchor sentence. Relevant context.'
    assert spans[0].score == .8


def test_overlapping_context_merges_without_duplicate_text():
    text = 'First anchor. Shared context. Last anchor.'
    spans = select('p',text,segments(text),[.8,.4,.7],.3)
    assert len(spans) == 1
    assert spans[0].text == text
    assert spans[0].sentence_indices == (0,1,2)


def test_full_rankings_reject_incomplete_unknown_and_duplicate_inputs():
    import pytest
    from ranking_evaluation.analysis.kireeti_validation_20261110.pipeline_context import validate_rankings
    docs = {str(i): {} for i in range(6)}
    claims = {'c': {}}
    assert validate_rankings({'rrf': {'c': list(docs)}}, claims, docs) == {'c': list(docs)[:5]}
    for ranks in [{}, {'c': ['0'] * 5}, {'c': ['0','1','2','3','unknown']}]:
        with pytest.raises(ValueError):
            validate_rankings({'rrf': ranks}, claims, docs)
