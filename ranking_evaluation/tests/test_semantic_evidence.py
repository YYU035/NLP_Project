from dataclasses import replace
from types import SimpleNamespace

import pytest

from ranking_evaluation.semantic_evidence import (
    SelectionConfig, LocalSemanticSelector, select_scored_sentences, person4_passages, select_ranked_evidence,
)


def test_threshold_abstention_ties_and_context_merge():
    sentences = ['First.', 'Second.', 'Third.', 'Fourth.']
    assert select_scored_sentences('p', sentences, [0.1]*4) == []
    spans = select_scored_sentences('p', sentences, [.8, .8, .2, .1],
        config=SelectionConfig(top_k=2, min_score=.3, context_radius=1))
    assert len(spans) == 1
    assert spans[0].sentence_indices == (0, 1, 2)
    assert spans[0].text == ' '.join(sentences[:3])
    assert select_scored_sentences('p', sentences, [.8]*4)[0].sentence_indices == (0,)


def test_rejects_nonfinite_scores_and_invalid_configuration():
    with pytest.raises(ValueError):
        select_scored_sentences('p', ['s'], [float('nan')])
    with pytest.raises(ValueError):
        SelectionConfig(min_score=float('nan'))
    with pytest.raises(ValueError):
        SelectionConfig(top_k=True)


def test_live_span_preserves_original_whitespace_and_person4_contract():
    selector = LocalSemanticSelector.__new__(LocalSemanticSelector)
    selector.config = SelectionConfig(context_radius=1)
    selector.select = lambda claim, pid, sentences: select_scored_sentences(
        pid, sentences, [.8]*len(sentences), config=selector.config)
    abstract = '  First sentence.   Second sentence.\nThird sentence.  '
    spans = selector.select_abstract('claim', 'p', abstract)
    assert spans[0].text == abstract[spans[0].start:spans[0].end]
    assert '   ' in spans[0].text
    payload = person4_passages(spans)
    assert set(payload[0]) == {'passage_id', 'paper_id', 'section', 'text', 'score'}
    with pytest.raises(ValueError):
        person4_passages([spans[0], spans[0]])


def test_ranking_handoff_keeps_source_and_original_claim():
    from ranking_evaluation.examples.mock_input import build_mock_input
    from ranking_evaluation.pipeline import rank_claim
    ranking = rank_claim(build_mock_input())
    selector = SimpleNamespace(config=SelectionConfig(), select_abstract=lambda *args: [])
    output = select_ranked_evidence(ranking, selector, tokenizer=SimpleNamespace(encode=lambda *a, **k: [1]), paper_depth=1)
    assert output['claim'] == ranking.inputs.candidates.original_claim
    assert output['papers'][0]['source_entry']['identity'] == ranking.papers[0].entry.identity
    assert output['papers'][0]['status'] == 'no_passage_above_threshold'


def test_frontend_bridge_checks_frozen_model_identity(tmp_path, monkeypatch):
    import json
    from ranking_evaluation.frontend_bridge import load_selector
    model=tmp_path/'model';model.mkdir()
    (model/'download_manifest.json').write_text(json.dumps({'model_id':'wrong','revision':'r','files_sha256':{}}))
    config=tmp_path/'config.json';config.write_text(json.dumps({'model_id':'expected','model_revision':'r'}))
    load_selector.cache_clear()
    with pytest.raises(ValueError,match='frozen'):
        load_selector(str(config),str(model))


def test_frontend_installer_is_idempotent_and_preserves_legacy(tmp_path):
    from ranking_evaluation.scripts.install_frontend_adapter import install
    module=tmp_path/'explanation_frontend';module.mkdir()
    (module/'passage_selection.py').write_text('def select_abstract(claim, identity, abstract, *, top_k=3):\n    return []\n')
    (module/'pipeline.py').write_text('import json\nclass Pipeline:\n    def run(self):\n        passages = {pid: select_abstract(claim, pid, entry_map[pid].paper.abstract) for pid in ordered}\n        data = {}\n        if source.result.status == \"failed\":\n            return data\n')
    install(tmp_path);install(tmp_path)
    text=(module/'passage_selection.py').read_text()
    assert text.count('# Person 3 opt-in semantic evidence bridge')==1
    assert 'return []' in text
    assert (module/'_person3/SOURCE.json').is_file()


def test_frontend_validation_accepts_explicit_partial_and_rejects_wrong_source(tmp_path):
    import json
    from ranking_evaluation.scripts.validate_frontend_evidence import validate
    for mode in ('recorded','live'):
        (tmp_path/f'{mode}-report.json').write_text(json.dumps({'candidate_count':1,'verified_papers':1,
            'passages':1,'status':'partial','displayable':True}))
        snapshot={'claim':'claim','stage_statuses':{'passages':'person3_semantic_train_frozen'},
            'discovery_context':{'result':{'original_claim':'claim','papers':[{'identity':'p','paper':{'abstract':'ABC'}}],
                'providers':{'openaire':{'status':'provider_failure','error':'network'}}}},
            'verification':{'evidence_passages':[{'passage_id':'p:abstract:0:3:123456789abc','paper_id':'p','text':'ABC'}]}}
        (tmp_path/f'{mode}-run.json').write_text(json.dumps({'snapshot':snapshot}))
    assert validate(tmp_path)['live']['application_status']=='partial'
    path=tmp_path/'live-run.json';raw=json.loads(path.read_text())
    raw['snapshot']['verification']['evidence_passages'][0]['text']='changed'
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='quotation'):
        validate(tmp_path)
