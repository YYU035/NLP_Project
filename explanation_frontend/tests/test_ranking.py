import json

import pytest

pytest.importorskip("paper_search")
from explanation_frontend.discovery import sample_discovery, export_discovery, import_discovery, discovery_handoff
from explanation_frontend.ranking import connect_ranking, ranking_request, ranked_candidates


def connected():
    snapshot = sample_discovery()
    request = ranking_request(snapshot)
    ids = [p.identity for p in snapshot.result.papers[:3]]
    request.update(bm25_ranking=ids, sbert_ranking=ids[::-1])
    output = connect_ranking(json.dumps(request), snapshot.result)
    return snapshot, request, output


def test_real_upstream_rrf_roundtrip_keeps_metadata_and_handoff():
    snapshot, request, output = connected()
    assert output['papers'][0]['rrf_score'] == pytest.approx(1/61+1/63)
    assert all(row['entry'] in request['candidates']['papers'] for row in output['papers'])
    ids = [row['entry']['identity'] for row in output['papers']]
    snapshot = type(snapshot).model_validate({**snapshot.model_dump(mode='json'),
        'ranking_context': output, 'selected_paper_ids': ids})
    restored = import_discovery(export_discovery(snapshot, ids))
    assert restored.ranking_context == output
    assert discovery_handoff(restored, ids)['ranking_context'] == output
    assert [p.identity for p in ranked_candidates(restored.result.papers[::-1], output)] == ids
    assert connect_ranking(json.dumps(output), snapshot.result) == output


@pytest.mark.parametrize('mutation', ['claim', 'metadata', 'duplicate', 'unknown', 'score', 'rank', 'count', 'rerank'])
def test_rejects_stale_or_tampered_rankings(mutation):
    snapshot, request, output = connected()
    if mutation == 'claim':
        output['inputs']['candidates']['original_claim'] += '!'
    elif mutation == 'metadata':
        output['inputs']['candidates']['papers'][0]['paper']['title'] = 'changed'
    elif mutation == 'duplicate':
        output['inputs']['bm25_ranking'] *= 2
    elif mutation == 'unknown':
        output['inputs']['sbert_ranking'] = ['unknown']
    elif mutation == 'score':
        output['papers'][0]['rrf_score'] = 999
    elif mutation == 'rank':
        output['papers'][0]['rank'] = 2
    elif mutation == 'count':
        output['ranked_union_size'] = 999
    else:
        output['configuration']['rerank_enabled'] = True
    with pytest.raises(ValueError):
        connect_ranking(json.dumps(output), snapshot.result)


def test_empty_branch_is_diagnostic_not_verdict_and_no_model_is_loaded(monkeypatch):
    import socket
    monkeypatch.setattr(socket, 'create_connection', lambda *a, **k: pytest.fail('Unexpected network'))
    snapshot = sample_discovery()
    output = connect_ranking(json.dumps(ranking_request(snapshot)), snapshot.result)
    assert output['papers'] == []
    assert len(output['warnings']) >= 2
    assert output['model'] is None
    assert output['latency']['end_to_end_seconds'] is None


def test_excluded_candidates_cannot_leak_into_verification_selection():
    snapshot, _, output = connected()
    data = snapshot.model_dump(mode='json')
    data['ranking_context'] = output
    with pytest.raises(ValueError):
        type(snapshot).model_validate(data)  # initial all-paper selection includes unranked papers


def test_ranked_browser_handoff_navigation_and_restore():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    snapshot, _, output = connected()
    ids = [row['entry']['identity'] for row in output['papers']]
    snapshot = type(snapshot).model_validate({**snapshot.model_dump(mode='json'),
        'ranking_context': output, 'selected_paper_ids': ids})
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py'))
    app.session_state['discovery_snapshot'] = snapshot
    app.session_state['discovery_selection'] = {p.identity: p.identity in ids for p in snapshot.result.papers}
    app.run()
    assert not app.exception
    assert len(app.checkbox) == 3
    assert any('RRF rank' in t.value for t in app.text)
    next(b for b in app.button if b.label == 'Prepare selected papers for verification').click().run()
    assert not app.exception
    assert set(app.session_state['pending_discovery']['ids']) == set(ids)
    assert app.session_state['pending_discovery']['snapshot'].ranking_context == output
    app.radio(key='workspace').set_value('Find papers').run()
    next(b for b in app.button if b.label == 'Restore discovery order').click().run()
    assert not app.exception
    assert app.session_state['discovery_snapshot'].ranking_context is None
    assert any(m.label == 'Selected' and m.value == '32' for m in app.metric)
