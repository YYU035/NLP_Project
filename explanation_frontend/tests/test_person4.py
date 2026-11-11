import pytest

from explanation_frontend.person4 import Person4Adapter
from explanation_frontend.providers import FixtureProvider
from explanation_frontend._upstream.claim_verifier.schemas import VerificationResponse


def native_fixture():
    source = FixtureProvider().run_fixture("supported")
    data = source.verification.model_dump(mode="json")
    data['claim'] = source.claim
    data['per_paper_verdicts'] = list(data['per_paper_verdicts'].values())
    for row in data['per_paper_verdicts']:
        row['evidence_type'] = 'abstract'
    data['per_passage_labels'] = list(data['per_passage_labels'].values())
    settings = data['model_settings']
    settings['tokenizer_max_length'] = settings.pop('max_length')
    settings['threshold'] = 0.50
    settings['seed'] = 42
    return source, data


def aggregator():
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from explanation_frontend._upstream.claim_verifier.verifier import ClaimVerifier
    # Actual pure-Python owner method; never construct/download a model.
    engine = object.__new__(ClaimVerifier)
    return engine.aggregate


def test_actual_owner_aggregation_maps_and_reuses_original_response():
    source, data = native_fixture()
    calls = []
    fn = aggregator()
    def record(response, **kwargs):
        calls.append(response)
        assert isinstance(response, VerificationResponse)
        return fn(response, **kwargs)
    adapter = Person4Adapter(record)
    snapshot = adapter.snapshot(data, list(source.papers.values()))
    assert snapshot.provisional.aggregated_verdict == 'SUPPORTED'
    assert all(v.evidence_type == 'abstract_only' for v in snapshot.verification.per_paper_verdicts.values())
    result = adapter.aggregate(snapshot, [])
    assert result.aggregated_verdict == 'INSUFFICIENT'
    assert result.sufficiency_cause == 'EMPTY_SELECTION'
    assert calls[0] is calls[1]
    assert adapter.aggregate(snapshot, list(snapshot.papers)).aggregated_verdict == 'SUPPORTED'
    with pytest.raises(ValueError):
        adapter.aggregate(snapshot, ['unknown'])


def test_total_truncation_is_operational_failure():
    source, data = native_fixture()
    for label in data['per_passage_labels']:
        label.update(truncated=True, operational_status='TRUNCATED')
    for row in data['per_paper_verdicts']:
        row.update(verdict='INSUFFICIENT', supporting_passage_ids=[], contradicting_passage_ids=[])
    adapter = Person4Adapter(lambda *a, **k: pytest.fail('Should not aggregate failed processing'))
    snapshot = adapter.snapshot(data, list(source.papers.values()))
    assert snapshot.provisional.status == 'failed'
    assert snapshot.provisional.aggregated_verdict is None


@pytest.mark.parametrize('mutation', ['duplicate', 'foreign', 'conflict'])
def test_invalid_or_unresolved_outputs_fail_closed(mutation):
    source, data = native_fixture()
    if mutation == 'duplicate':
        data['per_passage_labels'] *= 2
    elif mutation == 'foreign':
        data['per_paper_verdicts'][0]['paper_id'] = 'foreign'
    else:
        data['per_paper_verdicts'][0]['conflict_flag'] = True
    with pytest.raises(ValueError):
        Person4Adapter(aggregator()).snapshot(data, list(source.papers.values()))
