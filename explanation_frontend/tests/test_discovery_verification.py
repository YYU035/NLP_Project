import pytest

from explanation_frontend.discovery import sample_discovery
from explanation_frontend.models import RunSnapshot
from explanation_frontend.providers import FromDiscoveryProvider, FixtureProvider


def test_selected_discovery_verification_preserves_claim_and_calls_downstream_once():
    search = sample_discovery()
    pid = search.result.papers[0].identity
    calls = []
    def verify(handoff):
        calls.append(handoff)
        base = FixtureProvider().run_fixture("empty").model_dump()
        base.update(claim=search.result.original_claim, mode="cached")
        base["provisional"]["claim"] = search.result.original_claim
        return base
    provider = FromDiscoveryProvider(verify_selected=verify, aggregate=lambda response, ids: response.provisional)
    snapshot = provider.run_from_discovery(search, [pid])
    assert len(calls) == 1
    assert calls[0]["papers"][0]["identity"] == pid
    assert len(calls[0]["papers"]) == 1
    assert snapshot.claim == search.result.original_claim
    assert snapshot.mode == "cached"
    assert snapshot.discovery_context["result"]["papers"][0]["identity"] == pid
    assert provider.aggregate(snapshot, []).aggregated_verdict == "INSUFFICIENT"
    assert len(calls) == 1


def test_downstream_cannot_attach_a_verdict_to_a_different_claim_or_paper():
    search = sample_discovery()
    provider = FromDiscoveryProvider(verify_selected=lambda handoff: FixtureProvider().run_fixture("supported"), aggregate=lambda *a: None)
    with pytest.raises(ValueError): provider.run_from_discovery(search, [search.result.papers[0].identity])
    with pytest.raises(ValueError): provider.run_from_discovery(search, [])
