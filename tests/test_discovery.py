import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_verifier.discovery.service import import_discovery, export_discovery, DiscoveryResult, discovery_handoff, filter_candidates, sample_discovery, LiveDiscoveryBridge


def payload():
    return {"original_claim": "  Exact claim with 12% and uncertainty.  ", "top_k": 30, "limit_scope": "per_provider",
        "papers": [{"identity": "doi:10.1/example", "identity_aliases": ["doi:10.1/example", "arxiv:1234.5678"],
            "original_claim": "  Exact claim with 12% and uncertainty.  ",
            "paper": {"source": "openalex", "source_id": "W1", "title": None, "abstract": None,
                      "pdf_url": "https://example.org/file.pdf"},
            "provenance": [{"provider": "openalex", "source_id": "W1", "query_id": "q1", "rank": 1}],
            "source_records": [{"source": "openalex", "source_id": "W1", "title": None}]}],
        "providers": {"openalex": {"provider": "openalex", "status": "success", "queries": [{"id": "q1", "provider": "openalex", "strategy": "keybert", "stage": "primary", "query": "reformulation", "status": "success", "attempts": []}]},
            "arxiv": {"provider": "arxiv", "status": "zero_results"}}, "seconds": 2.5, "http_attempts": 1}


def test_import_exact_upstream_keeps_aliases_and_missing_metadata_without_verdict():
    snapshot = import_discovery(json.dumps(payload()))
    assert snapshot.result.original_claim == "  Exact claim with 12% and uncertainty.  "
    assert snapshot.mode == "cached"
    assert snapshot.retrieved_at is None  # raw RetrievalResult has no trustworthy retrieval timestamp
    p = snapshot.result.papers[0]
    assert p.paper.title is None and p.paper.abstract is None
    assert p.identity_aliases == ["doi:10.1/example", "arxiv:1234.5678"]
    assert "verdict" not in p.model_dump()
    assert not hasattr(p.paper, "has_fulltext")


def test_real_recorded_example_validates_and_has_real_sources():
    snapshot = sample_discovery()
    assert snapshot.mode == "cached"
    assert len(snapshot.result.papers) == 32
    assert set(snapshot.result.providers) == {"openalex", "openaire", "arxiv"}
    assert "259d2a9" in snapshot.source_reference


def test_discovery_export_roundtrip_and_selection_handoff_are_lossless():
    snapshot = import_discovery(json.dumps(payload()))
    raw = export_discovery(snapshot, ["doi:10.1/example"])
    replay = import_discovery(raw)
    assert replay.result == snapshot.result
    assert replay.selected_paper_ids == ["doi:10.1/example"]
    handoff = discovery_handoff(replay, replay.selected_paper_ids)
    assert handoff["original_claim"] == payload()["original_claim"]
    assert handoff["papers"][0]["source_records"][0]["source_id"] == "W1"
    assert handoff["verification_status"] == "not_yet_verified"
    changed = json.loads(raw)
    changed["snapshot"]["result"]["papers"][0]["paper"]["title"] = "Changed"
    with pytest.raises(ValueError):
        import_discovery(json.dumps(changed))
    with pytest.raises(ValueError):
        discovery_handoff(snapshot, ["unknown"])


@pytest.mark.parametrize("mutation", ["claim", "duplicate", "identity", "provider", "date", "nan", "pending"])
def test_malformed_or_incomplete_handoffs_fail_explicitly(mutation):
    data = payload()
    if mutation == "claim": data["papers"][0]["original_claim"] = "Rewritten claim"
    if mutation == "duplicate": data["papers"] *= 2
    if mutation == "identity": data["papers"][0]["identity_aliases"] = []
    if mutation == "provider": data["papers"][0]["provenance"][0]["provider"] = "absent"
    if mutation == "date": data.update(from_publication_date="2026-01-01", to_publication_date="2020-01-01")
    if mutation == "nan": data["seconds"] = float("nan")
    if mutation == "pending": data["providers"]["openalex"]["status"] = "pending"
    with pytest.raises(ValueError): import_discovery(json.dumps(data))


def test_zero_results_partial_failure_and_total_outage_are_not_scientific_insufficiency():
    data = payload()
    assert DiscoveryResult.model_validate(data).status == "ok"
    data["providers"]["arxiv"]["status"] = "provider_failure"
    assert DiscoveryResult.model_validate(data).status == "partial"
    data["papers"] = []
    data["providers"]["openalex"]["status"] = "query_generation_failure"
    assert DiscoveryResult.model_validate(data).status == "failed"
    for provider in data["providers"].values(): provider["status"] = "zero_results"
    assert DiscoveryResult.model_validate(data).status == "empty"


def test_candidate_filter_is_not_a_scientific_reranker():
    snapshot = sample_discovery()
    original = [p.identity for p in snapshot.result.papers]
    subset = filter_candidates(snapshot.result, text="", providers=["arxiv"], pdf_only=False)
    assert subset == []
    assert [p.identity for p in snapshot.result.papers] == original
    assert filter_candidates(snapshot.result, text="", providers=[], pdf_only=False) == snapshot.result.papers


def test_async_bridge_reuses_service_and_preserves_claim_and_options():
    instances, calls, closed = [], [], []
    class Service:
        async def __aenter__(self): instances.append(self); return self
        async def __aexit__(self, *args): closed.append(True)
        async def retrieve(self, claim, **options):
            calls.append((claim, options))
            return SimpleNamespace(model_dump=lambda **kwargs: payload())
    bridge = LiveDiscoveryBridge(service_factory=Service)
    try:
        for _ in range(2):
            result = bridge.retrieve(payload()["original_claim"], top_k=30, from_publication_date=None, to_publication_date=None)
            assert result.mode == "live"
        assert len(instances) == 1
        assert len(calls) == 2
        assert calls[0][0] == payload()["original_claim"]
        assert calls[0][1]["top_k"] == 30
    finally:
        bridge.close()
    assert closed == [True]


def test_async_outage_preserves_diagnostics_and_mismatched_claim_rejected():
    data = payload()
    data["papers"] = []
    for provider in data["providers"].values(): provider["status"] = "provider_failure"
    class Outage(Exception):
        result = SimpleNamespace(model_dump=lambda **kwargs: data)
    class Service:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def retrieve(self, claim, **kwargs): raise Outage()
    bridge = LiveDiscoveryBridge(service_factory=Service)
    try:
        assert bridge.retrieve(data["original_claim"]).result.status == "failed"
        with pytest.raises(ValueError): bridge.retrieve("Another claim")
    finally:
        bridge.close()
