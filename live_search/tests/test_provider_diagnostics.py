"""Offline checks for the bounded diagnostic request plan and observations."""

import httpx

from scripts.provider_diagnostics import (
    arxiv_followup_probes,
    arxiv_probes,
    summarize_response,
)


def test_followup_is_arxiv_only_and_reuses_saved_queries() -> None:
    saved = {
        "generation": {
            case: {"queries": ["original", "graph neural networks"]}
            for case in ("c01", "c02", "c08")
        }
    }
    probes = arxiv_followup_probes(saved)
    assert len(probes) == 7
    assert {probe.provider for probe in probes} == {"arxiv"}
    assert probes[0].params["id_list"] == "1706.03762"
    for probe in probes[1:]:
        assert "all:graph AND all:neural AND all:networks" in (
            probe.params["search_query"]
        )


def test_arxiv_matrix_preserves_terms_and_pairs_dates() -> None:
    saved = {
        "cases": [
            {
                "id": "c01",
                "claim": "TNF-α does not increase COVID-19 risk.",
            },
            {
                "id": "c02",
                "claim": "Lithium-ion batteries retain 90% capacity.",
            },
            {
                "id": "c08",
                "claim": "Graph neural networks improve predictions.",
            },
        ]
    }
    probes = arxiv_probes(saved)
    assert len(probes) == 26
    assert len({probe.id for probe in probes}) == 26
    for claim in ("c01", "c02", "c08"):
        matrix = [
            p
            for p in probes
            if p.claim_id == claim
            and p.variant in {"current", "broad", "phrases"}
        ]
        assert len(matrix) == 6
        for variant in ("current", "broad", "phrases"):
            pair = [p for p in matrix if p.variant == variant]
            assert {p.dated for p in pair} == {True, False}
            undated = next(p for p in pair if not p.dated)
            dated = next(p for p in pair if p.dated)
            assert dated.params["search_query"].startswith(
                "("
                + undated.params["search_query"]
                + ") AND submittedDate:"
            )
    assert (
        '"TNF-α"'
        in next(
            p for p in probes if p.id == "arxiv-c01-broad-dated"
        ).params["search_query"]
    )
    assert (
        "Drops"
        in next(
            p for p in probes if p.id == "arxiv-tnf-family"
        ).relaxation
    )


def test_provider_errors_are_not_zero_results() -> None:
    for provider in ("arxiv", "core", "openaire", "europe_pmc"):
        result = summarize_response(provider, httpx.Response(429))
        assert result["status"] == "http_error"
        assert result["result_count"] is None
    invalid = summarize_response(
        "arxiv", httpx.Response(200, text="<html/>")
    )
    assert invalid["status"] == "unexpected_xml"
    assert invalid["result_count"] is None


def test_semantic_sparse_empty_and_europe_pmc_abstracts() -> None:
    result = summarize_response(
        "semantic_scholar",
        httpx.Response(200, json={"total": 0, "offset": 0}),
    )
    assert result["status"] == "zero_results"
    observed = summarize_response(
        "europe_pmc",
        httpx.Response(
            200,
            json={
                "hitCount": 1,
                "resultList": {
                    "result": [
                        {
                            "id": "1",
                            "title": "A title",
                            "abstractText": "An abstract",
                            "pmcid": "PMC1",
                            "isOpenAccess": "Y",
                        }
                    ]
                },
            },
        ),
    )
    assert observed["records"][0]["abstractText"] == "An abstract"
