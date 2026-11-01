"""Offline coverage, identity grouping, archives, and experimental design."""

import asyncio
import hashlib
import json
from datetime import date
from pathlib import Path

import httpx
import pytest

from paper_search.clients.arxiv import ArxivClient
from paper_search.clients.openalex import OpenAlexClient
from paper_search.clients.semantic_scholar import (
    SemanticScholarClient,
)
from paper_search.config import Settings
from paper_search.deduplication import (
    arxiv_ids,
    group_papers,
    same_paper,
)
from paper_search.evaluation.analysis import compare_calls
from paper_search.evaluation.models import (
    ClaimCase,
    ExperimentOptions,
    SearchObservation,
)
from paper_search.evaluation.replay import (
    reparse_sparse_empty_responses,
)
from paper_search.evaluation.reporting import write_results
from paper_search.evaluation.runner import run_experiment
from paper_search.exceptions import APIClientError
from paper_search.metadata import ScientificPaper


def _paper(identifier: str, **kwargs) -> ScientificPaper:
    return ScientificPaper(
        source="example", source_id=identifier, **kwargs
    )


def _call(
    kind: str, papers=None, status="success"
) -> SearchObservation:
    return SearchObservation(
        id=kind,
        provider="example",
        claim_id="c01",
        kind=kind,
        query="claim",
        status=status,
        papers=papers or [],
        result_count=len(papers or [])
        if status != "provider_failure"
        else None,
        authentication="anonymous",
        attempts=[{"status": 200}],
        elapsed_seconds=1,
    )


def test_doi_arxiv_and_conservative_title_identity() -> None:
    assert same_paper(
        _paper("a", doi="https://doi.org/10.1/ABC"),
        _paper("b", doi="doi:10.1/abc"),
    )
    arxiv = ScientificPaper(
        source="arxiv", source_id="2401.12345v2"
    )
    repository = _paper(
        "a", pdf_url="https://arxiv.org/pdf/2401.12345v1.pdf"
    )
    assert arxiv_ids(arxiv) == {"2401.12345"}
    assert same_paper(arxiv, repository)
    attrs = {
        "title": "Scientific investigation of "
        "protein stability under pressure",
        "authors": ["Alice Example"],
        "year": 2024,
    }
    assert same_paper(_paper("a", **attrs), _paper("b", **attrs))
    assert not same_paper(
        _paper("a", **attrs), _paper("b", **{**attrs, "year": 2023})
    )
    assert not same_paper(
        _paper("a", **attrs),
        _paper("b", **{**attrs, "authors": []}),
    )
    assert not same_paper(
        _paper("a", title="Short title"),
        _paper("b", title="Short title"),
    )


def test_conflicting_ids_never_merge_through_title_bridge() -> None:
    attrs = {
        "title": "Scientific investigation of "
        "protein stability under pressure",
        "authors": ["Alice Example"],
        "year": 2024,
    }
    papers = [
        _paper("a", doi="10.1/first", **attrs),
        _paper("b", **attrs),
        _paper("c", doi="10.1/second", **attrs),
    ]
    assert len(group_papers(papers).groups) == 2
    assert not same_paper(papers[0], papers[2])


def test_natural_and_matched_budgets_use_actual_repeat_results() -> (
    None
):
    calls = [
        _call("original", [_paper("a")]),
        _call("alternative", [_paper("b")]),
        _call("original_repeat", [_paper("c")]),
    ]
    natural = compare_calls(calls)
    assert natural["original_unique"] == 1
    assert natural["reformulated_unique"] == 2
    assert natural["additional"] == 1
    assert natural["original_logical_requests"] == 1
    assert natural["reformulated_logical_requests"] == 2
    matched = compare_calls(calls, matched=True)
    assert matched["original_unique"] == 2
    assert matched["lost"] == 1
    assert matched["wire_budget_matched"]
    assert matched["original_latency_seconds"] == 2
    assert natural["precision_at_k"] is None


def test_failure_is_never_a_zero_result_or_no_difference() -> None:
    calls = [
        _call("original", [], "zero_results"),
        _call("alternative", status="provider_failure"),
    ]
    comparison = compare_calls(calls)
    assert comparison["original_zero"] is True
    assert comparison["reformulated_zero"] is None
    assert comparison["additional"] is None
    assert comparison["status"] == "incomplete_provider_failure"


def test_mocked_full_experiment_saves_responses_and_uses_identical_filters(
    tmp_path,
    monkeypatch,
) -> None:
    for cls in (OpenAlexClient, ArxivClient, SemanticScholarClient):
        monkeypatch.setattr(cls, "min_interval", 0)
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "api.openalex.org":
            assert request.url.params["per_page"] == "3"
            assert request.url.params["filter"] == (
                "from_publication_date:2020-01-01,to_publication_date:2026-10-30"
            )
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "https://openalex.org/W1",
                            "title": "Memory",
                            "publication_date": "2024-01-01",
                        }
                    ]
                },
            )
        if request.url.host == "api.semanticscholar.org":
            assert (
                request.url.params["publicationDateOrYear"]
                == "2020-01-01:2026-10-30"
            )
            assert request.url.params["limit"] == "3"
            return httpx.Response(200, json={"data": []})
        assert (
            "202001010000 TO 202610302359"
            in request.url.params["search_query"]
        )
        assert request.url.params["max_results"] == "3"
        return httpx.Response(
            200, text='<feed xmlns="http://www.w3.org/2005/Atom"/>'
        )

    cases = [
        ClaimCase(
            id="c01",
            domain="science",
            claim="Exercise improves memory.",
        )
    ]
    options = ExperimentOptions(
        max_results=3,
        from_publication_date=date(2020, 1, 1),
        to_publication_date=date(2026, 10, 30),
    )
    manifest = asyncio.run(
        run_experiment(
            cases,
            Settings(),
            options,
            tmp_path,
            transport=httpx.MockTransport(handler),
        )
    )
    results = write_results(manifest, tmp_path)
    assert (
        len(requests) == 3
    )  # Same original serves both arms, not six calls.
    assert len(list((tmp_path / "responses").glob("*.body"))) == 3
    assert (tmp_path / "calls.csv").exists()
    assert (tmp_path / "papers.csv").exists()
    assert (tmp_path / "report.md").exists()
    assert (
        results["provider_summary"]["openalex"][
            "original_successes"
        ]
        == 1
    )
    assert (
        results["provider_summary"]["semantic_scholar"][
            "zero_result_calls"
        ]
        == 1
    )
    saved = json.loads(
        (tmp_path / "results.json").read_text(encoding="utf-8")
    )
    assert saved["precision_at_k"] is None
    assert saved["comparisons"][0]["natural"][
        "logical_budget_matched"
    ]


@pytest.mark.parametrize(
    "payload,empty",
    [
        ({"total": 0, "offset": 0}, True),
        ({"total": 1, "offset": 0}, False),
        ({"total": False, "offset": 0}, False),
        ({"total": 0}, False),
        ({"total": 0, "offset": 0, "data": None}, False),
    ],
)
def test_observed_semantic_scholar_empty_envelope(
    payload, empty
) -> None:
    """Accept only explicit sparse empty envelopes as zero matches."""

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json=payload)
            )
        ) as http:
            if empty:
                assert (
                    await SemanticScholarClient(http).search(
                        "claim"
                    )
                    == []
                )
            else:
                with pytest.raises(APIClientError):
                    await SemanticScholarClient(http).search(
                        "claim"
                    )

    asyncio.run(run())


def test_sparse_empty_replay_retains_original_evidence(
    tmp_path,
) -> None:
    """Replay the fixture without changing observed request counts/timings."""
    fixture = (
        Path(__file__).parent
        / "fixtures/semantic_scholar_sparse_empty.json"
    )
    body = fixture.read_bytes()
    (tmp_path / "responses").mkdir()
    (tmp_path / "responses/empty.body").write_bytes(body)
    call = SearchObservation(
        id="ss-empty",
        provider="semantic_scholar",
        claim_id="c01",
        kind="original",
        query="claim",
        status="provider_failure",
        authentication="anonymous",
        error="no data list",
        elapsed_seconds=1.5,
        attempts=[
            {
                "status": 200,
                "response_file": "responses/empty.body",
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        ],
    )
    manifest = {
        "options": ExperimentOptions().model_dump(mode="json"),
        "calls": [call.model_dump(mode="json")],
    }
    corrected = reparse_sparse_empty_responses(manifest, tmp_path)
    result = corrected["calls"][0]
    assert result["status"] == "zero_results"
    assert result["result_count"] == 0
    assert result["elapsed_seconds"] == 1.5
    assert len(result["attempts"]) == 1
    assert (
        result["parser_correction"]["observed_status"]
        == "provider_failure"
    )
    assert manifest["calls"][0]["status"] == "provider_failure"


def test_full_two_query_budget_is_executed_and_counted(
    tmp_path, monkeypatch
) -> None:
    """Each extra query and fresh repeat control costs a real client call."""
    from paper_search.linguistic import Concept, ExtractionResult

    def extract(self, claim: str) -> ExtractionResult:
        return ExtractionResult(
            status="ok",
            concepts=[
                Concept(
                    text=text,
                    start=claim.index(text),
                    end=claim.index(text) + len(text),
                    kind="noun_phrase",
                )
                for text in ("physical activity", "working memory")
            ],
        )

    monkeypatch.setattr(
        "paper_search.linguistic.SpacyConceptExtractor.extract",
        extract,
    )
    monkeypatch.setattr(OpenAlexClient, "min_interval", 0)
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        identifier = (
            "W1"
            if request.url.params["search"].startswith("Our")
            else "W2"
        )
        return httpx.Response(
            200, json={"results": [{"id": identifier}]}
        )

    cases = [
        ClaimCase(
            id="c01",
            domain="science",
            claim=(
                "Our results show that physical activity improves "
                "working memory."
            ),
        )
    ]
    settings = Settings(
        arxiv_enabled=False, semantic_scholar_enabled=False
    )
    manifest = asyncio.run(
        run_experiment(
            cases,
            settings,
            ExperimentOptions(),
            tmp_path,
            transport=httpx.MockTransport(handler),
        )
    )
    result = write_results(manifest, tmp_path)
    pair = result["comparisons"][0]
    assert len(requests) == 3
    assert pair["natural"]["original_unique"] == 1
    assert pair["natural"]["reformulated_unique"] == 2
    assert pair["matched"]["original_api_requests"] == 2
    assert pair["matched"]["reformulated_api_requests"] == 2
    assert pair["matched"]["original_unique"] == 1


def test_live_rate_failure_remains_unavailable(
    tmp_path, monkeypatch
) -> None:
    """A server delay beyond the cap fails promptly, never as an empty list."""
    monkeypatch.setattr(SemanticScholarClient, "min_interval", 0)
    cases = [
        ClaimCase(
            id="c01",
            domain="science",
            claim="Exercise improves memory.",
        )
    ]
    settings = Settings(openalex_enabled=False, arxiv_enabled=False)
    manifest = asyncio.run(
        run_experiment(
            cases,
            settings,
            ExperimentOptions(),
            tmp_path,
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    429, headers={"Retry-After": "120"}
                )
            ),
        )
    )
    result = write_results(manifest, tmp_path)
    call = result["calls"][0]
    assert call["status"] == "rate_limited"
    assert call["result_count"] is None
    assert (
        result["comparisons"][0]["natural"]["original_zero"] is None
    )
