"""Offline checks of setup, portable model paths and the public handoff."""

import asyncio
import importlib.util
import json
import shutil
import subprocess
from unittest.mock import Mock

import httpx
import pytest

from paper_search.keybert_queries import DEFAULT_MODEL_PATH, MODEL_REVISION
from paper_search.model_setup import download_model
from paper_search.query_rendering import RenderedQuery
from paper_search.retrieval import RetrievalService, retrieve


def test_default_model_path_does_not_depend_on_caller_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    service = RetrievalService()
    assert service.generator.model_path == DEFAULT_MODEL_PATH
    assert DEFAULT_MODEL_PATH.is_absolute()


def test_model_setup_reuses_pinned_download_without_network(
    tmp_path, monkeypatch
):
    snapshot = Mock()
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    monkeypatch.setenv("HF_HOME", "test-cache")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    download_model(tmp_path)
    assert snapshot.call_args.kwargs["revision"] == MODEL_REVISION
    assert snapshot.call_args.kwargs["local_dir"] == str(tmp_path)
    assert (tmp_path / "pinned_revision.txt").read_text() == MODEL_REVISION


def test_public_retrieve_uses_final_policy_and_real_parsers_offline(
    monkeypatch,
):
    """Exercise the public wrapper with mocked HTTP and real client parsers."""
    requests = []
    doi = "10.1234/test"

    def handler(request):
        requests.append(request)
        if request.url.host == "api.openalex.org":
            assert request.url.params["per_page"] == "30"
            assert (
                request.url.params["search"] == "solar cells power conversion"
            )
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "https://openalex.org/W1",
                            "doi": doi,
                            "title": "Solar cells and power conversion",
                            "publication_date": "2024-01-01",
                            "abstract_inverted_index": {
                                "Test": [0],
                                "abstract": [1],
                            },
                            "primary_location": {
                                "pdf_url": "https://example.org/paper.pdf"
                            },
                        }
                    ]
                },
            )
        if request.url.host == "api.openaire.eu":
            assert request.url.params["pageSize"] == "30"
            assert (
                request.url.params["search"] == "solar cells power conversion"
            )
            return httpx.Response(
                200,
                json={
                    "header": {"numFound": 1},
                    "results": [
                        {
                            "id": "openaire-one",
                            "mainTitle": "Solar cells and power conversion",
                            "pids": [{"scheme": "doi", "value": doi}],
                            "descriptions": ["Test abstract"],
                        }
                    ],
                },
            )
        assert request.url.host == "export.arxiv.org"
        assert request.url.params["max_results"] == "30"
        assert request.url.params["search_query"] == (
            'all:"solar cells" AND all:"power conversion"'
        )
        return httpx.Response(
            200,
            text="""
          <feed xmlns="http://www.w3.org/2005/Atom"
                xmlns:arxiv="http://arxiv.org/schemas/atom">
            <entry><id>https://arxiv.org/abs/2401.00001</id>
              <title>Solar cells and power conversion</title>
              <summary>Test abstract</summary>
              <arxiv:doi>10.1234/test</arxiv:doi>
            </entry>
          </feed>""",
        )

    original_http = httpx.AsyncClient

    def offline_http(**kwargs):
        return original_http(
            **{**kwargs, "transport": httpx.MockTransport(handler)}
        )

    monkeypatch.setattr(
        "paper_search.retrieval.httpx.AsyncClient", offline_http
    )
    monkeypatch.setattr(
        "paper_search.keybert_queries.KeyBERTQueryGenerator.generate",
        lambda self, claim, **_: [claim, "solar cells power conversion"],
    )
    monkeypatch.setattr(
        "paper_search.retrieval.render_query",
        lambda q, _: RenderedQuery(
            original=q,
            text=q,
            selected_concepts=["solar cells", "power conversion"],
        ),
    )
    monkeypatch.setenv("PAPER_SEARCH_OPENALEX_ENABLED", "true")
    monkeypatch.setenv("PAPER_SEARCH_OPENAIRE_ENABLED", "true")
    monkeypatch.setenv("PAPER_SEARCH_ARXIV_ENABLED", "true")
    result = asyncio.run(retrieve("Original scientific claim"))
    assert (
        len(requests) == result.http_attempts == 3
    )  # no fallback / Semantic Scholar
    assert len(result.papers) == 1
    entry = result.papers[0]
    assert result.top_k == 30
    assert (
        result.original_claim
        == entry.original_claim
        == "Original scientific claim"
    )
    assert entry.paper.title and entry.paper.abstract and entry.paper.pdf_urls
    assert entry.identity == "doi:10.1234/test"
    assert {p.provider for p in entry.provenance} == {
        "openalex",
        "openaire",
        "arxiv",
    }
    assert len(entry.source_records) == 3
    assert {
        name: run.queries[0].strategy for name, run in result.providers.items()
    } == {"openalex": "keybert", "openaire": "keybert", "arxiv": "spacy"}


def test_person2_example_with_public_result_schema(monkeypatch, capsys):
    from paper_search.retrieval_models import RetrievalResult

    spec = importlib.util.spec_from_file_location(
        "handoff_example", "examples/person2_integration.py"
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)

    async def offline_retrieve(claim, top_k):
        return RetrievalResult(original_claim=claim, top_k=top_k)

    monkeypatch.setattr(example, "retrieve", offline_retrieve)
    asyncio.run(example.main("Original claim"))
    assert (
        json.loads(capsys.readouterr().out)["original_claim"]
        == "Original claim"
    )


def test_gitignore_preserves_research_and_excludes_private_runtime(tmp_path):
    git = shutil.which("git")
    if not git:
        pytest.skip("Git is needed to verify actual ignore semantics")
    shutil.copyfile(".gitignore", tmp_path / ".gitignore")
    subprocess.run(
        [git, "init", "--quiet", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    ignored = [
        ".env",
        ".env.local",
        ".venv/lib/a.py",
        "outputs/result.json",
        "artifacts/local-models/model/pytorch_model.bin",
        "artifacts/gemini-relevance-20261031/cache.sqlite3",
    ]
    retained = [
        ".env.example",
        "artifacts/readiness-live-results-20261101/report.md",
        "artifacts/arxiv-concepts-20261031/"
        "manual_assessment_provisional_250.csv",
        "artifacts/depth-approved-run/responses/example.body",
    ]
    completed = subprocess.run(
        [
            git,
            "-C",
            str(tmp_path),
            "check-ignore",
            "--no-index",
            "--stdin",
            "-z",
        ],
        input=("\0".join(ignored + retained) + "\0").encode(),
        capture_output=True,
        check=True,
    )
    matched = completed.stdout.decode().strip("\0").split("\0")
    assert set(matched) == set(ignored)
