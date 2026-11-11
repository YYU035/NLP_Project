"""Offline regressions for advisory abstract-quality assessment."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from paper_search.abstract_quality import assess_abstract, assess_retrieval
from paper_search.retrieval_models import RetrievalResult

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "artifacts" / "depth-approved-run"


@pytest.mark.parametrize("text", [None, "", " \n\t"])
def test_missing_preserves_exact_input(text):
    result = assess_abstract(text)
    assert result.status == "missing"
    assert result.text == text
    assert result.character_count == len(text or "")


@pytest.mark.parametrize(
    "text",
    [
        "A short review of lithium-ion battery recycling technologies.",
        "pH 7.4 and TNF-α were measured at 25 °C using 1.0 M KOH.",
        "Background: We tested cells. Methods: Samples were compared. "
        "Results: ATP increased. Conclusions: Treatment altered ATP.",
        "Funding and ethics approval were discussed in this brief review.",
    ],
)
def test_short_scientific_or_structured_text_is_not_a_fragment(text):
    assert assess_abstract(text).status == "likely_valid"


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "and 1.0 M KOH electrolytes, respectively. "
            "Moreover, ATP increased.",
            "suspected_incomplete",
        ),
        ("We tested the effect of the...", "suspected_incomplete"),
        ("Samples were collected...", "uncertain"),
        ("No abstract available", "uncertain"),
        (
            "Abstract: English summary. Resumen: Resumen traducido.",
            "uncertain",
        ),
        ("A description without a sentence boundary", "uncertain"),
        ("A broken \ufffd description.", "uncertain"),
        ("word " * 1001, "body_like"),
        ("word " * 301 + "Funding. Ethics approval.", "body_like"),
    ],
)
def test_warning_rules(text, expected):
    assessment = assess_abstract(text)
    assert assessment.status == expected
    assert assessment.text == text
    assert (
        assessment.text_sha256
        == hashlib.sha256(text.encode("utf-8")).hexdigest()
    )


@pytest.mark.parametrize(
    "claim, source_id, status, reason",
    [
        (
            "c09",
            "https://openalex.org/W3048912808",
            "suspected_incomplete",
            "continuation_with_unresolved_reference",
        ),
        (
            "c01",
            "https://openalex.org/W3033478000",
            "body_like",
            "excessive_length",
        ),
        (
            "c01",
            "doi_________::e540accc2df9a7ff03f1f4e7a1518fb8",
            "body_like",
            "excessive_length",
        ),
        (
            "c04",
            "doi_dedup___::e378a51af7a356283f358a447cdf106f",
            "body_like",
            "excessive_length",
        ),
    ],
)
def test_saved_problem_records(claim, source_id, status, reason):
    data = json.loads((SAVED / f"{claim}.json").read_text(encoding="utf-8"))
    papers = [
        record["paper"]
        for run in data["providers"].values()
        for record in run["records"]
    ]
    paper = next(p for p in papers if p["source_id"] == source_id)
    assessment = assess_abstract(paper["abstract"])
    assert assessment.status == status
    assert reason in assessment.reason_codes
    assert assessment.text == paper["abstract"]


@pytest.mark.parametrize(
    "source_id",
    [
        "https://openalex.org/W4406382476",
        "https://openalex.org/W3159198112",
    ],
)
def test_saved_short_descriptions_not_incomplete(source_id):
    data = json.loads((SAVED / "c02.json").read_text(encoding="utf-8"))
    paper = next(
        r["paper"]
        for r in data["providers"]["openalex"]["records"]
        if r["paper"]["source_id"] == source_id
    )
    assert assess_abstract(paper["abstract"]).status == "likely_valid"


def test_assessment_preserves_schema_metadata_and_all_provenance():
    result = RetrievalResult.model_validate_json(
        (SAVED / "c01.json").read_bytes()
    )
    before = result.model_dump_json()
    rows = assess_retrieval(result, claim_id="c01")
    assert result.model_dump_json() == before
    provider_rows = [r for r in rows if r.scope == "provider_record"]
    merged_rows = [r for r in rows if r.scope == "deduplicated"]
    assert len(provider_rows) == sum(
        len(run.records) for run in result.providers.values()
    )
    assert len(merged_rows) == len(result.papers)
    for row, paper in zip(merged_rows, result.papers, strict=True):
        assert row.identity == paper.identity
        assert row.identity_aliases == paper.identity_aliases
        assert row.provenance == [p.model_dump() for p in paper.provenance]
        assert row.assessment.text == paper.paper.abstract
        assert row.source_id == paper.paper.source_id
        assert row.original_claim == result.original_claim
    for row in provider_rows:
        origin = row.provenance[0]
        record = next(
            r
            for r in result.providers[row.provider].records
            if r.query_id == origin["query_id"] and r.rank == origin["rank"]
        )
        assert row.assessment.text == record.paper.abstract
        assert row.source_id == record.paper.source_id


def test_provider_and_selected_abstract_are_assessed_independently():
    result = RetrievalResult.model_validate_json(
        (SAVED / "c01.json").read_bytes()
    )
    record = result.providers["openalex"].records[0]
    record.paper.abstract = None
    rows = assess_retrieval(result, claim_id="c01")
    source = next(
        r for r in rows if r.scope == "provider_record"
        and r.provider == "openalex"
        and r.provenance[0]["rank"] == record.rank
        and r.provenance[0]["query_id"] == record.query_id
    )
    merged = next(
        r for r in rows if r.scope == "deduplicated"
        and r.identity == source.identity
    )
    assert source.assessment.status == "missing"
    assert merged.assessment.status == "likely_valid"
    assert record.paper.abstract is None
    assert merged.assessment.text is not None


def test_offline_runner_outputs_and_refuses_overwrite(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "assess_saved", ROOT / "scripts" / "assess_abstract_quality.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    payload = (SAVED / "c01.json").read_bytes()
    (inputs / "c01.json").write_bytes(payload)
    output = tmp_path / "output"
    report = module.evaluate(inputs, output, [10, 30])
    assert {p.name for p in output.iterdir()} == {
        "assessment.csv",
        "assessment.json",
        "report.md",
    }
    assert (inputs / "c01.json").read_bytes() == payload
    assert len(report["inputs"]) == 1
    assert {row["scope"] for row in report["assessments"]} == {
        "provider_record",
        "deduplicated",
    }
    assert {row["top_k"] for row in report["assessments"]} == {10, 30}
    with pytest.raises(FileExistsError):
        module.evaluate(inputs, output, [10])
    with pytest.raises(ValueError, match="separate"):
        module.evaluate(inputs, inputs / "output", [10])
