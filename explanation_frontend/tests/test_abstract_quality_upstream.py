"""Behavior and empty-report regressions for the isolated advisory helper."""
import csv
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from explanation_frontend._upstream.discovery_quality.abstract_quality import (
    AbstractQualityAssessment, AbstractQualityRecord, assess_abstract,
    assess_retrieval,
)
from explanation_frontend.discovery import sample_discovery


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "fixtures" / "discovery" / "raghavi-c01.json"


@pytest.mark.parametrize("text", [None, "", " \n\t"])
def test_missing_text_retains_exact_value_and_hash(text):
    result = assess_abstract(text)
    assert result.status == "missing"
    assert result.text == text
    assert result.character_count == len(text or "")
    assert result.text_sha256 == (
        hashlib.sha256(text.encode()).hexdigest() if text is not None else None
    )


@pytest.mark.parametrize("text", [
    "A short review of lithium-ion battery recycling technologies.",
    "pH 7.4 and TNF-α were measured at 25 °C using 1.0 M KOH.",
    "Background: Cells tested. Methods: Samples compared. Results: ATP rose.",
    "Funding and ethics approval were discussed in this brief review.",
])
def test_short_scientific_text_does_not_receive_incomplete_flag(text):
    assert assess_abstract(text).status == "likely_valid"


@pytest.mark.parametrize("text,status", [
    ("and 1.0 M KOH electrolytes, respectively. ATP increased.",
     "suspected_incomplete"),
    ("We tested the effect of the...", "suspected_incomplete"),
    ("Samples were collected...", "uncertain"),
    ("No abstract available", "uncertain"),
    ("Abstract: English summary. Resumen: Resumen traducido.", "uncertain"),
    ("A description without a sentence boundary", "uncertain"),
    ("A broken \ufffd description.", "uncertain"),
    ("word " * 1001, "body_like"),
    ("word " * 301 + "Funding. Ethics approval.", "body_like"),
])
def test_warning_categories_preserve_source_text(text, status):
    result = assess_abstract(text)
    assert result.status == status
    assert result.text == text
    assert result.text_sha256 == hashlib.sha256(text.encode()).hexdigest()


def test_frontend_records_keep_identity_provenance_and_serialization():
    snapshot = sample_discovery()
    before = snapshot.model_dump_json()
    rows = assess_retrieval(snapshot.result, claim_id="recorded-search")
    assert snapshot.model_dump_json() == before
    providers = [row for row in rows if row.scope == "provider_record"]
    merged = [row for row in rows if row.scope == "deduplicated"]
    assert len(providers) == sum(
        len(run.records) for run in snapshot.result.providers.values()
    )
    assert len(merged) == len(snapshot.result.papers)
    for row, paper in zip(merged, snapshot.result.papers, strict=True):
        assert row.identity == paper.identity
        assert row.identity_aliases == paper.identity_aliases
        assert row.provenance == [origin.model_dump() for origin in paper.provenance]
        assert row.assessment.text == paper.paper.abstract
        assert row.original_claim == snapshot.result.original_claim


def test_assessment_import_does_not_require_live_search_package():
    # The normal UI must remain usable without the optional live-search package.
    code = """
import importlib.abc
import sys
class BlockLiveSearch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'paper_search' or fullname.startswith('paper_search.'):
            raise AssertionError('Unexpected optional live-search import')
sys.meta_path.insert(0, BlockLiveSearch())
from explanation_frontend._upstream.discovery_quality.abstract_quality import assess_abstract
assert assess_abstract('A complete scientific summary.').status == 'likely_valid'
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT.parent,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def _runner():
    pytest.importorskip("paper_search")
    return importlib.import_module(
        "explanation_frontend._upstream.discovery_quality.assess_saved"
    )


def test_zero_result_batch_emits_complete_reports_and_stable_csv_header(tmp_path):
    runner = _runner()
    from paper_search.retrieval_models import RetrievalResult, ProviderRun
    result = RetrievalResult(
        original_claim="No papers returned for this diagnostic claim.",
        top_k=30,
        providers={"openalex": ProviderRun(
            provider="openalex", status="zero_results"
        )},
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    payload = result.model_dump_json().encode()
    (inputs / "c01.json").write_bytes(payload)
    output = tmp_path / "review"
    report = runner.evaluate(inputs, output, [10, 30])
    assert report["assessments"] == []
    assert len(report["counts"]) == 4
    assert all(row["total"] == 0 for row in report["counts"])
    assert {path.name for path in output.iterdir()} == {
        "assessment.json", "assessment.csv", "report.md"
    }
    with (output / "assessment.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "top_k",
            *[name for name in AbstractQualityRecord.model_fields
              if name != "assessment"],
            *AbstractQualityAssessment.model_fields,
        ]
        assert list(reader) == []
    assert (inputs / "c01.json").read_bytes() == payload
    with pytest.raises(FileExistsError):
        runner.evaluate(inputs, output, [10, 30])


def test_nonempty_batch_retains_saved_text_and_refuses_input_overlap(tmp_path):
    runner = _runner()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    payload = FIXTURE.read_bytes()
    (inputs / "c01.json").write_bytes(payload)
    report = runner.evaluate(inputs, tmp_path / "review", [30])
    assert report["assessments"]
    assert report["inputs"] == [{
        "file": "c01.json", "sha256": hashlib.sha256(payload).hexdigest()
    }]
    assert {row["scope"] for row in report["assessments"]} == {
        "provider_record", "deduplicated"
    }
    assert (inputs / "c01.json").read_bytes() == payload
    with pytest.raises(ValueError, match="separate"):
        runner.evaluate(inputs, inputs / "review", [30])


def test_source_manifest_records_exact_adapted_module_hashes():
    package = ROOT / "_upstream" / "discovery_quality"
    source = json.loads((package / "SOURCE.json").read_text())
    assert source["revision"] == "0ee95651cdd0c9d40a0ffb09a6bba896db3be455"
    for item in source["sources"]:
        assert hashlib.sha256((package / item["path"]).read_bytes()).hexdigest() == item["adapted_sha256"]
