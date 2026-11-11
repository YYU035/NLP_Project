"""Advisory quality checks preserve the scientific inputs and UI selections."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from research_verifier.discovery.service import DiscoverySnapshot, discovery_handoff, sample_discovery


APP = str(Path(__file__).parents[1] / "src/research_verifier/ui/app.py")
EXACT_ABSTRACT = "  pH 7.4 and TNF-α were measured at 25 °C using 1.0 M KOH. \n"


def discovery_source(exact_abstract=EXACT_ABSTRACT):
    """Three source-linked papers, including two descriptions needing review."""
    data = sample_discovery().model_dump(mode="json")
    data["result"]["papers"] = data["result"]["papers"][:3]
    texts = ["We tested the effect of the...", None, exact_abstract]
    origins = set()
    descriptions = {}
    for entry, text in zip(data["result"]["papers"], texts, strict=True):
        entry["paper"]["abstract"] = text
        for source in entry["source_records"]:
            source["abstract"] = text
            descriptions[(source["source"], source["source_id"])] = text
        for origin in entry["provenance"]:
            origins.add((origin["provider"], origin["source_id"],
                         origin["query_id"], origin["rank"]))
    for provider, run in data["result"]["providers"].items():
        run["records"] = [record for record in run["records"] if
                          (provider, record["paper"]["source_id"],
                           record["query_id"], record["rank"]) in origins]
        for record in run["records"]:
            record["paper"]["abstract"] = descriptions[
                (record["paper"]["source"], record["paper"]["source_id"])]
    data["selected_paper_ids"] = [entry["identity"] for entry in data["result"]["papers"]]
    return DiscoverySnapshot.model_validate(data)


def button(app, label):
    return next(widget for widget in app.button if widget.label == label)


def discovery_app(monkeypatch, source):
    import research_verifier.ui.discovery as view
    monkeypatch.setattr(view, "sample_discovery", lambda: source)
    app = AppTest.from_file(APP, default_timeout=15)
    app.session_state["workspace"] = "Find papers"
    app.run()
    button(app, "Load recorded search").click().run()
    assert not app.exception
    return app


def test_quality_report_keeps_ranking_selection_exact_text_and_handoff():
    from research_verifier.discovery.abstract_quality import review_discovery
    from research_verifier.retrieval.ranking import connect_ranking, ranking_request

    # The existing native ranking contract accepts normalized provider text.
    # Whitespace-rich input preservation is exercised separately below.
    source = discovery_source(EXACT_ABSTRACT.strip())
    request = ranking_request(source)
    ids = [entry.identity for entry in source.result.papers]
    request.update(bm25_ranking=ids[::-1], sbert_ranking=ids[::-1],
                   bm25_seconds=0.0, sbert_seconds=0.0)
    data = source.model_dump(mode="json")
    data["ranking_context"] = connect_ranking(json.dumps(request), source.result)
    source = DiscoverySnapshot.model_validate(data)
    before = source.model_dump_json()
    handoff = discovery_handoff(source, ids[:2])

    review = review_discovery(source)
    report = json.loads(review.to_json())
    assert review.to_json() == review_discovery(source).to_json()
    assert source.model_dump_json() == before
    assert discovery_handoff(source, ids[:2]) == handoff
    assert report["advisory_only"] is True
    assert report["source_fingerprint"] == source.fingerprint
    assert report["original_claim"] == source.result.original_claim
    assert report["rule_version"] == "abstract-quality-v1"
    assert set(report["scope_counts"]) == {"provider_record", "deduplicated"}
    assert review.counts().get("suspected_incomplete") == 1
    assert review.counts().get("missing") == 1
    assert review.counts().get("likely_valid") == 1
    selected = {row["identity"]: row for row in report["assessments"]
                if row["scope"] == "deduplicated"}
    for entry in source.result.papers:
        row = selected[entry.identity]
        assert row["assessment"]["text"] == entry.paper.abstract
        assert row["provenance"] == [o.model_dump() for o in entry.provenance]
    assert selected[ids[2]]["assessment"]["text"] == EXACT_ABSTRACT.strip()
    assert review.by_identity[ids[2]].text == EXACT_ABSTRACT.strip()


def test_quality_export_preserves_whitespace_unicode_and_source_hash():
    from research_verifier.discovery.abstract_quality import review_discovery

    source = discovery_source()
    before = source.model_dump_json()
    report = json.loads(review_discovery(source).to_json())
    selected = next(row for row in report["assessments"]
                    if row["scope"] == "deduplicated"
                    and row["identity"] == source.result.papers[2].identity)
    assert selected["assessment"]["text"] == EXACT_ABSTRACT
    assert selected["assessment"]["text_sha256"] == hashlib.sha256(EXACT_ABSTRACT.encode()).hexdigest()
    assert source.model_dump_json() == before


def test_orphan_provider_occurrence_does_not_prevent_candidate_checks():
    from research_verifier.discovery.abstract_quality import review_discovery

    source = discovery_source()
    data = source.model_dump(mode="json")
    provider = next(run for run in data["result"]["providers"].values()
                    if run["records"])
    orphan = copy.deepcopy(provider["records"][0])
    orphan["paper"]["source_id"] += ":unmerged"
    orphan["rank"] += 1000
    provider["records"].append(orphan)
    source = DiscoverySnapshot.model_validate(data)
    before = source.model_dump_json()

    review = review_discovery(source)
    assert review.provider_error
    assert len(review.by_identity) == 3
    assert review.counts().get("suspected_incomplete") == 1
    assert source.model_dump_json() == before
    assert json.loads(review.to_json())["advisory_only"] is True


@pytest.mark.parametrize("remove_all", [True, False], ids=["all-records-omitted", "one-origin-omitted"])
def test_missing_provider_occurrences_are_reported_as_partial_without_losing_candidate_checks(remove_all):
    from research_verifier.discovery.abstract_quality import review_discovery

    data = discovery_source().model_dump(mode="json")
    assert any(entry["provenance"] for entry in data["result"]["papers"])
    if remove_all:
        for provider in data["result"]["providers"].values():
            provider["records"] = []
    else:
        provider = next(run for run in data["result"]["providers"].values()
                        if run["records"])
        provider["records"].pop(0)
    source = DiscoverySnapshot.model_validate(data)
    before = source.model_dump_json()
    available = sum(len(provider.records) for provider in source.result.providers.values())

    review = review_discovery(source)
    report = json.loads(review.to_json())
    assert review.provider_error
    assert report["provider_records_complete"] is False
    assert report["provider_error"] == review.provider_error
    assert len([row for row in review.records if row.scope == "provider_record"]) == available
    assert sum(review.counts("provider_record").values()) == available
    assert len(review.by_identity) == len(source.result.papers) == 3
    assert sum(review.counts().values()) == 3
    assert review.by_identity[source.result.papers[2].identity].text == EXACT_ABSTRACT
    assert source.model_dump_json() == before


def test_discovery_ui_warns_without_filtering_or_rewriting_sources(monkeypatch):
    source = discovery_source()
    before = source.model_dump_json()
    app = discovery_app(monkeypatch, source)
    assert any(heading.value == "Abstract quality checks" for heading in app.subheader)
    assert len(app.checkbox) == 3
    assert any(metric.label == "Selected" and metric.value == "3" for metric in app.metric)
    assert any("Abstract check:" in caption.value for caption in app.caption)
    assert any("incomplete" in item.value.lower() or "cut-off" in item.value.lower()
               for item in app.warning)
    assert any(item.value == source.result.papers[0].paper.abstract for item in app.text)

    app.selectbox(key="discovery_inspect").set_value(source.result.papers[2].identity).run()
    assert not app.exception
    # Streamlit's text renderer trims outer whitespace; the stored source and
    # downloadable report above must retain it byte-for-byte.
    assert any(item.value == EXACT_ABSTRACT.strip() for item in app.text)
    assert app.session_state["discovery_snapshot"].result.papers[2].paper.abstract == EXACT_ABSTRACT
    assert any("No warning detected" in caption.value for caption in app.caption)
    assert not app.success  # Passing the heuristics is never a correctness badge.
    assert source.model_dump_json() == before


def test_discovery_flags_do_not_change_selection_filtering_or_verification_handoff(monkeypatch):
    source = discovery_source()
    app = discovery_app(monkeypatch, source)
    app.checkbox[0].uncheck().run()
    app.text_input(key="discovery_filter").set_value("zzzz-no-quality-test-match").run()
    assert not app.checkbox
    assert any(metric.label == "Selected" and metric.value == "2" for metric in app.metric)
    app.text_input(key="discovery_filter").set_value("").run()
    assert not app.exception
    assert not app.checkbox[0].value
    assert app.checkbox[1].value  # Missing text is advisory, not auto-excluded.
    button(app, "Prepare selected papers for verification").click().run()
    assert not app.exception
    pending = app.session_state["pending_discovery"]
    assert pending["ids"] == source.selected_paper_ids[1:]
    assert pending["snapshot"].fingerprint == source.fingerprint
    assert app.text_area(key="claim").value == source.result.original_claim


def test_empty_discovery_has_valid_quality_report_and_usable_ui(monkeypatch):
    from research_verifier.discovery.abstract_quality import review_discovery

    data = discovery_source().model_dump(mode="json")
    data["result"]["papers"] = []
    data["selected_paper_ids"] = []
    for run in data["result"]["providers"].values():
        run["records"] = []
        run["status"] = "zero_results"
    source = DiscoverySnapshot.model_validate(data)
    review = review_discovery(source)
    assert not review.records
    assert not review.by_identity
    assert json.loads(review.to_json())["assessments"] == []
    app = discovery_app(monkeypatch, source)
    assert not app.exception
    assert any(metric.label == "Candidates" and metric.value == "0" for metric in app.metric)
    assert button(app, "Prepare selected papers for verification").disabled
    assert not app.checkbox


def _source_backed_run():
    from research_verifier.discovery.service import import_discovery
    from research_verifier.contracts.models import RunSnapshot
    from research_verifier.verification.providers import FixtureProvider

    run = FixtureProvider().run_fixture("conflict")
    papers, records = [], []
    for rank, (identity, paper) in enumerate(run.papers.items(), 1):
        metadata = {"source": "openalex", "source_id": identity,
                    "title": paper.title, "abstract": "We tested the effect of the..."}
        papers.append({"identity": identity, "identity_aliases": [identity],
                       "original_claim": run.claim, "paper": metadata,
                       "provenance": [{"provider": "openalex", "source_id": identity,
                                       "query_id": "q1", "rank": rank}],
                       "source_records": [metadata]})
        records.append({"paper": metadata, "query_id": "q1", "rank": rank})
    source = import_discovery(json.dumps({
        "original_claim": run.claim, "top_k": 2, "papers": papers,
        "providers": {"openalex": {"provider": "openalex", "status": "success",
            "queries": [{"id": "q1", "provider": "openalex", "strategy": "saved",
                         "stage": "primary", "query": run.claim, "status": "success"}],
            "records": records}},
    }))
    data = run.model_dump(mode="json")
    data.update(mode="cached", discovery_context=source.model_dump(mode="json"))
    return RunSnapshot.model_validate(data)


def test_cached_verification_displays_source_flags_without_recomputing_predictions(monkeypatch):
    import research_verifier.pipeline as pipeline

    def no_inference():
        raise AssertionError("Rendering a quality flag must not load a model")
    monkeypatch.setattr(pipeline, "local_verifier", no_inference)
    monkeypatch.setattr(pipeline, "local_encoder", no_inference)
    run = _source_backed_run()
    before = run.model_dump_json()
    app = AppTest.from_file(APP, default_timeout=15)
    app.session_state["workspace"] = "Verify evidence"
    app.session_state["mode"] = "Cached replay"
    app.session_state["claim"] = run.claim
    app.session_state["snapshot"] = run
    app.run()
    assert not app.exception
    assert any("Abstract check:" in caption.value for caption in app.caption)
    assert any("incomplete" in item.value.lower() or "cut-off" in item.value.lower()
               for item in app.warning)
    assert any(metric.label == "Selected-evidence verdict" and metric.value == "CONFLICTING"
               for metric in app.metric)
    assert app.session_state["snapshot"].model_dump_json() == before


def test_demo_does_not_classify_evidence_fragments_as_provider_abstracts():
    app = AppTest.from_file(APP, default_timeout=15)
    app.session_state["workspace"] = "Verify evidence"
    app.run()
    button(app, "Verify claim").click().run()
    assert not app.exception
    assert app.metric
    assert not any("Abstract check:" in caption.value for caption in app.caption)
