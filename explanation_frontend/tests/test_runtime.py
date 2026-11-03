import json
import threading
import time

import pytest

from explanation_frontend.export import export_snapshot, import_snapshot
from explanation_frontend.llm import generate_llm, OpenAIClient
from explanation_frontend.models import Explanation, make_context
from explanation_frontend.providers import FixtureProvider, IntegratedProvider
from explanation_frontend.state import Workspace, start_job, poll_job
from explanation_frontend.templates import build_template
from explanation_frontend.validation import safe_url


def ctx(name="supported"):
    p = FixtureProvider()
    s = p.run_fixture(name)
    return make_context(s, list(s.papers), p.aggregate(s, list(s.papers)))


class FakeClient:
    model = "test-client"

    def __init__(self, payload=None, error=None):
        self.payload, self.error = payload, error

    def complete(self, messages):
        if self.error:
            raise self.error
        return self.payload


def valid_payload(context):
    return json.dumps({"verdict": context.aggregation.aggregated_verdict,
                       "statements": [{"text": p.text, "citation_ids": [p.passage_id]}
                                      for p in context.aggregation.primary_evidence], "correction": None})


def test_valid_llm_has_only_cited_quotations_and_recorded_settings():
    c = ctx()
    result = generate_llm(c, FakeClient(valid_payload(c)))
    assert result.status == "accepted"
    assert result.displayed.method == "llm"
    assert result.validation.displayable
    assert result.prompt_version
    assert result.model == "test-client"


@pytest.mark.parametrize("fault", ["fabricated", "irrelevant", "partial", "numbers", "qualifier", "verdict", "correction", "extra", "invalid", "empty"])
def test_bad_llm_is_retained_for_evaluation_but_never_displayed(fault):
    c = ctx("qualified")
    d = json.loads(valid_payload(c))
    if fault == "fabricated":
        d["statements"][0]["citation_ids"] = ["invented"]
    elif fault in {"irrelevant", "partial"}:
        d["statements"][0]["text"] += " Therefore all other systems are safe."
    elif fault == "numbers":
        d["statements"][0]["text"] = d["statements"][0]["text"].replace("12%", "20%")
    elif fault == "qualifier":
        d["statements"][0]["text"] = d["statements"][0]["text"].split(";")[0]
    elif fault == "verdict":
        d["verdict"] = "CONTRADICTED"
    elif fault == "correction":
        d["correction"] = d["statements"][0]
    elif fault == "extra":
        d["summary"] = "Unvalidated prose"
    elif fault == "empty":
        d["statements"] = []
    raw = "not json" if fault == "invalid" else json.dumps(d)
    result = generate_llm(c, FakeClient(raw))
    assert result.status == "rejected"
    assert result.raw == raw
    assert result.displayed.method == "template"
    assert result.displayed == build_template(c)


def test_llm_timeout_redacts_exception_and_context_overflow_skips_call():
    result = generate_llm(ctx(), FakeClient(error=TimeoutError("secret-api-key")))
    assert result.status == "unavailable"
    assert "secret-api-key" not in result.model_dump_json()
    result = generate_llm(ctx("conflict"), FakeClient(error=AssertionError("must not call")), max_context_chars=1)
    assert result.status == "context_limit"
    assert len(result.omitted_passage_ids) == 2


def test_empty_llm_configuration_never_exposes_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EXPLANATION_LLM_MODEL", raising=False)
    with pytest.raises(ValueError, match="Configure"):
        OpenAIClient.from_env()


def test_conflict_side_cannot_be_omitted():
    c = ctx("conflict")
    d = json.loads(valid_payload(c))
    d["statements"] = d["statements"][:1]
    assert generate_llm(c, FakeClient(json.dumps(d))).status == "rejected"


def test_export_replay_preserves_exact_view_and_labels_cached():
    c = ctx("conflict")
    raw = export_snapshot(c, build_template(c))
    bundle = import_snapshot(raw)
    assert bundle.snapshot.claim == c.snapshot.claim
    assert bundle.snapshot.mode == "demo"  # provenance preserved; replay is a separate flag
    assert bundle.replay_mode == "cached"
    assert bundle.selected_paper_ids == ["demo:alpha", "demo:beta"]
    assert bundle.explanation.statements[0].text == c.passages["demo:alpha#p0"].text
    d = json.loads(raw)
    d["snapshot"]["claim"] = "Tampered claim"
    with pytest.raises(ValueError):
        import_snapshot(json.dumps(d))


def test_export_checks_integrity_and_explanation_scope():
    c = ctx()
    e = build_template(c).model_copy(update={"context_key": "wrong"})
    with pytest.raises(ValueError):
        export_snapshot(c, e)
    with pytest.raises(ValueError):
        import_snapshot("x" * 10_000_001)


def test_state_rejects_late_claim_and_selection_responses():
    a, b = Workspace(), Workspace()
    token_a = a.begin("claim A", {"mode": "demo"})
    assert a.begin("claim A", {"mode": "demo"}) == token_a
    token_b = a.begin("claim B", {"mode": "demo"})
    assert not a.accept(token_a, "old result")
    assert a.accept(token_b, "new result")
    assert b.result is None
    a.invalidate()
    assert not a.accept(token_b, "late result")
    assert a.result is None


def test_background_cancel_and_timeout_do_not_publish_late_result():
    gate = threading.Event()
    job = start_job(lambda: gate.wait(0.2), "key", timeout=0.001)
    job.deadline = -1  # deterministic deadline expiry, independent of AppTest's clock
    assert poll_job(job)[0] == "timeout"
    gate.set()
    assert poll_job(job)[0] == "timeout"


@pytest.mark.parametrize("value", ["javascript:alert(1)", "data:text/html,x", "https://user:password@example.org", "https://example.org\n/x", "http://["])
def test_unsafe_source_links_are_not_clickable(value):
    assert safe_url(value) is None


def test_integration_subset_does_not_repeat_expensive_pipeline():
    c = ctx("conflict")
    counts = {"search": 0, "rank": 0, "select": 0, "verify": 0}
    def stage(name, output):
        def call(*args):
            counts[name] += 1
            return output
        return call
    def aggregate(response, ids):
        return FixtureProvider().aggregate(c.snapshot, ids).model_dump()
    live = c.snapshot.model_dump()
    live["mode"] = "live"
    adapter = IntegratedProvider(search=stage("search", []), rank=stage("rank", [p.model_dump() for p in c.snapshot.papers.values()]),
        select=stage("select", {}), verify=stage("verify", c.snapshot.verification.model_dump()),
        aggregate=aggregate, pack=lambda *args: live)
    snapshot = adapter.run(c.snapshot.claim, {})
    assert adapter.aggregate(snapshot, ["demo:alpha"]).aggregated_verdict == "SUPPORTED"
    assert adapter.aggregate(snapshot, []).aggregated_verdict == "INSUFFICIENT"
    assert counts == {"search": 1, "rank": 1, "select": 1, "verify": 1}
