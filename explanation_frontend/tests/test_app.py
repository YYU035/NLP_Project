from pathlib import Path
import threading

from streamlit.testing.v1 import AppTest
from explanation_frontend.providers import FixtureProvider

APP = str(Path(__file__).parents[1] / "app.py")


def button(at, label):
    return next(b for b in at.button if b.label == label)


def test_demo_conflict_subset_and_changed_claim():
    at = AppTest.from_file(APP).run()
    assert not at.exception
    button(at, "Verify claim").click().run()
    assert not at.exception
    assert any(m.value == "CONFLICTING" for m in at.metric)
    assert len(at.checkbox) == 2
    at.checkbox[1].uncheck().run()
    assert not at.exception
    assert any(m.value == "SUPPORTED" for m in at.metric)
    at.checkbox[0].uncheck().run()
    assert any(m.value == "INSUFFICIENT" for m in at.metric)
    at.text_area[0].set_value("A different claim").run()
    assert not at.metric
    button(at, "Verify claim").click().run()
    assert at.error
    assert not at.metric


def test_total_failure_shows_no_scientific_verdict():
    at = AppTest.from_file(APP).run()
    at.selectbox(key="scenario").set_value("failure").run()
    button(at, "Verify claim").click().run()
    assert not at.exception
    assert any(m.value == "PROCESSING FAILURE" for m in at.metric)
    assert not any(m.value == "INSUFFICIENT" for m in at.metric)


def test_sessions_are_isolated_and_live_does_not_silently_demo(monkeypatch):
    monkeypatch.delenv("EXPLANATION_PIPELINE_FACTORY", raising=False)
    a, b = AppTest.from_file(APP).run(), AppTest.from_file(APP).run()
    button(a, "Verify claim").click().run()
    assert a.metric and not b.metric
    b.selectbox(key="mode").set_value("Live").run()
    assert not b.metric
    button(b, "Verify claim").click().run()
    assert b.error
    assert not b.metric


def test_untrusted_passage_is_text_and_source_status_remains_visible():
    at = AppTest.from_file(APP).run()
    at.selectbox(key="scenario").set_value("untrusted").run()
    button(at, "Verify claim").click().run()
    assert not at.exception
    assert any("<script>" in t.value for t in at.text)
    assert any("source status: synthetic" in t.value for t in at.text)


def test_live_subset_is_background_cached_and_cancellable():
    at = AppTest.from_file(APP).run()
    at.selectbox(key="mode").set_value("Live").run()
    snapshot = FixtureProvider().run_fixture("conflict").model_copy(update={"mode": "live"})
    gate = threading.Event()
    calls = []
    class SlowAggregator:
        def aggregate(self, snap, ids):
            calls.append(list(ids))
            gate.wait(1)
            return FixtureProvider().aggregate(snap, ids)
    at.session_state["snapshot"] = snapshot
    at.session_state["provider"] = SlowAggregator()
    for pid in snapshot.papers:
        at.session_state[f"paper:{snapshot.fingerprint}:{pid}"] = True
    at.run()
    assert calls == []  # all-paper provisional aggregation is reused
    at.checkbox[1].uncheck().run()
    assert calls == [["demo:alpha"]]
    assert "aggregation_job" in at.session_state
    assert not any(m.label == "Selected-evidence verdict" for m in at.metric)
    gate.set()
    for _ in range(5):
        at.run()
        if any(m.value == "SUPPORTED" for m in at.metric):
            break
    assert any(m.value == "SUPPORTED" for m in at.metric)
    at.run()
    assert calls == [["demo:alpha"]]
