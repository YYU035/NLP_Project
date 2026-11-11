from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "src/research_verifier/ui/app.py")


def discovery_app():
    app=AppTest.from_file(APP)
    app.session_state['workspace']='Find papers'
    return app.run()


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_recorded_real_discovery_has_no_scientific_verdict_and_resolves_details():
    app = discovery_app()
    assert not app.exception
    button(app, "Load recorded search").click().run()
    assert not app.exception
    assert any(m.label == "Candidates" and m.value == "32" for m in app.metric)
    assert not any(m.value in {"SUPPORTED", "INSUFFICIENT", "CONFLICTING"} for m in app.metric)
    assert any("Not yet verified" in t.value for t in app.text)
    assert any("OpenAlex" in t.value for t in app.text) or app.metric
    assert app.selectbox(key="discovery_inspect").value


def test_filtering_and_paging_do_not_lose_selections():
    app = discovery_app()
    button(app, "Load recorded search").click().run()
    app.checkbox[0].uncheck().run()
    assert any(m.label == "Selected" and m.value == "31" for m in app.metric)
    app.text_input(key="discovery_filter").set_value("not-a-paper-title-xyz").run()
    assert not app.checkbox
    assert any(m.label == "Selected" and m.value == "31" for m in app.metric)
    app.text_input(key="discovery_filter").set_value("").run()
    assert not app.checkbox[0].value
    app.selectbox(key="discovery_page").set_value(2).run()
    assert any(m.label == "Selected" and m.value == "31" for m in app.metric)
    assert len(app.checkbox) <= 10


def test_live_discovery_is_explicit_and_input_edits_clear_previous_result(monkeypatch):
    app = discovery_app()
    app.selectbox(key="discovery_source").set_value("Live search").run()
    assert not app.metric
    assert any("local models" in t.value.lower() for t in app.info)
    app.text_area(key="discovery_claim").set_value(" ").run()
    button(app, "Find papers").click().run()
    assert app.error
    assert not app.metric


def test_navigation_retains_discovery_without_attaching_a_verdict_to_it():
    app = discovery_app()
    button(app, "Load recorded search").click().run()
    app.radio(key="workspace").set_value("Verify evidence").run()
    assert not app.exception
    assert not app.metric
    app.radio(key="workspace").set_value("Find papers").run()
    assert any(m.label == "Candidates" and m.value == "32" for m in app.metric)
    assert app.text_area(key="discovery_claim").value == app.session_state["discovery_snapshot"].result.original_claim
    app.radio(key="workspace").set_value("Evaluate explanations").run()
    app.radio(key="workspace").set_value("Find papers").run()
    assert app.text_area(key="discovery_claim").value == app.session_state["discovery_snapshot"].result.original_claim


def test_navigation_restores_exact_claim_and_source_mode():
    app = discovery_app()
    app.radio(key="workspace").set_value("Verify evidence").run()
    app.selectbox(key="mode").set_value("Live").run()
    app.text_area(key="claim").set_value("Exact edited claim with 12%.").run()
    app.radio(key="workspace").set_value("Find papers").run()
    app.radio(key="workspace").set_value("Verify evidence").run()
    assert app.selectbox(key="mode").value == "Live"
    assert app.text_area(key="claim").value == "Exact edited claim with 12%."


def test_selected_search_handoff_opens_verification_without_running_or_fabricating_results(monkeypatch):
    monkeypatch.setenv("EXPLANATION_VERIFY_DISCOVERY_FACTORY", "")
    app = discovery_app()
    button(app, "Load recorded search").click().run()
    app.checkbox[0].uncheck().run()
    button(app, "Prepare selected papers for verification").click().run()
    assert app.radio(key="workspace").value == "Verify evidence"
    assert not app.exception
    assert len(app.session_state["pending_discovery"]["ids"]) == 31
    assert not app.metric
    button(app, "Verify claim").click().run()
    assert app.error
    assert not app.metric


def test_returning_to_selected_search_restores_its_exact_locked_claim():
    app = discovery_app()
    button(app, "Load recorded search").click().run()
    button(app, "Prepare selected papers for verification").click().run()
    exact = app.session_state["pending_discovery"]["snapshot"].result.original_claim
    app.selectbox(key="mode").set_value("Demo").run()
    app.selectbox(key="mode").set_value("Selected search").run()
    assert app.text_area(key="claim").value == exact
