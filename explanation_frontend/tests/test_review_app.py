from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "app.py")


def review_app():
    app = AppTest.from_file(APP).run()
    app.radio(key="workspace").set_value("Evaluate explanations").run()
    return app


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_review_workspace_keeps_missing_human_scores_pending():
    app = review_app()
    button(app, "Load synthetic review set").click().run()
    assert not app.exception
    assert any(m.label == "Saved human reviews" and m.value == "0" for m in app.metric)
    button(app, "Save human review").click().run()
    assert app.error
    assert any(m.label == "Saved human reviews" and m.value == "0" for m in app.metric)


def test_reviewer_explicitly_scores_raw_output_against_supplied_passages():
    app = review_app()
    button(app, "Load synthetic review set").click().run()
    app.text_input(key="reviewer_id").set_value("r1").run()
    app.checkbox(key="review_confirm").check()
    app.selectbox(key="review_usefulness").set_value(4)
    button(app, "Save human review").click().run()
    assert not app.exception
    assert any(m.label == "Saved human reviews" and m.value == "1" for m in app.metric)
    assert any("Saved" in s.value for s in app.success)
