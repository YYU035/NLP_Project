"""Integration checks for the consolidated source and resource layout."""
from pathlib import Path

import paper_search
import research_verifier

from research_verifier.runtime.paths import PROJECT_ROOT


def test_packages_resolve_to_this_checkout():
    # A previously installed discovery wheel must not mask a missing bundled module.
    for package in (paper_search, research_verifier):
        assert Path(package.__file__).resolve().is_relative_to((PROJECT_ROOT / "src").resolve())


def test_readonly_resources_work_from_an_unrelated_working_directory(tmp_path, monkeypatch):
    from research_verifier.discovery.service import sample_discovery
    from research_verifier.ui.benchmarks import load_results
    from research_verifier.ui.overview import load_showcase
    from research_verifier.verification.providers import FixtureProvider

    monkeypatch.chdir(tmp_path)
    assert sample_discovery().result.papers
    assert FixtureProvider().run_fixture("supported").provisional.aggregated_verdict == "SUPPORTED"
    assert load_showcase("scifact-230.json").snapshot.verification.model_settings.revision
    assert {item["metadata"]["id"] for item in load_results()} == {"scifact", "csfcube", "reciper"}
    assert not list(tmp_path.iterdir())  # Inspection creates no local output or cache.
