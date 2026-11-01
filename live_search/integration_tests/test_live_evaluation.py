"""Ten-claim validation/evaluation, separately opt-in from one-claim smoke."""

import asyncio
from datetime import date
from pathlib import Path

import pytest

from paper_search.config import Settings
from paper_search.evaluation.dataset import CLAIMS
from paper_search.evaluation.models import ExperimentOptions
from paper_search.evaluation.reporting import write_results
from paper_search.evaluation.runner import run_experiment

pytestmark = pytest.mark.live


@pytest.fixture(scope="session")
def experiment_results(pytestconfig: pytest.Config) -> dict:
    """Run once for all provider assertions and preserve partial failures."""
    if not pytestconfig.getoption("--run-evaluation"):
        pytest.skip(
            "Also requires --run-evaluation (many real API calls)"
        )
    options = ExperimentOptions(
        max_results=pytestconfig.getoption("--live-limit"),
        from_publication_date=date.fromisoformat(
            pytestconfig.getoption("--evaluation-from-date")
        ),
        to_publication_date=date.fromisoformat(
            pytestconfig.getoption("--evaluation-to-date")
        ),
    )
    output = Path(pytestconfig.getoption("--evaluation-dir"))
    manifest = asyncio.run(
        run_experiment(
            CLAIMS, Settings.from_environment(), options, output
        )
    )
    return write_results(manifest, output)


@pytest.mark.parametrize(
    "provider", ["openalex", "arxiv", "semantic_scholar"]
)
def test_ten_live_claims(
    provider: str, experiment_results: dict
) -> None:
    """Provider errors fail validation; valid zero results pass distinctly."""
    if provider not in experiment_results["providers"]:
        pytest.skip("Provider disabled through configuration")
    originals = [
        call
        for call in experiment_results["calls"]
        if call["provider"] == provider
        and call["kind"] == "original"
    ]
    assert len(originals) == 10
    failures = [
        call["id"] + ": " + call["status"]
        for call in originals
        if call["status"] not in {"success", "zero_results"}
    ]
    assert not failures, "; ".join(failures)
    assert all(
        call["metadata"]["date_filter_violations"] == 0
        for call in originals
    )
