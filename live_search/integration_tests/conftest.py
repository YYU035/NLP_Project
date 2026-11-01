"""Explicit opt-in controls and safe JSON reporting for live smoke tests."""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Keep live access opt-in even when this directory is selected."""
    parser.addoption("--run-live", action="store_true", default=False)
    parser.addoption("--run-evaluation", action="store_true", default=False)
    parser.addoption("--evaluation-dir", default="artifacts/live-evaluation")
    parser.addoption("--evaluation-from-date", default="2020-01-01")
    parser.addoption("--evaluation-to-date", default="2026-10-30")
    parser.addoption(
        "--live-claim", default="Exercise improves memory",
        help="One scientific claim sent to all three search providers",
    )
    parser.addoption("--live-limit", type=int, default=5)
    parser.addoption(
        "--live-report", default="artifacts/live-search-report.json"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Prevent accidental external calls during collection or normal tests."""
    if not config.getoption("--run-live"):
        for item in items:
            if "live" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Requires --run-live"))


@pytest.fixture(scope="session")
def live_report(pytestconfig: pytest.Config) -> Iterator[dict[str, Any]]:
    """Write observations even when a provider fails or an assertion fails."""
    report: dict[str, Any] = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "claim": pytestconfig.getoption("--live-claim"),
        "limit_per_provider": pytestconfig.getoption("--live-limit"),
        "providers": {},
    }
    yield report
    report["finished_at_utc"] = datetime.now(UTC).isoformat()
    path = Path(pytestconfig.getoption("--live-report"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nLive smoke report: {path.resolve()}")
