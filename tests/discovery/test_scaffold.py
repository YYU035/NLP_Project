"""Offline checks for the implemented configuration and data boundaries."""

import pytest
from pydantic import ValidationError

from paper_search.config import Settings
from paper_search.metadata import PaperMetadata


def test_settings_load_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment overrides are explicitly loaded and validated."""
    monkeypatch.setenv("PAPER_SEARCH_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("PAPER_SEARCH_MAX_RESULTS", "5")
    monkeypatch.setenv("PAPER_SEARCH_LOG_LEVEL", "DEBUG")
    settings = Settings.from_environment()
    assert settings.timeout_seconds == 12.5
    assert settings.max_results == 5
    assert settings.log_level == "DEBUG"


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_settings_reject_invalid_timeout(timeout: float) -> None:
    """Network timeouts must be positive and finite."""
    with pytest.raises(ValidationError):
        Settings(timeout_seconds=timeout)


def test_metadata_rejects_blank_title() -> None:
    """Whitespace-only titles cannot enter canonical metadata."""
    with pytest.raises(ValidationError):
        PaperMetadata(title="  ", source="example", source_id="1")


