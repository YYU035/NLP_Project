"""Validated runtime settings with explicit environment loading."""

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class Settings(BaseModel):
    """Shared settings; constructing this model performs no I/O."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    max_results: int = Field(default=20, ge=1, le=100)
    max_queries: int = Field(default=2, ge=1)
    query_mode: Literal["original_only", "original_plus_concepts"] = (
        "original_plus_concepts"
    )
    query_model: str = Field(default="en_core_web_sm", min_length=1)
    openalex_enabled: bool = True
    arxiv_enabled: bool = True
    semantic_scholar_enabled: bool = True
    openaire_enabled: bool = False
    openalex_api_key: SecretStr | None = None
    semantic_scholar_api_key: SecretStr | None = None
    openaire_token: SecretStr | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        "INFO"
    )

    @classmethod
    def from_environment(cls) -> "Settings":
        """Load supported environment values, raising on invalid settings."""
        values = {
            field: os.environ[variable]
            for field, variable in {
                "timeout_seconds": "PAPER_SEARCH_TIMEOUT_SECONDS",
                "max_results": "PAPER_SEARCH_MAX_RESULTS",
                "max_queries": "PAPER_SEARCH_MAX_QUERIES",
                "query_mode": "PAPER_SEARCH_QUERY_MODE",
                "query_model": "PAPER_SEARCH_QUERY_MODEL",
                "log_level": "PAPER_SEARCH_LOG_LEVEL",
                "openalex_enabled": "PAPER_SEARCH_OPENALEX_ENABLED",
                "arxiv_enabled": "PAPER_SEARCH_ARXIV_ENABLED",
                "semantic_scholar_enabled": (
                    "PAPER_SEARCH_SEMANTIC_SCHOLAR_ENABLED"
                ),
                "openaire_enabled": "PAPER_SEARCH_OPENAIRE_ENABLED",
                "openaire_token": "OPENAIRE_TOKEN",
                "openalex_api_key": "OPENALEX_API_KEY",
                "semantic_scholar_api_key": "SEMANTIC_SCHOLAR_API_KEY",
            }.items()
            if variable in os.environ
        }
        return cls.model_validate(values)
