"""Application-owned logging configuration."""

import logging

from paper_search.config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure logging when explicitly called by the application."""
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
