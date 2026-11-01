"""Small shared helpers for nullable provider metadata."""

from datetime import date
from typing import Any

from pydantic import HttpUrl, TypeAdapter, ValidationError

_URL = TypeAdapter(HttpUrl)


def text(value: object) -> str | None:
    """Return a nonempty string if present."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def mapping(value: object) -> dict[str, Any]:
    """Treat null or malformed nested objects as absent."""
    return value if isinstance(value, dict) else {}


def url(value: object) -> str | None:
    """Keep only usable HTTP(S) links from provider metadata."""
    try:
        return str(_URL.validate_python(value))
    except ValidationError:
        return None


def publication_date(value: object) -> date | None:
    """Parse a provider ISO date, omitting malformed values."""
    try:
        return date.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None


def publication_year(value: object, published: date | None) -> int | None:
    """Prefer a valid year, otherwise derive it from a known date."""
    if type(value) is int and value >= 1:
        return value
    return published.year if published else None


def doi(value: object) -> str | None:
    """Normalize raw or resolver-prefixed DOI strings to HTTPS DOI URLs."""
    result = text(value)
    if result is None:
        return None
    for prefix in ("https://doi.org/", "http://doi.org/",
                   "https://dx.doi.org/", "http://dx.doi.org/", "doi:"):
        if result.lower().startswith(prefix):
            result = result[len(prefix):].strip()
            break
    if result.startswith("10."):
        return url("https://doi.org/" + result)
    return None
