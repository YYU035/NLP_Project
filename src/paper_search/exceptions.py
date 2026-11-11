"""Domain errors for future service implementations."""


class PaperSearchError(Exception):
    """Base error for expected application failures."""


class APIClientError(PaperSearchError):
    """An upstream paper provider could not satisfy a request."""


class PDFVerificationError(PaperSearchError):
    """A PDF could not be retrieved, parsed, or verified."""
