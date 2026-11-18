"""Typed errors for indexing and embedding.

Separate from :mod:`llm.errors` on purpose: a failed embedding batch is an
indexing failure (the orchestrator reports it differently from a broken answer
generation) and the auth/retry semantics belong to the embedding provider, not
the chat provider. The taxonomy mirrors ``llm/errors.py`` so operators do not
have to learn two shapes: retryable (timeout, rate limit, server/transport) vs
terminal (auth, bad request, malformed response, wrong dimension).
"""

from __future__ import annotations


class IndexingError(Exception):
    """Base class for every indexing failure."""


class EmbeddingError(IndexingError):
    """An embedding provider call failed."""

    retryable: bool = False

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


class EmbeddingTimeoutError(EmbeddingError):
    """The provider did not answer within the configured timeout."""

    retryable = True


class EmbeddingRateLimitError(EmbeddingError):
    """Provider rejected the call due to rate limiting (HTTP 429)."""

    retryable = True

    def __init__(
        self, message: str, *, retry_after_s: float | None = None, provider: str | None = None
    ) -> None:
        super().__init__(message, provider=provider)
        self.retry_after_s = retry_after_s


class EmbeddingServerError(EmbeddingError):
    """Provider returned a 5xx or a transient transport failure."""

    retryable = True


class EmbeddingAuthError(EmbeddingError):
    """Missing/invalid credentials — retrying cannot help."""


class EmbeddingBadRequestError(EmbeddingError):
    """Provider rejected the request shape (HTTP 4xx, excluding 429)."""


class EmbeddingResponseError(EmbeddingError):
    """Provider answered, but the payload could not be parsed into vectors."""


class EmbeddingDimensionError(EmbeddingError):
    """The provider returned vectors whose size is not the configured one.

    This is a *configuration* error (model swapped without a migration), so it
    is terminal and surfaces once at startup rather than per chunk.
    """


class IndexingConfigError(IndexingError):
    """Required configuration (e.g. ``DATABASE_URL``) is missing or invalid."""
