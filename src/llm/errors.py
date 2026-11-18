"""Typed errors for LLM calls.

The orchestrator maps these to user-visible messages (docs/agents/00-orchestrator.md)
instead of leaking stack traces, so the taxonomy has to be explicit:
retryable conditions (timeout, rate limit, server error) vs terminal ones
(auth, malformed response).
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for every LLM-client failure."""

    retryable: bool = False

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


class LLMTimeoutError(LLMError):
    """The provider did not answer within the configured timeout."""

    retryable = True


class LLMRateLimitError(LLMError):
    """Provider rejected the call due to rate limiting (HTTP 429)."""

    retryable = True

    def __init__(
        self, message: str, *, retry_after_s: float | None = None, provider: str | None = None
    ) -> None:
        super().__init__(message, provider=provider)
        self.retry_after_s = retry_after_s


class LLMServerError(LLMError):
    """Provider returned a 5xx or a transient transport failure."""

    retryable = True


class LLMAuthError(LLMError):
    """Missing/invalid credentials — retrying cannot help."""


class LLMBadRequestError(LLMError):
    """Provider rejected the request shape (HTTP 4xx, excluding 429)."""


class LLMResponseError(LLMError):
    """Provider answered, but the payload could not be parsed into a completion."""
