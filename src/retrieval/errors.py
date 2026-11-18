"""Typed retrieval failures.

Retrieval degrades rather than crashes where the brief says so: a failed query
embedding falls back to FTS-only and is reported in ``EvidencePack.degraded_reason``
plus a ``retrieval.fallback`` trace event. The errors here are the cases where
silently returning nothing would hide a real misconfiguration.
"""

from __future__ import annotations


class RetrievalError(Exception):
    """Base class for retrieval failures."""


class RetrievalConfigError(RetrievalError):
    """Retrieval was asked to run without the resources it needs.

    Raised for an unconfigured database (or an embedder with the wrong vector
    dimension on the *write* path). An unavailable embedding *provider* is not
    this: that is the documented FTS fallback.
    """


class RetrievalQueryError(RetrievalError):
    """The request cannot be turned into a valid query (e.g. ``top_k`` <= 0)."""


__all__ = ["RetrievalConfigError", "RetrievalError", "RetrievalQueryError"]
