"""PII detection and redaction (M3+).

Public entry point: :func:`pii.scrubber.scrub`. The scrubber is a hard gate
between retrieval and any LLM (``docs/06-guardrails-pii.md``); it either returns
a fully redacted :class:`~orchestrator.pipeline.ScrubbedPack` or raises
:class:`~pii.errors.PiiError`.
"""

from __future__ import annotations

from pii.audit import RedactionRecord, salted_hash
from pii.errors import PiiError, PiiScrubError, PiiUnavailableError
from pii.scrubber import scrub

__all__ = [
    "PiiError",
    "PiiScrubError",
    "PiiUnavailableError",
    "RedactionRecord",
    "salted_hash",
    "scrub",
]
