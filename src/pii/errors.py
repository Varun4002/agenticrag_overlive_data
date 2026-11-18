"""Typed failures for the PII scrubber.

The scrubber is a hard gate (``docs/06-guardrails-pii.md``): it either returns a
fully scrubbed pack or raises. There is no partial or "best effort" result, and
no code path returns the input unchanged because a dependency was missing.

``PiiUnavailableError`` is raised when the detection machinery itself cannot
run (Presidio not importable, spaCy model missing, engine construction failed,
analyzer raised). It also covers the anonymizer: ``pii.analyzer.anonymize_text``
wraps both an engine error and a span it cannot map to its allocated
placeholder in this type, so no partially replaced text is returned. Callers
must fail the request — never fall back to raw text.

``PiiScrubError`` is raised for any other internal failure mid-scrub: a span
that could not be replaced, or an entity class the analyzer returned that has no
placeholder label (``pii.placeholders.LABEL_BY_ENTITY``). An unmapped class is
never skipped — a detected span left in place is a potential leak, and an
unknown class means the mapping drifted from the analyzer (ruling #15 in
``docs/08-decisions.md``).
"""

from __future__ import annotations


class PiiError(Exception):
    """Base class for every scrubber failure."""


class PiiUnavailableError(PiiError):
    """PII detection is not usable in this process (fail closed)."""


class PiiScrubError(PiiError):
    """Scrubbing started but could not produce a complete redacted pack."""
