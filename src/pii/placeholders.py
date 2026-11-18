"""Stable, pack-scoped placeholder allocation.

Placeholders are assigned once per *(label, identity)* pair and reused for the
rest of the pack, so the same original value always renders as the same token
and coreference survives redaction ("assigned to [PERSON_1]" in one chunk and
"Priya" in another both become ``[PERSON_1]``).

Numbering is deterministic: chunks in pack order, then text position, counters
per label starting at 1. Pre-existing placeholders in the input seed the
counters, so scratching an already-scrubbed text never reuses a number (and
never re-redacts a placeholder).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Placeholder labels. ``PERSON``/``EMAIL``/``PHONE`` mirror the policy doc;
#: the rest follow the same shape so a prompt can treat them opaquely.
LABELS: tuple[str, ...] = (
    "PERSON",
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "LOCATION",
    "CREDIT_CARD",
    "API_KEY",
    "PRIVATE_KEY",
    "DOB",
)

PLACEHOLDER_PATTERN = re.compile(r"\[(" + "|".join(LABELS) + r")_(\d+)\]")

#: Presidio entity class -> placeholder label.
LABEL_BY_ENTITY: dict[str, str] = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "LOCATION": "LOCATION",
    "ADDRESS": "ADDRESS",
    "CREDIT_CARD": "CREDIT_CARD",
    "API_KEY": "API_KEY",
    "PRIVATE_KEY": "PRIVATE_KEY",
    "DATE_OF_BIRTH": "DOB",
}

#: Placeholder label -> ``audit.pii_type`` enum value (docs/05-data-model.md).
#: ``LOCATION`` is audited as ``address``: geographic identifiers are the
#: closest member of the frozen enum, and the placeholder keeps the detail.
PII_TYPE_BY_LABEL: dict[str, str] = {
    "PERSON": "person",
    "EMAIL": "email",
    "PHONE": "phone",
    "ADDRESS": "address",
    "LOCATION": "address",
    "CREDIT_CARD": "other",
    "API_KEY": "api_key",
    "PRIVATE_KEY": "api_key",
    "DOB": "other",
}


def normalize_value(value: str) -> str:
    """Identity of an original value: case-folded, whitespace collapsed."""
    return " ".join(value.split()).casefold()


@dataclass
class PlaceholderRegistry:
    """Allocates ``[LABEL_n]`` tokens across one evidence pack."""

    _counters: dict[str, int] = field(default_factory=dict)
    _assignments: dict[tuple[str, str], str] = field(default_factory=dict)

    def observe(self, text: str) -> None:
        """Raise counters past any placeholder already present in ``text``."""
        for match in PLACEHOLDER_PATTERN.finditer(text):
            label, index = match.group(1), int(match.group(2))
            self._counters[label] = max(self._counters.get(label, 0), index)

    def placeholder_for(self, label: str, identity: str) -> str:
        """Return the stable placeholder for ``identity``, allocating if new."""
        if label not in LABELS:  # pragma: no cover - guarded by LABEL_BY_ENTITY
            raise KeyError(f"unknown placeholder label: {label}")
        key = (label, identity)
        existing = self._assignments.get(key)
        if existing is not None:
            return existing
        self._counters[label] = self._counters.get(label, 0) + 1
        placeholder = f"[{label}_{self._counters[label]}]"
        self._assignments[key] = placeholder
        return placeholder

    @property
    def assignment_count(self) -> int:
        """Distinct (label, identity) pairs allocated so far."""
        return len(self._assignments)
