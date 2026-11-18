"""Custom Presidio recognizers for the classes the built-in registry misses.

Why each one exists (all verified by probe against the fixture corpus, see
``src/pii/HANDOFF.md``):

* :class:`EmailAddressRecognizer` — Presidio's ``EmailRecognizer`` validates a
  match with ``tldextract``. ``.example`` is an RFC 2606 reserved TLD that
  tldextract does not know, so **every** corpus email
  (``priya.raghunathan@northwind.example``) was dropped by the stock recognizer.
  This one uses the structural regex and no TLD lookup (also keeps the
  scrubber offline: no suffix-list download).
* :class:`ApiKeyRecognizer` — provider tokens (``sk-``, AWS, GitHub, Slack,
  ``Bearer``) plus a generic long-token pattern gated by a Shannon-entropy
  check so ordinary long words/slugs are not redacted.
* :class:`PemBlockRecognizer` — multi-line ``-----BEGIN ... PRIVATE KEY``
  blocks (Layer 1 of the guardrail stack flags these at ingestion too).
* :class:`AddressRecognizer` — Presidio ships no street-address recognizer;
  ``LOCATION`` alone only catches city names.
* :class:`DateOfBirthRecognizer` — deliberately narrow: Presidio's
  ``DATE_TIME`` redacts every deadline, which the policy forbids. Only an
  explicit birth phrase ("date of birth: …") is matched.
* :class:`PersonRegistryRecognizer` — the entity-registry person list (per-call
  ad-hoc recognizer). It also gives first-name mentions the *same* identity as
  the full name so coreference survives redaction.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import ClassVar

from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult

#: Entity classes this module introduces (Presidio allows custom names).
API_KEY = "API_KEY"
PRIVATE_KEY = "PRIVATE_KEY"
ADDRESS = "ADDRESS"
DATE_OF_BIRTH = "DATE_OF_BIRTH"

# Deliberately stricter than Presidio's RFC-5322-ish pattern: the stock one
# allows "/", "?" and "=" in the local part, so an email inside a URL was
# matched from "ci.example.com/job/PHX-142?notify=…" and swallowed the URL.
_EMAIL_REGEX = (
    r"\b[A-Za-z0-9._%+'-]+@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}\b"
)

_KEY_PREFIX_RE = re.compile(r"^(?:sk-|AKIA|ASIA|gh[pousr]_|github_pat_|xox[abprs]-|Bearer\s)")
_GENERIC_TOKEN_RE = re.compile(r"^[A-Za-z0-9+/=_-]+$")

_MIN_TOKEN_LENGTH = 32
_MIN_TOKEN_ENTROPY = 3.5
#: Separator-heavy tokens are slugs/release tags ("deployment-pipeline-configuration-2026-11-17"),
#: not credentials. Secrets use at most a couple of "-"/"_" as base64url filler.
_MAX_TOKEN_SEPARATORS = 2

_NAME_TOKEN_RE = re.compile(r"[A-Z][A-Za-z'\u2019.-]+")


def shannon_entropy(value: str) -> float:
    """Bits of entropy per character of ``value``."""
    if not value:
        return 0.0
    counts = Counter(value)
    total = len(value)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


class EmailAddressRecognizer(PatternRecognizer):
    """Email addresses with any dotted domain, including reserved TLDs."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity="EMAIL_ADDRESS",
            name="EmailAddressRecognizer",
            patterns=[Pattern("email", _EMAIL_REGEX, 0.7)],
            context=["email", "mail", "contact"],
        )

    def validate_result(self, pattern_text: str) -> bool:
        """Accept the structural match; the regex already requires a dotted domain."""
        local, _, domain = pattern_text.partition("@")
        return bool(local and "." in domain and not domain.startswith("."))


class ApiKeyRecognizer(PatternRecognizer):
    """Provider API keys and generic high-entropy tokens."""

    PATTERNS: ClassVar[list[Pattern]] = [
        Pattern("sk-key", r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{15,}\b", 0.9),
        Pattern("aws-access-key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", 0.95),
        Pattern("github-token", r"\b(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{20,}\b", 0.92),
        Pattern("slack-token", r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b", 0.9),
        Pattern("bearer-token", r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}", 0.85),
        Pattern(
            "high-entropy-token",
            rf"\b[A-Za-z0-9+/=_-]{{{_MIN_TOKEN_LENGTH},}}\b",
            0.5,
        ),
    ]

    def __init__(self) -> None:
        super().__init__(
            supported_entity=API_KEY,
            name="ApiKeyRecognizer",
            patterns=self.PATTERNS,
            context=["key", "token", "secret", "credential", "password", "api"],
        )

    def validate_result(self, pattern_text: str) -> bool:
        """Known provider prefixes pass; the generic pattern must look random.

        The entropy gate is why the scrubber does not redact ordinary long
        identifiers (issue keys, slugs, release tags) as credentials.
        """
        if _KEY_PREFIX_RE.match(pattern_text):
            return True
        if len(pattern_text) < _MIN_TOKEN_LENGTH or not _GENERIC_TOKEN_RE.match(pattern_text):
            return False
        if not any(char.isdigit() for char in pattern_text):
            return False
        separators = sum(1 for char in pattern_text if char in "-_")
        if separators > _MAX_TOKEN_SEPARATORS:
            return False
        return shannon_entropy(pattern_text) >= _MIN_TOKEN_ENTROPY


class PemBlockRecognizer(PatternRecognizer):
    """``-----BEGIN ... PRIVATE KEY-----`` blocks, including their body."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity=PRIVATE_KEY,
            name="PemBlockRecognizer",
            patterns=[
                Pattern(
                    "pem-private-key",
                    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
                    1.0,
                )
            ],
            context=["private", "key", "certificate"],
        )


class AddressRecognizer(PatternRecognizer):
    """US-style street addresses, with optional suite and city/state/zip."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity=ADDRESS,
            name="AddressRecognizer",
            patterns=[
                Pattern(
                    "street-address",
                    r"\b\d{1,6}\s+(?:[A-Z][A-Za-z0-9.'-]*\s+){1,4}"
                    r"(?:Street|St|Avenue|Ave|Drive|Dr|Road|Rd|Boulevard|Blvd|Lane|Ln|Way"
                    r"|Court|Ct|Place|Pl|Terrace|Terr|Parkway|Pkwy|Highway|Hwy)\b"
                    r"(?:,?\s*(?:Suite|Ste|Unit|Apt|Floor|Fl|#)\.?\s*[0-9A-Za-z]+)?"
                    r"(?:,?\s*[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*)*"
                    r",\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)?",
                    0.85,
                )
            ],
            context=["address", "office", "located", "headquarters", "visit"],
        )


class DateOfBirthRecognizer(PatternRecognizer):
    """Only explicit birth statements; ordinary dates/deadlines are untouched."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity=DATE_OF_BIRTH,
            name="DateOfBirthRecognizer",
            patterns=[
                Pattern(
                    "date-of-birth",
                    r"\b(?:date\s+of\s+birth|dob|born)\b\s*(?:on|:)?\s*"
                    r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}"
                    r"|[A-Z][a-z]+\s+\d{1,2},\s*\d{4})",
                    0.85,
                )
            ],
            context=["birth", "born", "dob"],
        )


def person_like(value: str) -> bool:
    """Heuristic shape of a personal name (used only without a person registry).

    Two or three capitalised alphabetic tokens; no digits, punctuation runs,
    newlines or lowercase connectives. Deliberately strict: ``en_core_web_sm``
    labels many org/title strings ``PERSON`` on the corpus, and the registry is
    the primary precision signal when it is supplied.
    """
    if "\n" in value or "\r" in value:
        return False
    tokens = value.split()
    if not 2 <= len(tokens) <= 3:
        return False
    return all(_NAME_TOKEN_RE.fullmatch(token) for token in tokens)


_LOCATION_TOKEN_RE = re.compile(r"[A-Z][A-Za-z'\u2019.-]*")


def location_like(value: str) -> bool:
    """Placename shape: no digits, no identifiers, at most four capitalised words.

    ``en_core_web_sm`` labels issue keys and field names ``LOCATION``
    (``PHX-131``), so a bare ``LOCATION`` result is not trusted on its own.
    """
    if "\n" in value or "\r" in value:
        return False
    tokens = value.split()
    if not 1 <= len(tokens) <= 4:
        return False
    return all(_LOCATION_TOKEN_RE.fullmatch(token) for token in tokens)


class PersonRegistryRecognizer(EntityRecognizer):
    """Match registry person names, including whitespace and first-name variants.

    ``recognition_metadata["pii_identity"]`` carries the canonical registry name
    so ``Priya`` and ``Priya Raghunathan`` resolve to the same placeholder.
    """

    def __init__(self, names: Sequence[str], supported_language: str = "en") -> None:
        super().__init__(
            supported_entities=["PERSON"],
            name="PersonRegistryRecognizer",
            supported_language=supported_language,
        )
        self._full_names: list[tuple[str, re.Pattern[str]]] = []
        first_names: Counter[str] = Counter()
        for name in dict.fromkeys(n.strip() for n in names if n and n.strip()):
            tokens = name.split()
            if not tokens:
                continue
            pieces = [re.escape(token) for token in tokens]
            self._full_names.append(
                (name, re.compile(r"(?<![\w-])" + r"\s+".join(pieces) + r"(?![\w-])"))
            )
            if len(tokens) > 1:
                first_names[tokens[0]] += 1

        # A bare first name is only redacted when it identifies exactly one
        # registry person; otherwise "Daniel" could belong to two people and the
        # placeholder would be a guess.
        self._aliases: list[tuple[str, str, re.Pattern[str]]] = [
            (
                first,
                _canonical_for(first, names),
                re.compile(r"(?<![\w-])" + re.escape(first) + r"(?![\w-])"),
            )
            for first, count in first_names.items()
            if count == 1
        ]

    def load(self) -> None:
        """No model to load: matching is regex-based."""

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: object | None = None,
    ) -> list[RecognizerResult]:
        """Emit one result per registry name occurrence."""
        if "PERSON" not in entities:
            return []
        results: list[RecognizerResult] = []
        for canonical, pattern in self._full_names:
            for match in pattern.finditer(text):
                results.append(self._result(match.start(), match.end(), canonical, 0.95))
        for _first, canonical, pattern in self._aliases:
            for match in pattern.finditer(text):
                results.append(self._result(match.start(), match.end(), canonical, 0.9))
        return EntityRecognizer.remove_duplicates(results)

    def _result(
        self, start: int, end: int, canonical: str, score: float
    ) -> RecognizerResult:
        return RecognizerResult(
            entity_type="PERSON",
            start=start,
            end=end,
            score=score,
            recognition_metadata={
                RecognizerResult.RECOGNIZER_NAME_KEY: type(self).__name__,
                "pii_identity": canonical,
            },
        )


def _canonical_for(first_name: str, names: Sequence[str]) -> str:
    for name in names:
        if name and name.split() and name.split()[0] == first_name:
            return name.strip()
    return first_name
