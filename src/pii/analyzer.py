"""Presidio engine construction and the analyzer/anonymizer seam.

Presidio + spaCy are expensive to build and are process-wide singletons; the
per-request person registry is attached as an *ad-hoc* recognizer so the engine
can stay cached. Everything here fails closed: if the engine cannot be built
(missing package, missing ``en_core_web_sm``) or a call raises,
:class:`~pii.errors.PiiUnavailableError` is raised and no pack is produced.

spaCy pipelines are not safe to share across threads, so calls are serialized
by a process-wide lock and executed in a worker thread (``asyncio.to_thread``)
to keep the event loop responsive.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

from pii.errors import PiiUnavailableError
from pii.recognizers import (
    ADDRESS,
    API_KEY,
    DATE_OF_BIRTH,
    PRIVATE_KEY,
    AddressRecognizer,
    ApiKeyRecognizer,
    DateOfBirthRecognizer,
    EmailAddressRecognizer,
    PemBlockRecognizer,
    PersonRegistryRecognizer,
)

logger = logging.getLogger(__name__)

#: spaCy model the analyzer needs. The *medium* model is deliberate: the small
#: one mis-tags person names as ORG (e.g. "Grace Adeyemi"), which silently leaks
#: names. Install: ``python -m spacy download en_core_web_md``.
SPACY_MODEL = "en_core_web_md"

LANGUAGE = "en"

#: Entity classes the scrubber understands. ``PII_ENTITIES_ENABLED`` narrows it.
DEFAULT_ENTITIES: tuple[str, ...] = (
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "CREDIT_CARD",
    ADDRESS,
    API_KEY,
    PRIVATE_KEY,
    DATE_OF_BIRTH,
)

#: One analyzer call is serialized at a time (spaCy is not thread-safe).
_ENGINE_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class DetectedSpan:
    """One PII candidate found in a text, before policy filtering."""

    start: int
    end: int
    entity_type: str
    score: float
    #: Canonical registry identity when the span came from the person registry.
    identity: str | None = None


class SpanLike(Protocol):
    """Anything with character offsets and an entity class (see ``DetectedSpan``)."""

    start: int
    end: int
    entity_type: str
    score: float


@dataclass(frozen=True, slots=True)
class Engines:
    """The cached Presidio pair."""

    analyzer: Any
    anonymizer: Any


@lru_cache(maxsize=1)
def _build_engines() -> Engines:
    """Construct the analyzer + anonymizer; raises ``PiiUnavailableError``."""
    try:
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine
    except ImportError as exc:  # pragma: no cover - dependency-level failure
        raise PiiUnavailableError(
            "Presidio is not installed; install presidio-analyzer and presidio-anonymizer"
        ) from exc

    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": LANGUAGE, "model_name": SPACY_MODEL}],
        }
    )
    try:
        nlp_engine = provider.create_engine()
    except Exception as exc:  # pragma: no cover - missing model / broken install
        raise PiiUnavailableError(
            f"spaCy model {SPACY_MODEL!r} is unavailable; run "
            f"`python -m spacy download {SPACY_MODEL}`"
        ) from exc

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(languages=[LANGUAGE], nlp_engine=nlp_engine)
    # Replace Presidio's email recognizer: it rejects reserved TLDs (".example"),
    # which is what the whole fixture corpus and most test data use.
    registry.remove_recognizer("EmailRecognizer")
    for recognizer in (
        EmailAddressRecognizer(),
        ApiKeyRecognizer(),
        PemBlockRecognizer(),
        AddressRecognizer(),
        DateOfBirthRecognizer(),
    ):
        registry.add_recognizer(recognizer)

    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine, registry=registry, supported_languages=[LANGUAGE]
    )
    return Engines(analyzer=analyzer, anonymizer=AnonymizerEngine())  # type: ignore[no-untyped-call]


def get_engines() -> Engines:
    """Return the cached engines, converting any construction failure to a typed error."""
    try:
        return _build_engines()
    except PiiUnavailableError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise PiiUnavailableError(f"PII analyzer could not be initialized: {exc}") from exc


def reset_engines() -> None:
    """Drop the cached engines (tests only)."""
    _build_engines.cache_clear()


def analyze_text(
    text: str,
    *,
    entities: Sequence[str],
    person_names: Sequence[str] = (),
) -> list[DetectedSpan]:
    """Analyze one text; raises ``PiiUnavailableError`` if detection cannot run."""
    if not text.strip():
        return []
    engines = get_engines()
    ad_hoc = [PersonRegistryRecognizer(person_names)] if person_names else []
    try:
        with _ENGINE_LOCK:
            results = engines.analyzer.analyze(
                text=text,
                language=LANGUAGE,
                entities=list(entities),
                ad_hoc_recognizers=ad_hoc,
                score_threshold=0.0,
            )
    except Exception as exc:
        raise PiiUnavailableError(f"PII analysis failed: {exc}") from exc

    spans: list[DetectedSpan] = []
    for result in results:
        identity = None
        metadata = getattr(result, "recognition_metadata", None)
        if isinstance(metadata, dict):
            candidate = metadata.get("pii_identity")
            if isinstance(candidate, str) and candidate:
                identity = candidate
        spans.append(
            DetectedSpan(
                start=int(result.start),
                end=int(result.end),
                entity_type=str(result.entity_type),
                score=float(result.score),
                identity=identity,
            )
        )
    return spans


async def analyze(
    text: str,
    *,
    entities: Sequence[str],
    person_names: Sequence[str] = (),
) -> list[DetectedSpan]:
    """Async wrapper: run :func:`analyze_text` off the event loop."""
    return await asyncio.to_thread(
        analyze_text, text, entities=entities, person_names=person_names
    )


def anonymize_text(
    text: str,
    spans: Sequence[SpanLike],
    replacements: dict[tuple[int, int], str],
) -> str:
    """Replace every span with its placeholder via the Presidio anonymizer.

    ``replacements`` is keyed by ``(start, end)`` so two identical values with
    different placeholders can never collide.
    """
    if not spans:
        return text
    from presidio_anonymizer.entities import OperatorConfig, RecognizerResult

    def replace(matched: str) -> str:
        for span in spans:
            if text[span.start : span.end] == matched:
                placeholder = replacements.get((span.start, span.end))
                if placeholder is not None:
                    return placeholder
        # A span the anonymizer merged or shifted: fail closed rather than ship
        # text we did not map to a placeholder.
        raise LookupError(f"unmapped PII span ({len(matched)} chars)")

    try:
        result = get_engines().anonymizer.anonymize(
            text=text,
            analyzer_results=[
                RecognizerResult(
                    entity_type=span.entity_type,
                    start=span.start,
                    end=span.end,
                    score=span.score,
                )
                for span in spans
            ],
            operators={"DEFAULT": OperatorConfig("custom", {"lambda": replace})},
            # Never merge neighbouring spans: each one has its own placeholder.
            merge_entities_with_spaces=False,
        )
    except Exception as exc:
        raise PiiUnavailableError(f"PII anonymization failed: {exc}") from exc
    return str(result.text)
