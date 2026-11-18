"""The PII scrubber — the only path from retrieval to any LLM.

Contract (Lead-frozen, ``src/orchestrator/pipeline.py``)::

    EvidencePack --scrub--> ScrubbedPack

Behavior, per ``docs/agents/04-pii-scrubber-agent.md`` and the PII policy in
``docs/06-guardrails-pii.md``:

1. Presidio analyzer over every chunk text *and title* (a title is part of the
   chunk and would otherwise leak) for person, email, phone, location, card and
   the custom classes in :mod:`pii.recognizers`.
2. Placeholders are allocated once per ``(label, identity)`` across the whole
   pack, in chunk order then text position, so the same value (and the same
   person under a different mention) always renders identically.
3. Allowlisted non-person terms (project names, the ``Alice Migration``
   codename, coarse geo) are never redacted.
4. Every redaction is audited (``pii.audit``) and one ``pii.scrubbed`` event is
   emitted with counts by type. No original values leave this module.
5. Fail closed: a missing/broken analyzer raises
   :class:`~pii.errors.PiiUnavailableError`; an entity class we cannot map to a
   placeholder is never skipped (a detected span left in place is a potential
   leak), and it raises :class:`~pii.errors.PiiScrubError` like any other
   internal failure. A partially scrubbed pack is never returned.
6. Idempotent: existing ``[LABEL_n]`` placeholders are masked before analysis
   and seed the numbering, so a second pass changes nothing.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import cast

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from config import get_settings
from db.session import get_session_factory
from db.tracing import Tracer, make_sink, new_trace_id
from orchestrator.pipeline import EvidenceChunk, EvidencePack, ScrubbedChunk, ScrubbedPack
from pii import analyzer as pii_analyzer
from pii import audit
from pii.allowlist import AllowList, build_allowlist
from pii.errors import PiiError, PiiScrubError, PiiUnavailableError
from pii.placeholders import (
    LABEL_BY_ENTITY,
    PII_TYPE_BY_LABEL,
    PLACEHOLDER_PATTERN,
    PlaceholderRegistry,
    normalize_value,
)
from pii.recognizers import location_like, person_like

logger = logging.getLogger(__name__)

AGENT = "pii"

#: Entity classes where a term can legitimately be an org/project/codename.
#: Hard signals (email, phone, card, keys) are never allowlist-suppressed.
NAME_LIKE_ENTITIES = frozenset({"PERSON", "LOCATION", "ADDRESS"})

#: Resolution priority: the more reliable signal wins an overlap. Without this,
#: spaCy's ``PERSON`` span over "Email tobias.lindqvist@…" would swallow the
#: email address and redact the whole phrase as one name placeholder.
ENTITY_PRIORITY: dict[str, int] = {
    "PRIVATE_KEY": 0,
    "API_KEY": 1,
    "EMAIL_ADDRESS": 2,
    "CREDIT_CARD": 2,
    "PHONE_NUMBER": 3,
    "DATE_OF_BIRTH": 4,
    "ADDRESS": 5,
    "PERSON": 6,
    "LOCATION": 7,
}


@dataclass(slots=True)
class _Stats:
    """Scrub counters for the audit event (never contains original values)."""

    detected_spans: int = 0
    allowlisted_spans: int = 0
    declined_person_spans: int = 0
    declined_location_spans: int = 0


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A detected span that passed policy filtering."""

    start: int
    end: int
    entity_type: str
    label: str
    identity: str
    score: float


@dataclass(slots=True)
class _ScrubContext:
    """Per-call policy, shared by every field of every chunk."""

    entities: tuple[str, ...]
    persons: tuple[str, ...]
    allowlist: AllowList
    registry: PlaceholderRegistry
    salt: str
    stats: _Stats = field(default_factory=_Stats)
    records: list[audit.RedactionRecord] = field(default_factory=list)


async def scrub(
    pack: EvidencePack,
    *,
    trace_id: str | None = None,
    known_person_names: list[str] | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    allowlist_terms: Iterable[str] | None = None,
) -> ScrubbedPack:
    """Redact personal data from every chunk of ``pack``.

    Args:
        pack: Retrieval output. Never sent to an LLM unchanged.
        trace_id: Request trace for ``audit.redactions``/``audit.events``. A new
            id is generated (and logged) when omitted.
        known_person_names: People from the entity registry. With a registry,
            ``PERSON`` is redacted only for corroborated names, which keeps org
            and project names intact; without one, name-shaped NER results are
            still redacted (precision drops, but nothing personal leaks).
        session_factory: Where to write ``audit.redactions``. Defaults to the
            process-wide factory; no database means rows are logged and dropped
            (the API layer enforces ``AUDIT_STRICT``, not this module).
        allowlist_terms: Extra non-person terms to preserve (project names,
            codenames) on top of the configured allowlist.

    Raises:
        PiiUnavailableError: the analyzer/model is unavailable, or the
            anonymizer could not replace a span.
        PiiScrubError: scrubbing failed part-way (including an entity class with
            no placeholder label); no pack is returned.
    """
    started = time.perf_counter()
    try:
        settings = get_settings()
        context = _ScrubContext(
            entities=tuple(settings.pii_entities_enabled) or pii_analyzer.DEFAULT_ENTITIES,
            persons=_clean_names(known_person_names),
            allowlist=build_allowlist(allowlist_terms),
            registry=PlaceholderRegistry(),
            salt=audit.resolve_salt(),
        )
        for chunk in pack.chunks:
            context.registry.observe(chunk.text)
            context.registry.observe(chunk.title)

        if not context.persons:
            logger.warning(
                "scrub() was called without known_person_names: only name-shaped NER matches "
                "are redacted, so names the small model misses will pass through "
                "(see src/pii/HANDOFF.md)"
            )

        chunks = [await _scrub_chunk(chunk, context) for chunk in pack.chunks]
    except PiiError:
        raise
    except Exception as exc:
        raise PiiScrubError(f"PII scrubbing failed: {exc}") from exc

    return await _finish(
        pack,
        chunks,
        context,
        trace_id=trace_id,
        session_factory=session_factory,
        started=started,
    )


async def _scrub_chunk(chunk: EvidenceChunk, context: _ScrubContext) -> ScrubbedChunk:
    """Scrub one chunk's body and title with a shared placeholder registry."""
    first_record = len(context.records)
    text = await _scrub_field(chunk.text, chunk, context)
    title = await _scrub_field(chunk.title, chunk, context)
    redacted_types = sorted(
        {record.pii_type for record in context.records[first_record:]}
    )
    return ScrubbedChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        text=text,
        score=chunk.score,
        source=chunk.source,
        source_id=chunk.source_id,
        title=title,
        url=chunk.url,
        updated_at=chunk.updated_at,
        stale=chunk.stale,
        meta=dict(chunk.meta),
        redacted_types=redacted_types,
    )


async def _scrub_field(text: str, chunk: EvidenceChunk, context: _ScrubContext) -> str:
    """Redact one text field (chunk body or title) and record what was redacted."""
    if not text.strip():
        return text

    placeholders = [(m.start(), m.end()) for m in PLACEHOLDER_PATTERN.finditer(text)]
    # Mask existing placeholders so the NER model cannot fold them into a new
    # span; spaces keep every offset valid, so results map onto ``text``.
    masked = PLACEHOLDER_PATTERN.sub(lambda match: " " * len(match.group(0)), text)

    detected = await _analyze(masked, context)
    context.stats.detected_spans += len(detected)

    allow_ranges = context.allowlist.ranges(text)
    candidates: list[_Candidate] = []
    for span in detected:
        original = text[span.start : span.end]
        if not original.strip():
            continue
        if _overlaps(span.start, span.end, placeholders):
            continue  # already a placeholder: never re-redact
        label = LABEL_BY_ENTITY.get(span.entity_type)
        if label is None:
            # Ruling #15 (docs/08-decisions.md): skipping a detected span is a
            # silent leak, and an unmapped class means our mapping drifted from
            # the analyzer. No span text in the message.
            raise PiiScrubError(
                f"PII entity class {span.entity_type!r} has no placeholder label; "
                "refusing to return a pack with an unredactable span"
            )
        if (
            span.entity_type in NAME_LIKE_ENTITIES
            and allow_ranges
            and _overlaps(span.start, span.end, allow_ranges)
        ):
            context.stats.allowlisted_spans += 1
            continue
        identity = span.identity
        if span.entity_type == "PERSON" and identity is None:
            if context.persons:
                # A registry is available: uncorroborated NER names are the main
                # source of org/title false positives (measured, see HANDOFF).
                context.stats.declined_person_spans += 1
                continue
            if not person_like(original):
                context.stats.declined_person_spans += 1
                continue
        elif span.entity_type == "LOCATION" and not location_like(original):
            # "PHX-131", "order_id", "Q4" are not places.
            context.stats.declined_location_spans += 1
            continue
        candidates.append(
            _Candidate(
                start=span.start,
                end=span.end,
                entity_type=span.entity_type,
                label=label,
                identity=normalize_value(identity if identity is not None else original),
                score=span.score,
            )
        )

    accepted = _resolve(candidates)
    if not accepted:
        return text

    replacements: dict[tuple[int, int], str] = {}
    for candidate in accepted:
        placeholder = context.registry.placeholder_for(candidate.label, candidate.identity)
        replacements[(candidate.start, candidate.end)] = placeholder
        original = text[candidate.start : candidate.end]
        context.records.append(
            audit.RedactionRecord(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                pii_type=PII_TYPE_BY_LABEL[candidate.label],
                placeholder=placeholder,
                original_hash=audit.salted_hash(original, context.salt),
            )
        )
    return pii_analyzer.anonymize_text(
        text, cast("Sequence[pii_analyzer.SpanLike]", accepted), replacements
    )


async def _analyze(text: str, context: _ScrubContext) -> list[pii_analyzer.DetectedSpan]:
    """Analyzer call with fail-closed error handling."""
    try:
        return await pii_analyzer.analyze(
            text, entities=context.entities, person_names=context.persons
        )
    except PiiError:
        raise
    except Exception as exc:
        raise PiiUnavailableError(f"PII analysis failed: {exc}") from exc


def _resolve(candidates: list[_Candidate]) -> list[_Candidate]:
    """Pick a non-overlapping, deterministic subset; higher-confidence classes win."""
    ordered = sorted(
        candidates,
        key=lambda c: (
            ENTITY_PRIORITY.get(c.entity_type, len(ENTITY_PRIORITY)),
            c.start,
            -(c.end - c.start),
        ),
    )
    accepted: list[_Candidate] = []
    for candidate in ordered:
        if any(
            _overlaps(candidate.start, candidate.end, [(other.start, other.end)])
            for other in accepted
        ):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda c: (c.start, c.end))


def _overlaps(start: int, end: int, ranges: Iterable[tuple[int, int]]) -> bool:
    return any(start < range_end and range_start < end for range_start, range_end in ranges)


def _clean_names(names: Iterable[str] | None) -> tuple[str, ...]:
    """Deduplicate registry names, preserving order."""
    if not names:
        return ()
    return tuple(dict.fromkeys(name.strip() for name in names if name and name.strip()))


def _counts(records: Iterable[audit.RedactionRecord]) -> dict[str, int]:
    return dict(sorted(Counter(record.pii_type for record in records).items()))


async def _finish(
    pack: EvidencePack,
    chunks: list[ScrubbedChunk],
    context: _ScrubContext,
    *,
    trace_id: str | None,
    session_factory: async_sessionmaker[AsyncSession] | None,
    started: float,
) -> ScrubbedPack:
    """Write the audit trail and build the pack. Failures never yield a pack."""
    trace = trace_id or new_trace_id()
    if trace_id is None:
        logger.warning("scrub() called without a trace_id; generated %s for the audit rows", trace)

    factory = session_factory if session_factory is not None else get_session_factory()
    persisted = await audit.make_redaction_writer(factory).write(trace, context.records)

    payload: dict[str, object] = {
        "chunks": len(chunks),
        "redactions": len(context.records),
        "by_type": _counts(context.records),
        "chunks_with_redactions": sum(1 for chunk in chunks if chunk.redacted_types),
        "placeholders": context.registry.assignment_count,
        "detected_spans": context.stats.detected_spans,
        "allowlisted_spans": context.stats.allowlisted_spans,
        "declined_person_spans": context.stats.declined_person_spans,
        "declined_location_spans": context.stats.declined_location_spans,
        "person_registry_size": len(context.persons),
        "entities_enabled": list(context.entities),
        "audit_persisted": persisted,
    }
    latency_ms = int((time.perf_counter() - started) * 1000)
    await Tracer(make_sink(factory), trace).emit(
        AGENT, "pii.scrubbed", payload, latency_ms=latency_ms
    )

    return ScrubbedPack(
        question=pack.question,
        chunks=chunks,
        redaction_count=len(context.records),
        degraded_reason=pack.degraded_reason,
    )
