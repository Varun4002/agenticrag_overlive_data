"""The PII audit trail must report what actually happened.

Ruling #18 (``docs/08-decisions.md``): a writer that only logs must not make
``pii.scrubbed`` claim ``audit_persisted: true``. The database-backed writer is
proved against a scratch Postgres via the fixtures in ``conftest.py``; those
tests skip with an explicit reason when no scratch database is configured and
refuse any database whose name does not end in ``_pii_test``.

The planted values come from ``pii_fixtures``/the corpus registry, so the rows
below are the same redactions the rest of the suite sees.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pii_fixtures import CHUNK_ID, DOCUMENT_ID, make_chunk, make_pack
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from db.models import AuditEvent, Redaction
from db.tracing import EventRecord, InMemoryEventSink, read_trace
from orchestrator.pipeline import EvidencePack
from pii import audit, scrub
from pii import scrubber as scrubber_module

TRACE = "trace-audit"

PII_TEXT = (
    "Reach Priya Raghunathan at priya.raghunathan@northwind.example or +1-415-555-0142."
)

#: placeholder -> (audit pii_type, the original value it replaced).
PLANTED: dict[str, tuple[str, str]] = {
    "[PERSON_1]": ("person", "Priya Raghunathan"),
    "[EMAIL_1]": ("email", "priya.raghunathan@northwind.example"),
    "[PHONE_1]": ("phone", "+1-415-555-0142"),
}


def _audit_pack() -> EvidencePack:
    """One chunk whose ids exist in the scratch database (the conftest seeds them)."""
    return EvidencePack(
        question="Who do I contact about the cutover?",
        chunks=[make_chunk(PII_TEXT, chunk_id=str(CHUNK_ID), document_id=str(DOCUMENT_ID))],
    )


async def _redaction_rows(
    factory: async_sessionmaker[AsyncSession], trace_id: str
) -> list[Redaction]:
    async with factory() as session:
        result = await session.scalars(select(Redaction).where(Redaction.trace_id == trace_id))
        return list(result)


def _event_payloads(
    events: Sequence[EventRecord | AuditEvent], name: str
) -> list[dict[str, object]]:
    return [event.payload for event in events if event.event == name]


async def test_null_writer_does_not_claim_persistence(
    monkeypatch: pytest.MonkeyPatch, person_names: list[str]
) -> None:
    """No database: the rows are logged and dropped, so the event says ``false``."""
    sink = InMemoryEventSink()
    # The scrubber binds both helpers into its own namespace at import time.
    monkeypatch.setattr(scrubber_module, "get_session_factory", lambda: None)
    monkeypatch.setattr(scrubber_module, "make_sink", lambda factory: sink)

    result = await scrub(
        make_pack(PII_TEXT), trace_id=TRACE, known_person_names=person_names
    )

    assert result.redaction_count == 3
    payloads = _event_payloads(sink.events_for(TRACE), "pii.scrubbed")
    assert len(payloads) == 1
    assert payloads[0]["audit_persisted"] is False
    assert payloads[0]["redactions"] == 3


@pytest.mark.usefixtures("clean_pii_db")
async def test_redactions_are_persisted_with_salted_hashes(
    pii_session_factory: async_sessionmaker[AsyncSession],
    person_names: list[str],
) -> None:
    """One ``audit.redactions`` row per redaction; only the salted hash is stored."""
    result = await scrub(
        _audit_pack(),
        trace_id=TRACE,
        known_person_names=person_names,
        session_factory=pii_session_factory,
    )
    assert result.redaction_count == 3

    rows = await _redaction_rows(pii_session_factory, TRACE)
    assert len(rows) == result.redaction_count
    assert {row.placeholder for row in rows} == set(PLANTED)

    salt = audit.resolve_salt()
    for row in rows:
        pii_type, original = PLANTED[row.placeholder]
        assert row.pii_type.value == pii_type
        assert row.original_hash == audit.salted_hash(original, salt)
        assert row.document_id == DOCUMENT_ID
        assert row.chunk_id == CHUNK_ID
        stored = (
            f"{row.trace_id} {row.document_id} {row.chunk_id} {row.pii_type.value} "
            f"{row.placeholder} {row.original_hash}"
        ).casefold()
        assert original.casefold() not in stored, "the original value reached the audit row"


@pytest.mark.usefixtures("clean_pii_db")
async def test_db_backed_scrub_reports_persisted_audit(
    pii_session_factory: async_sessionmaker[AsyncSession],
    person_names: list[str],
) -> None:
    """With a database the same payload claims persistence, and the rows exist."""
    await scrub(
        _audit_pack(),
        trace_id=TRACE,
        known_person_names=person_names,
        session_factory=pii_session_factory,
    )

    rows = await _redaction_rows(pii_session_factory, TRACE)
    events = list(await read_trace(pii_session_factory, TRACE))
    payloads = _event_payloads(events, "pii.scrubbed")

    assert rows, "no redaction rows were written"
    assert len(payloads) == 1
    assert payloads[0]["audit_persisted"] is True
    assert payloads[0]["redactions"] == len(rows)
