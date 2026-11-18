"""Audit trail for redactions: ``audit.redactions`` rows + ``pii.scrubbed`` event.

Every redaction writes one row: which request (``trace_id``), where (document /
chunk), what class (``pii_type``), which placeholder, and a **salted** hash of
the original value. The original value is never stored, never logged, and never
put in an event payload; the hash exists only so a repeat occurrence of the same
value can be recognised later (and so an operator can confirm that a document
was redacted without recovering its content).

Salt comes from ``PII_HASH_SALT``. The default is a documented dev-only value:
with a shared, known salt the hashes of common names/emails are dictionary-
attackable, so a real deployment must set its own.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from config import get_settings
from db.models import PiiType, Redaction
from pii.placeholders import normalize_value

logger = logging.getLogger(__name__)

#: Dev-only fallback (``PII_HASH_SALT``). Never use in a real deployment.
DEV_ONLY_SALT = "dev-only-pii-salt-change-me"


class RedactionAuditError(RuntimeError):
    """Redaction rows were built but could not be persisted."""


def resolve_salt() -> str:
    """The configured hash salt, warning once when the dev default is in use."""
    salt = get_settings().pii_hash_salt or DEV_ONLY_SALT
    if salt == DEV_ONLY_SALT:
        logger.warning(
            "PII_HASH_SALT is not set: using the dev-only salt. Set a deployment-specific "
            "salt before redaction hashes leave a dev machine."
        )
    return salt


def salted_hash(value: str, salt: str) -> str:
    """Salted sha256 of the normalised original value (hex)."""
    normalized = normalize_value(value)
    return hashlib.sha256(f"{salt}\x00{normalized}".encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class RedactionRecord:
    """One redaction, ready for ``audit.redactions``."""

    chunk_id: str | None
    document_id: str | None
    pii_type: str
    placeholder: str
    original_hash: str


def _as_uuid(value: str | None) -> uuid.UUID | None:
    """Parse an id for the UUID-typed audit columns; ``None`` when unusable.

    ``EvidenceChunk`` ids are strings. Retrieval produces UUID strings, which
    map straight onto ``audit.redactions.document_id/chunk_id``; a non-UUID id
    (hand-built pack, external id) leaves the column NULL rather than failing
    the whole redaction batch.
    """
    if not value:
        return None
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


class RedactionWriter(Protocol):
    """Persists redaction rows; never raises, reports success instead."""

    async def write(self, trace_id: str, records: Sequence[RedactionRecord]) -> bool: ...


class NullRedactionWriter:
    """Writer used when no database is configured (dev/tests)."""

    async def write(self, trace_id: str, records: Sequence[RedactionRecord]) -> bool:
        logger.debug(
            "redaction audit dropped (%d rows, trace %s): no database",
            len(records),
            trace_id,
        )
        return True


class DatabaseRedactionWriter:
    """Insert redaction rows in one short transaction."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self.failed_writes = 0

    async def write(self, trace_id: str, records: Sequence[RedactionRecord]) -> bool:
        if not records:
            return True
        rows = [
            {
                "trace_id": trace_id,
                "document_id": _as_uuid(record.document_id),
                "chunk_id": _as_uuid(record.chunk_id),
                "pii_type": PiiType(record.pii_type),
                "placeholder": record.placeholder,
                "original_hash": record.original_hash,
            }
            for record in records
        ]
        try:
            async with self._session_factory() as session, session.begin():
                await session.execute(insert(Redaction), rows)
        except Exception as exc:
            self.failed_writes += 1
            logger.warning(
                "failed to persist %d redaction rows for trace %s: %s",
                len(rows),
                trace_id,
                exc,
            )
            return False
        return True


def make_redaction_writer(
    session_factory: async_sessionmaker[AsyncSession] | None,
) -> RedactionWriter:
    """Database-backed writer when a database is configured, else a no-op."""
    return DatabaseRedactionWriter(session_factory) if session_factory else NullRedactionWriter()
