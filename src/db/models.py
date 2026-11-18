"""ORM models mirroring ``docs/05-data-model.md``.

The migration in ``src/db/migrations/versions`` is the source of truth for the
deployed schema; these models exist so later milestones can query the tables
with the ORM and so Alembic autogenerate stays useful.

Schemas: ``core`` (documents/chunks/sync state), ``mem`` (memory), ``audit``.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Computed,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import (
    text as sa_text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base

# Embedding dimension is fixed by migration 0002 (vector(768), matching the
# local nomic-embed-text model); changing the embedding model requires a new
# migration.
EMBEDDING_DIMENSIONS = 768


class Source(enum.StrEnum):
    """Where a document came from."""

    jira = "jira"
    notion = "notion"
    gmail = "gmail"


class DocType(enum.StrEnum):
    """Canonical document kinds."""

    issue = "issue"
    issue_comment = "issue_comment"
    page = "page"
    page_section = "page_section"
    email = "email"


class EntityType(enum.StrEnum):
    """Kinds of entities tracked in the registry."""

    project = "project"
    task = "task"
    person = "person"
    team = "team"
    page = "page"


class PiiType(enum.StrEnum):
    """Categories of redacted personal data."""

    person = "person"
    email = "email"
    phone = "phone"
    address = "address"
    api_key = "api_key"
    other = "other"


def _enum(py_enum: type[enum.Enum], name: str, schema: str) -> Enum:
    """Native Postgres enum type; created by the migration, not by the ORM."""
    return Enum(
        py_enum,
        name=name,
        schema=schema,
        native_enum=True,
        create_type=False,
        values_callable=lambda e: [member.value for member in e],
    )


class Document(Base):
    """One ingested source item (Jira issue/comment, Notion page/section, email)."""

    __tablename__ = "documents"
    __table_args__ = (
        # One *current* row per item; any number of historical (is_current=false)
        # rows. Partial unique index — see migration 0003 and docs/08-decisions.md.
        Index(
            "uq_documents_current_source_id",
            "source",
            "source_id",
            unique=True,
            postgresql_where=sa_text("is_current"),
        ),
        Index("ix_documents_source_source_id", "source", "source_id"),
        Index("ix_documents_source_updated_at", "source", sa_text("updated_at DESC")),
        {"schema": "core"},
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source: Mapped[Source] = mapped_column(_enum(Source, "source_type", "core"), nullable=False)
    source_id: Mapped[str] = mapped_column(Text, nullable=False)
    parent_source_id: Mapped[str | None] = mapped_column(Text)
    doc_type: Mapped[DocType] = mapped_column(_enum(DocType, "doc_type", "core"), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str | None] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    acl: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=sa_text("'{}'::jsonb")
    )
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    is_current: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sa_text("true")
    )


class Chunk(Base):
    """A retrievable slice of a document, with its embedding and FTS vector."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "seq", name="uq_chunks_document_id_seq"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_fts", "fts", postgresql_using="gin"),
        {"schema": "core"},
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("core.documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    fts: Mapped[str | None] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )
    meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=sa_text("'{}'::jsonb")
    )
    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)


class AnswerCache(Base):
    """Cached synthesized answers, keyed to the data version they were built from."""

    __tablename__ = "answer_cache"
    __table_args__ = (
        Index("ix_mem_answer_cache_question_normalized", "question_normalized"),
        {"schema": "mem"},
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question_normalized: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=sa_text("'[]'::jsonb")
    )
    evidence_hashes: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=sa_text("'{}'::text[]")
    )
    source_versions: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=sa_text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EntityRegistry(Base):
    """Cross-system aliases for projects, tasks, people, teams, pages."""

    __tablename__ = "entity_registry"
    __table_args__ = (
        UniqueConstraint("canonical_name", name="uq_mem_entity_registry_canonical_name"),
        Index("ix_mem_entity_registry_aliases_gin", "aliases", postgresql_using="gin"),
        {"schema": "mem"},
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    canonical_name: Mapped[str] = mapped_column(Text, nullable=False)
    entity_type: Mapped[EntityType] = mapped_column(
        _enum(EntityType, "entity_type", "mem"), nullable=False
    )
    aliases: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=sa_text("'{}'::text[]")
    )
    source_refs: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=sa_text("'{}'::jsonb")
    )
    # Entries below 0.5 confidence must not auto-resolve (docs/07-memory-design.md).
    confidence: Mapped[float] = mapped_column(Float, nullable=False, server_default=sa_text("1.0"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class SessionTurn(Base):
    """Last-N-turns working memory for a session."""

    __tablename__ = "session_turns"
    __table_args__ = (
        Index(
            "ix_mem_session_turns_session_id_created_at",
            "session_id",
            sa_text("created_at DESC"),
        ),
        {"schema": "mem"},
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    entity_hits: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=sa_text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Redaction(Base):
    """Audit trail of PII redactions applied to an evidence pack."""

    __tablename__ = "redactions"
    __table_args__ = (
        Index("ix_audit_redactions_trace_id", "trace_id"),
        {"schema": "audit"},
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id: Mapped[str] = mapped_column(Text, nullable=False)
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("core.documents.id", ondelete="SET NULL")
    )
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("core.chunks.id", ondelete="SET NULL")
    )
    pii_type: Mapped[PiiType] = mapped_column(_enum(PiiType, "pii_type", "audit"), nullable=False)
    placeholder: Mapped[str] = mapped_column(Text, nullable=False)
    # Salted hash of the original value: good for joins, not reversible.
    original_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class AuditEvent(Base):
    """Structured trace row written by every agent."""

    __tablename__ = "events"
    __table_args__ = (
        Index("ix_audit_events_trace_id", "trace_id"),
        Index("ix_audit_events_created_at", sa_text("created_at DESC")),
        {"schema": "audit"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(Text, nullable=False)
    agent: Mapped[str] = mapped_column(Text, nullable=False)
    event: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=sa_text("'{}'::jsonb")
    )
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class SyncState(Base):
    """Per-source sync cursor and last-run status."""

    __tablename__ = "sync_state"
    # Tuple form keeps RUF012 (mutable class attribute) quiet while matching
    # SQLAlchemy's "dict last" table-args convention.
    __table_args__ = ({"schema": "core"},)

    source: Mapped[str] = mapped_column(String(32), primary_key=True)
    watermark: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_status: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
