"""Initial schema: core / mem / audit (docs/05-data-model.md).

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-11-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIMENSIONS = 1536

source_type = postgresql.ENUM(
    "jira", "notion", "gmail", name="source_type", schema="core", create_type=False
)
doc_type = postgresql.ENUM(
    "issue",
    "issue_comment",
    "page",
    "page_section",
    "email",
    name="doc_type",
    schema="core",
    create_type=False,
)
entity_type = postgresql.ENUM(
    "project",
    "task",
    "person",
    "team",
    "page",
    name="entity_type",
    schema="mem",
    create_type=False,
)
pii_type = postgresql.ENUM(
    "person",
    "email",
    "phone",
    "address",
    "api_key",
    "other",
    name="pii_type",
    schema="audit",
    create_type=False,
)


def upgrade() -> None:
    # `vector` and `pg_trgm` are not trusted extensions: on a stock install the
    # migrating role needs superuser rights (or the extensions must already be
    # created by provisioning — see HANDOFF.md). With them present these
    # statements are no-ops, so a non-superuser app role can migrate normally.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    # gen_random_uuid() is built into Postgres 13+, so no pgcrypto dependency.

    op.execute("CREATE SCHEMA IF NOT EXISTS core")
    op.execute("CREATE SCHEMA IF NOT EXISTS mem")
    op.execute("CREATE SCHEMA IF NOT EXISTS audit")

    op.execute("CREATE TYPE core.source_type AS ENUM ('jira', 'notion', 'gmail')")
    op.execute(
        "CREATE TYPE core.doc_type AS ENUM "
        "('issue', 'issue_comment', 'page', 'page_section', 'email')"
    )
    op.execute(
        "CREATE TYPE mem.entity_type AS ENUM ('project', 'task', 'person', 'team', 'page')"
    )
    op.execute(
        "CREATE TYPE audit.pii_type AS ENUM "
        "('person', 'email', 'phone', 'address', 'api_key', 'other')"
    )

    # --- core.documents -------------------------------------------------
    op.create_table(
        "documents",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("source", source_type, nullable=False),
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("parent_source_id", sa.Text(), nullable=True),
        sa.Column("doc_type", doc_type, nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("author", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "acl",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column(
            "is_current", sa.Boolean(), nullable=False, server_default=sa.text("true")
        ),
        sa.UniqueConstraint(
            "source", "source_id", "is_current", name="uq_documents_source_source_id_is_current"
        ),
        schema="core",
    )
    op.create_index(
        "ix_documents_source_updated_at",
        "documents",
        ["source", sa.text("updated_at DESC")],
        schema="core",
    )

    # --- core.chunks ----------------------------------------------------
    # Written as raw DDL because the vector dimension and the generated tsvector
    # have no clean expression in Alembic's column DSL.
    op.execute(
        f"""
        CREATE TABLE core.chunks (
            id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id  uuid NOT NULL REFERENCES core.documents(id) ON DELETE CASCADE,
            seq          integer NOT NULL,
            "text"       text NOT NULL,
            embedding    vector({EMBEDDING_DIMENSIONS}),
            fts          tsvector GENERATED ALWAYS AS (to_tsvector('english', "text")) STORED,
            meta         jsonb NOT NULL DEFAULT '{{}}'::jsonb,
            embedded_at  timestamptz,
            content_hash text NOT NULL,
            CONSTRAINT uq_chunks_document_id_seq UNIQUE (document_id, seq)
        )
        """
    )
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        schema="core",
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index(
        "ix_chunks_fts", "chunks", ["fts"], schema="core", postgresql_using="gin"
    )

    # --- core.sync_state ------------------------------------------------
    op.create_table(
        "sync_state",
        sa.Column("source", sa.String(length=32), primary_key=True),
        sa.Column("watermark", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_status", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        schema="core",
    )

    # --- mem.answer_cache ----------------------------------------------
    op.create_table(
        "answer_cache",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("question_normalized", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column(
            "citations",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "evidence_hashes",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
        sa.Column(
            "source_versions",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        schema="mem",
    )
    op.create_index(
        "ix_mem_answer_cache_question_normalized",
        "answer_cache",
        ["question_normalized"],
        schema="mem",
    )

    # --- mem.entity_registry -------------------------------------------
    op.create_table(
        "entity_registry",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("canonical_name", sa.Text(), nullable=False),
        sa.Column("entity_type", entity_type, nullable=False),
        sa.Column(
            "aliases",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
        sa.Column(
            "source_refs",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "confidence",
            sa.REAL(),
            nullable=False,
            server_default=sa.text("1.0"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("canonical_name", name="uq_mem_entity_registry_canonical_name"),
        schema="mem",
    )
    op.create_index(
        "ix_mem_entity_registry_aliases_gin",
        "entity_registry",
        ["aliases"],
        schema="mem",
        postgresql_using="gin",
    )

    # --- mem.session_turns ---------------------------------------------
    op.create_table(
        "session_turns",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "entity_hits",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        schema="mem",
    )
    op.create_index(
        "ix_mem_session_turns_session_id_created_at",
        "session_turns",
        ["session_id", sa.text("created_at DESC")],
        schema="mem",
    )

    # --- audit.redactions ----------------------------------------------
    op.create_table(
        "redactions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("trace_id", sa.Text(), nullable=False),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("core.documents.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "chunk_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("core.chunks.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("pii_type", pii_type, nullable=False),
        sa.Column("placeholder", sa.Text(), nullable=False),
        sa.Column("original_hash", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        schema="audit",
    )
    op.create_index(
        "ix_audit_redactions_trace_id", "redactions", ["trace_id"], schema="audit"
    )

    # --- audit.events ---------------------------------------------------
    op.create_table(
        "events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("trace_id", sa.Text(), nullable=False),
        sa.Column("agent", sa.Text(), nullable=False),
        sa.Column("event", sa.Text(), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        schema="audit",
    )
    op.create_index("ix_audit_events_trace_id", "events", ["trace_id"], schema="audit")
    op.create_index(
        "ix_audit_events_created_at", "events", [sa.text("created_at DESC")], schema="audit"
    )


def downgrade() -> None:
    op.drop_table("events", schema="audit")
    op.drop_table("redactions", schema="audit")
    op.drop_table("session_turns", schema="mem")
    op.drop_table("entity_registry", schema="mem")
    op.drop_table("answer_cache", schema="mem")
    op.drop_table("sync_state", schema="core")
    op.drop_table("chunks", schema="core")
    op.drop_table("documents", schema="core")

    op.execute("DROP TYPE IF EXISTS audit.pii_type")
    op.execute("DROP TYPE IF EXISTS mem.entity_type")
    op.execute("DROP TYPE IF EXISTS core.doc_type")
    op.execute("DROP TYPE IF EXISTS core.source_type")

    op.execute("DROP SCHEMA IF EXISTS audit")
    op.execute("DROP SCHEMA IF EXISTS mem")
    op.execute("DROP SCHEMA IF EXISTS core")
    # Extensions are left installed: other schemas may depend on them.
