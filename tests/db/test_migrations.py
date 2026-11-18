"""Database smoke tests.

These are skipped — never failed — when ``DATABASE_URL`` is unset or the server
is unreachable, so the suite stays runnable on a machine without Postgres.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from config import get_settings
from db import session as db_session
from db.tracing import DatabaseEventSink, EventRecord, new_trace_id, read_trace

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_TABLES = {
    ("core", "documents"),
    ("core", "chunks"),
    ("core", "sync_state"),
    ("mem", "answer_cache"),
    ("mem", "entity_registry"),
    ("mem", "session_turns"),
    ("audit", "redactions"),
    ("audit", "events"),
}


def _sync_url() -> str | None:
    return os.environ.get("DATABASE_URL") or get_settings().database_url


@pytest.fixture(scope="module")
def sync_engine() -> Engine:
    """A live sync engine, or skip the whole module with a clear reason."""
    url = _sync_url()
    if not url:
        pytest.skip(
            "DATABASE_URL is not set (see .env.example) — skipping database smoke tests"
        )
    try:
        engine = create_engine(url, connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(
            f"DATABASE_URL is unreachable ({type(exc).__name__}: {exc}) — "
            "skipping database smoke tests"
        )
    yield engine
    engine.dispose()


def _alembic_config() -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "src/db/migrations"))
    return config


def test_migrations_upgrade_to_head(sync_engine: Engine) -> None:
    command.upgrade(_alembic_config(), "head")

    inspector = inspect(sync_engine)
    present = {
        (schema, table)
        for schema in ("core", "mem", "audit")
        for table in inspector.get_table_names(schema=schema)
    }
    missing = EXPECTED_TABLES - present
    assert not missing, f"missing tables after upgrade: {sorted(missing)}"


def test_migrations_are_at_head(sync_engine: Engine) -> None:
    """A second upgrade must be a no-op: proves the revision chain is linear."""
    command.upgrade(_alembic_config(), "head")
    with sync_engine.connect() as conn:
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
    assert version == "0003_documents_partial_unique"


def test_required_extensions_and_types(sync_engine: Engine) -> None:
    with sync_engine.connect() as conn:
        extensions = set(conn.execute(text("SELECT extname FROM pg_extension")).scalars())
        enum_types = set(
            conn.execute(
                text(
                    "SELECT n.nspname || '.' || t.typname FROM pg_type t "
                    "JOIN pg_namespace n ON n.oid = t.typnamespace "
                    "WHERE t.typtype = 'e'"
                )
            ).scalars()
        )
    assert {"vector", "pg_trgm"} <= extensions
    assert {
        "core.source_type",
        "core.doc_type",
        "mem.entity_type",
        "audit.pii_type",
    } <= enum_types


def test_chunks_has_vector_and_generated_fts(sync_engine: Engine) -> None:
    with sync_engine.connect() as conn:
        embedding_type = conn.execute(
            text(
                "SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
                "WHERE a.attrelid = 'core.chunks'::regclass AND a.attname = 'embedding'"
            )
        ).scalar_one()
        fts_type = conn.execute(
            text(
                "SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
                "WHERE a.attrelid = 'core.chunks'::regclass AND a.attname = 'fts'"
            )
        ).scalar_one()
        generated = conn.execute(
            text(
                "SELECT a.attgenerated FROM pg_attribute a "
                "WHERE a.attrelid = 'core.chunks'::regclass AND a.attname = 'fts'"
            )
        ).scalar_one()

    assert embedding_type == "vector(768)"
    assert fts_type == "tsvector"
    assert generated == "s"  # STORED generated column


def test_required_indexes_exist(sync_engine: Engine) -> None:
    inspector = inspect(sync_engine)

    def index_names(schema: str, table: str) -> set[str]:
        return {index["name"] for index in inspector.get_indexes(table, schema=schema)}

    assert "ix_documents_source_updated_at" in index_names("core", "documents")
    chunks = index_names("core", "chunks")
    assert "ix_chunks_embedding_hnsw" in chunks
    assert "ix_chunks_fts" in chunks
    assert "ix_mem_answer_cache_question_normalized" in index_names("mem", "answer_cache")
    assert "ix_mem_entity_registry_aliases_gin" in index_names("mem", "entity_registry")
    assert "ix_audit_events_trace_id" in index_names("audit", "events")
    assert "ix_audit_events_created_at" in index_names("audit", "events")


async def test_trace_event_roundtrip(sync_engine: Engine) -> None:
    """Proves the real (database) sink path used by /ask works end to end."""
    get_settings.cache_clear()
    db_session.reset_engine_state()
    factory = db_session.get_session_factory()
    if factory is None:
        pytest.skip("DATABASE_URL is not set — skipping trace roundtrip")

    trace_id = new_trace_id()
    sink = DatabaseEventSink(factory)
    await sink.write(
        EventRecord(
            trace_id=trace_id,
            agent="test",
            event="migration_smoke.probe",
            payload={"ok": True},
            latency_ms=3,
        )
    )
    rows = await read_trace(factory, trace_id)
    await db_session.dispose_engine()

    assert len(rows) == 1
    row = rows[0]
    payload: Any = row.payload
    assert row.agent == "test"
    assert row.event == "migration_smoke.probe"
    assert payload == {"ok": True}
    assert row.latency_ms == 3
