"""Scratch-database fixtures for eval / observability tests.

These tests need the real schema (``audit.events``, jsonb, Postgres), so they
run against a throwaway database derived from ``DATABASE_URL`` — never the dev
database. The database name must end in ``_eval_test``; anything else is
refused rather than truncated.

Provision (once)::

    sudo -u postgres createdb -O rag agentic_rag_eval_test
    sudo -u postgres psql -d agentic_rag_eval_test \\
        -c "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_trgm;"

Without ``DATABASE_URL`` (or with the scratch DB missing) these tests skip with
an explicit reason instead of failing.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, delete, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from db.models import AuditEvent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_DB = "agentic_rag_eval_test"


def _alembic_config() -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "src/db/migrations"))
    return config


def _scratch_url() -> str | None:
    explicit = os.environ.get("EVAL_TEST_DATABASE_URL")
    if explicit:
        return explicit
    base = os.environ.get("DATABASE_URL")
    if not base:
        return None
    parts = urlsplit(base)
    if not parts.path.strip("/"):
        return None
    return urlunsplit(parts._replace(path="/" + SCRATCH_DB))


def _require_scratch(url: str) -> str:
    name = urlsplit(url).path.lstrip("/")
    if not name.endswith("_eval_test"):
        raise RuntimeError(
            f"refusing to run eval tests against {name!r}: expected a database name "
            "ending in '_eval_test' (set EVAL_TEST_DATABASE_URL to override)"
        )
    return url


@pytest.fixture(scope="session")
def eval_db_url() -> str:
    url = _scratch_url()
    if not url:
        pytest.skip("DATABASE_URL is not set — skipping scratch-database trace tests")
    _require_scratch(url)
    try:
        engine = create_engine(url, connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
    except Exception as exc:
        pytest.skip(
            f"scratch database {SCRATCH_DB} unreachable ({type(exc).__name__}); create it "
            f"with: sudo -u postgres createdb -O rag {SCRATCH_DB} (plus the vector/pg_trgm "
            "extensions)"
        )
    return url


@pytest.fixture(scope="session")
def eval_schema(eval_db_url: str) -> str:
    """Bring the scratch database to Alembic head (idempotent)."""
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = eval_db_url
    try:
        command.upgrade(_alembic_config(), "head")
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous
        from config import get_settings

        get_settings.cache_clear()
    return eval_db_url


@pytest.fixture(scope="session")
def eval_session_factory(eval_schema: str) -> Iterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(eval_schema, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    asyncio.run(engine.dispose())


@pytest.fixture
async def clean_events(
    eval_session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[None]:
    """Empty ``audit.events`` in the scratch DB before each test."""
    async with eval_session_factory() as session, session.begin():
        await session.execute(delete(AuditEvent))
    yield
