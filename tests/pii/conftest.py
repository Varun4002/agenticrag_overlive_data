"""Fixtures for the PII scrubber tests.

Planted values come from the shared corpus manifest (``fixtures/corpus``) — the
same list the eval agent asserts on — so the scrubber is tested against the data
the rest of the system uses.

Postgres-backed audit tests run against a throwaway database named
``agentic_rag_pii_test`` (override with ``PII_TEST_DATABASE_URL``), brought to
Alembic head by the session fixture. Any other database name is refused: these
tests truncate audit and core tables.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest
from alembic import command
from alembic.config import Config
from pii_fixtures import CHUNK_ID, DOCUMENT_ID, FIXED_NOW, corpus_pack, load_manifest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from orchestrator.pipeline import EvidencePack
from pii import analyzer as pii_analyzer
from pii.errors import PiiUnavailableError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_DB = "agentic_rag_pii_test"


@pytest.fixture(scope="session")
def manifest() -> dict[str, object]:
    return load_manifest()


@pytest.fixture(scope="session")
def planted_pii(manifest: dict[str, object]) -> list[str]:
    """Every raw value the corpus plants, across all kinds."""
    pii = manifest["pii"]
    assert isinstance(pii, dict)
    kinds = pii["kinds"]
    assert isinstance(kinds, dict)
    return [value for values in kinds.values() for value in values]


@pytest.fixture(scope="session")
def person_names(manifest: dict[str, object]) -> list[str]:
    """The registry person list the scrubber is given (Agent 08 stand-in)."""
    pii = manifest["pii"]
    assert isinstance(pii, dict)
    kinds = pii["kinds"]
    assert isinstance(kinds, dict)
    names: list[str] = kinds["person"]
    return names


@pytest.fixture(scope="session")
def false_positive_name(manifest: dict[str, object]) -> str:
    section = manifest["false_positive_names"]
    assert isinstance(section, dict)
    items = section["items"]
    assert isinstance(items, list)
    value: str = items[0]["value"]
    return value


@pytest.fixture(scope="session")
def engines() -> Iterator[None]:
    """Skip (never fail) when Presidio or the spaCy model is unavailable."""
    try:
        pii_analyzer.get_engines()
    except PiiUnavailableError as exc:  # pragma: no cover - the model is installed here
        pytest.skip(f"PII analyzer unavailable: {exc}")
    yield


@pytest.fixture(scope="session")
def corpus_evidence_pack() -> EvidencePack:
    return corpus_pack()


def _alembic_config() -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "src/db/migrations"))
    return config


def _scratch_url() -> str | None:
    explicit = os.environ.get("PII_TEST_DATABASE_URL")
    if explicit:
        return explicit
    base = os.environ.get("DATABASE_URL")
    if not base:
        return None
    parts = urlsplit(base)
    if not parts.path.strip("/"):
        return None
    return urlunsplit(parts._replace(path="/" + SCRATCH_DB))


@pytest.fixture(scope="session")
def pii_db_url() -> str:
    url = _scratch_url()
    if not url:
        pytest.skip(
            "DATABASE_URL is not set — skipping scratch-database PII audit tests. "
            "Prepare one with: sudo -u postgres createdb -O rag agentic_rag_pii_test"
        )
    name = urlsplit(url).path.lstrip("/")
    if not name.endswith("_pii_test"):
        pytest.fail(
            f"refusing to run PII tests against {name!r}: the database name must end in "
            "'_pii_test' (set PII_TEST_DATABASE_URL to override) — these tests truncate tables"
        )
    if os.environ.get("DATABASE_URL"):
        dev_name = urlsplit(os.environ["DATABASE_URL"]).path.lstrip("/")
        if dev_name == name:
            pytest.fail(f"scratch database {name!r} would be the dev database")
    try:
        engine = create_engine(url, connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
    except Exception as exc:
        pytest.skip(f"scratch database {SCRATCH_DB} unreachable ({type(exc).__name__}: {exc})")
    return url


@pytest.fixture(scope="session")
def pii_schema(pii_db_url: str) -> str:
    """Bring the scratch database to Alembic head (idempotent)."""
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = pii_db_url
    try:
        command.upgrade(_alembic_config(), "head")
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous
        from config import get_settings

        get_settings.cache_clear()
    return pii_db_url


@pytest.fixture(scope="session")
def pii_engine(pii_schema: str) -> Iterator[Engine]:
    engine = create_engine(pii_schema, connect_args={"connect_timeout": 3})
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def pii_session_factory(pii_schema: str) -> Iterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(pii_schema, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    asyncio.run(engine.dispose())


@pytest.fixture
def clean_pii_db(pii_engine: Engine) -> Iterator[None]:
    """Empty the audit/core tables and seed one document + chunk for FK-valid rows."""
    with pii_engine.begin() as conn:
        conn.execute(
            text("TRUNCATE audit.redactions, audit.events, core.chunks, core.documents CASCADE")
        )
        conn.execute(
            text(
                "INSERT INTO core.documents "
                "(id, source, source_id, doc_type, title, content, created_at, updated_at, "
                " content_hash) VALUES "
                "(:id, 'notion', 'pii-test-page', 'page', 'Team Directory', "
                " 'fixture document for PII audit tests', :now, :now, :hash)"
            ),
            {"id": DOCUMENT_ID, "now": FIXED_NOW, "hash": "0" * 64},
        )
        conn.execute(
            text(
                "INSERT INTO core.chunks (id, document_id, seq, text, content_hash) "
                "VALUES (:id, :document_id, 0, 'fixture chunk', :hash)"
            ),
            {"id": CHUNK_ID, "document_id": DOCUMENT_ID, "hash": "1" * 64},
        )
    yield
