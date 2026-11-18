"""Scratch-database fixtures for retrieval tests.

Retrieval needs the real schema (pgvector operator, generated tsvector, jsonb
ACLs), so these tests run against a throwaway database derived from
``DATABASE_URL`` — never the dev database. The database name must end in
``_retrieval_test``; anything else is refused rather than truncated.

Provision (once)::

    sudo -u postgres createdb -O rag agentic_rag_retrieval_test
    sudo -u postgres psql -d agentic_rag_retrieval_test \\
        -c "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_trgm;"

Migrations are brought to head by a session fixture (not pinned to a revision),
so the suite follows schema changes automatically. Without ``DATABASE_URL`` (or
with the scratch database missing) the DB tests skip with an explicit reason.
The pure-ranker and SQL-compilation tests never need a database and always run.
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
from fakes import IndexedCorpus, LexicalEmbedder
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from evals.fixtures import fixture_document_ids, load_fixture_corpus, seed_fixture_corpus
from indexing.indexer import Indexer, IndexRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_DB = "agentic_rag_retrieval_test"
SCRATCH_SUFFIX = "_retrieval_test"


def _alembic_config() -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "src/db/migrations"))
    return config


def _scratch_url() -> str | None:
    explicit = os.environ.get("RETRIEVAL_TEST_DATABASE_URL")
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
    if not name.endswith(SCRATCH_SUFFIX):
        raise RuntimeError(
            f"refusing to run retrieval tests against {name!r}: expected a database name "
            f"ending in {SCRATCH_SUFFIX!r} (set RETRIEVAL_TEST_DATABASE_URL to override)"
        )
    return url


@pytest.fixture(scope="session")
def retrieval_db_url() -> str:
    url = _scratch_url()
    if not url:
        pytest.skip("DATABASE_URL is not set — skipping scratch-database retrieval tests")
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
def retrieval_schema(retrieval_db_url: str) -> str:
    """Bring the scratch database to Alembic head (idempotent)."""
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = retrieval_db_url
    try:
        command.upgrade(_alembic_config(), "head")
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous
        from config import get_settings

        get_settings.cache_clear()
    return retrieval_db_url


@pytest.fixture(scope="session")
def retrieval_session_factory(
    retrieval_schema: str,
) -> Iterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(retrieval_schema, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    asyncio.run(engine.dispose())


@pytest.fixture
async def clean_retrieval_db(
    retrieval_session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[None]:
    """Empty documents (cascades to chunks) and audit events before each test."""
    async with retrieval_session_factory() as session, session.begin():
        await session.execute(text("TRUNCATE core.documents CASCADE"))
        await session.execute(text("TRUNCATE audit.events"))
    yield


@pytest.fixture
async def indexed_corpus(
    clean_retrieval_db: None,
    retrieval_session_factory: async_sessionmaker[AsyncSession],
) -> IndexedCorpus:
    """Seed ``fixtures/corpus`` and index it with the offline lexical embedder."""
    corpus = load_fixture_corpus()
    seeded = await seed_fixture_corpus(retrieval_session_factory, corpus)
    embedder = LexicalEmbedder()
    indexer = Indexer(session_factory=retrieval_session_factory, embedder=embedder)
    await indexer.run(IndexRequest(document_ids=list(fixture_document_ids(corpus))))

    async with retrieval_session_factory() as session:
        chunk_count = len((await session.execute(text("SELECT id FROM core.chunks"))).all())
    return IndexedCorpus(
        factory=retrieval_session_factory,
        corpus=corpus,
        embedder=embedder,
        documents=seeded.documents_seeded,
        chunks=chunk_count,
    )
