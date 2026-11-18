"""Test doubles and row builders for the retrieval suite.

Kept out of ``conftest.py`` so test modules can import them directly
(``from fakes import LexicalEmbedder``) — pytest puts this directory on
``sys.path`` because the suite has no ``__init__.py``.

Why a second fake embedder
--------------------------
``indexing.embedder.DeterministicFakeEmbedder`` produces hash-random vectors, so
every chunk is equally (dis)similar to a query. That is perfect for indexer
tests and useless for retrieval: the ANN ranking would be noise, and "the
relevant chunk ranks top-3" would assert luck.

:class:`LexicalEmbedder` is deterministic *and* semantically meaningful: it
hashes word counts into the 768 schema dimensions, so cosine similarity tracks
lexical overlap. No network, no model, stable across runs — but the vector list
and the FTS list now agree, which is what makes a hybrid-retrieval assertion
worth writing.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from db.models import EMBEDDING_DIMENSIONS, DocType, Source
from db.models import Chunk as ChunkRow
from db.models import Document as DocumentRow
from evals.fixtures import FixtureCorpus
from indexing.embedder import EmbeddingResult
from indexing.errors import EmbeddingServerError
from indexing.indexer import chunk_id_for
from indexing.tokens import estimate_tokens, sha256_text
from orchestrator.pipeline import EvidencePack
from retrieval.search import RetrieveRequest, retrieve

#: Namespace for synthetic document ids: same key → same row id across runs.
SYNTHETIC_ID_NAMESPACE = uuid.UUID("1f4c8e02-6b7d-4a95-b3e8-2c0d5a7f9b61")

#: Identities used by the suite. Phoenix documents carry this user in their
#: mirrored ACL; Atlas documents carry ATLAS_USER; STRANGER_USER appears in none.
PHOENIX_USER = "priya.raghunathan@northwind.example"
ATLAS_USER = "grace.adeyemi@northwind.example"
STRANGER_USER = "stranger@northwind.example"

#: Fixed request clock matching the corpus manifest (today = 2026-11-17), so
#: recency and staleness assertions never depend on when the suite runs.
CORPUS_NOW = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class LexicalEmbedder:
    """Hashed bag-of-words vectors: cosine similarity ≈ shared vocabulary."""

    dimensions: int = EMBEDDING_DIMENSIONS
    model: str = "test-lexical"
    max_batch_size: int = 128
    calls: list[int] = field(default_factory=list)

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        self.calls.append(len(texts))
        vectors = [self.vector(text) for text in texts]
        return EmbeddingResult(
            vectors=vectors,
            tokens=sum(estimate_tokens(text) for text in texts),
            batches=1 if texts else 0,
        )

    async def aclose(self) -> None:
        return None

    def vector(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        tokens = _WORD_RE.findall(text.casefold())
        if not tokens:
            # Never emit the zero vector: pgvector's cosine distance is NaN for
            # it, which would poison the ANN ordering.
            tokens = [f"empty:{text}"]
        for token in tokens:
            bucket = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % self.dimensions
            values[bucket] += 1.0
        norm = sum(value * value for value in values) ** 0.5
        return [value / norm for value in values]


@dataclass
class FailingEmbedder:
    """Always raises, to exercise the FTS-only fallback."""

    dimensions: int = EMBEDDING_DIMENSIONS
    model: str = "failing-fake"

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        raise EmbeddingServerError("simulated embedding provider outage")

    async def aclose(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class IndexedCorpus:
    """The fixture corpus, seeded and chunked/embedded in the scratch database."""

    factory: async_sessionmaker[AsyncSession]
    corpus: FixtureCorpus
    embedder: Any
    documents: int
    chunks: int

    async def retrieve(
        self,
        req: RetrieveRequest,
        *,
        user_id: str = PHOENIX_USER,
        **kwargs: Any,
    ) -> EvidencePack:
        """Call ``retrieve`` against this database with the offline embedder."""
        embedder = kwargs.pop("embedder", self.embedder)
        return await retrieve(
            req,
            user_id=user_id,
            session_factory=self.factory,
            embedder=embedder,
            **kwargs,
        )


async def insert_document(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    source_id: str,
    title: str,
    content: str | None = None,
    source: str = "jira",
    doc_type: str = "issue",
    updated_at: datetime | None = None,
    acl: Mapping[str, list[str]] | None = None,
    is_current: bool = True,
    parent_source_id: str | None = None,
) -> uuid.UUID:
    """Insert one synthetic document; returns its deterministic id."""
    document_id = uuid.uuid5(SYNTHETIC_ID_NAMESPACE, f"{source}:{source_id}")
    body = content if content is not None else f"{title}\n\n{source_id} body text."
    stamp = updated_at or datetime(2026, 11, 17, 9, 0, tzinfo=UTC)
    row: dict[str, Any] = {
        "id": document_id,
        "source": Source(source),
        "source_id": source_id,
        "parent_source_id": parent_source_id,
        "doc_type": DocType(doc_type),
        "title": title,
        "content": body,
        "url": f"https://northwind.atlassian.example/browse/{source_id}",
        "author": "Test Author",
        "created_at": stamp,
        "updated_at": stamp,
        "fetched_at": datetime.now(UTC),
        "acl": dict(acl or {}),
        "content_hash": sha256_text(body),
        "is_current": is_current,
    }
    async with session_factory() as session, session.begin():
        await session.execute(insert(DocumentRow).values(**row))
    return document_id


async def insert_chunk(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    document_id: uuid.UUID,
    text: str,
    seq: int = 0,
    heading_path: Sequence[str] = ("description",),
    section: str = "description",
    embedding: Sequence[float] | None = None,
    labels: Sequence[str] = (),
) -> uuid.UUID:
    """Insert one chunk for a synthetic document; returns its deterministic id."""
    chunk_id = chunk_id_for(document_id, seq)
    meta: dict[str, Any] = {
        "source": "jira",
        "source_id": "",
        "doc_type": "issue",
        "title": "",
        "heading_path": list(heading_path),
        "section": section,
        "part": 0,
        "labels": list(labels),
        "token_count": estimate_tokens(text),
        "content_hash": sha256_text(text),
        "entity_ids": [],
    }
    row: dict[str, Any] = {
        "id": chunk_id,
        "document_id": document_id,
        "seq": seq,
        "text": text,
        "embedding": list(embedding) if embedding is not None else None,
        "meta": meta,
        "embedded_at": datetime.now(UTC) if embedding is not None else None,
        "content_hash": sha256_text(text),
    }
    async with session_factory() as session, session.begin():
        await session.execute(insert(ChunkRow).values(**row))
    return chunk_id


async def set_document_acl(
    session_factory: async_sessionmaker[AsyncSession],
    document_id: uuid.UUID,
    acl: Mapping[str, list[str]],
) -> None:
    async with session_factory() as session, session.begin():
        await session.execute(
            update(DocumentRow).where(DocumentRow.id == document_id).values(acl=dict(acl))
        )


async def retire_document(
    session_factory: async_sessionmaker[AsyncSession], document_id: uuid.UUID
) -> None:
    """Mark a document superseded *without* deleting its chunks (gap #1)."""
    async with session_factory() as session, session.begin():
        await session.execute(
            update(DocumentRow).where(DocumentRow.id == document_id).values(is_current=False)
        )


async def count_chunks(
    session_factory: async_sessionmaker[AsyncSession], document_id: uuid.UUID
) -> int:
    async with session_factory() as session:
        result = await session.execute(
            select(ChunkRow.id).where(ChunkRow.document_id == document_id)
        )
        return len(result.all())


__all__ = [
    "ATLAS_USER",
    "CORPUS_NOW",
    "PHOENIX_USER",
    "STRANGER_USER",
    "FailingEmbedder",
    "IndexedCorpus",
    "LexicalEmbedder",
    "count_chunks",
    "insert_chunk",
    "insert_document",
    "retire_document",
    "set_document_acl",
]
