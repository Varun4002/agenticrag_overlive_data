"""The indexer: canonical documents → ``core.chunks`` (text + vector + FTS).

Public contract (frozen, ``docs/agents/02-indexer-agent.md``)::

    async def index(req: IndexRequest) -> IndexResult

Everything else here is implementation. The indexer is the **only writer of
``core.chunks``**, which is what makes these guarantees possible:

- **Idempotent.** Rows are keyed on ``(document_id, seq)`` and chunk ids are
  deterministic (``uuid5`` of document + seq), so re-indexing an unchanged
  document writes nothing and keeps the same ids.
- **No wasted embedding calls.** A chunk is skipped when its ``content_hash``
  and ``meta`` are unchanged *and* it already has a vector. Chunks that only
  need metadata refreshed are updated without touching the embedding column.
  Identical text inside one run is embedded once and the vector reused.
- **Transaction per batch.** Each embedding batch is committed on its own, so a
  provider failure mid-run leaves earlier batches intact and never writes half
  a chunk. Remaining batches are attempted; failures are logged and reported.
- **No stale retrieval.** Chunks of a superseded document
  (``is_current = false``) are deleted, and trailing chunks left behind by a
  document that shrank are deleted too. The schema has no ``is_current`` flag
  on chunks, so deletion is the only way stale text stays unretrievable.

Audit: one ``indexer.completed`` event per run (with a generated ``trace_id``
when the caller did not supply one).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from pydantic import BaseModel
from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from db.models import EMBEDDING_DIMENSIONS
from db.models import Chunk as ChunkRow
from db.models import Document as DocumentRow
from db.session import get_session_factory
from db.tracing import EventSink, Tracer, make_sink
from indexing.chunker import DEFAULT_CHUNKER, Chunk, Chunker, SourceDocument
from indexing.embedder import Embedder, build_embedder
from indexing.errors import EmbeddingError, IndexingConfigError
from indexing.tokens import OVERSIZED_DOCUMENT_TOKENS, estimate_tokens

logger = logging.getLogger(__name__)

AGENT = "indexer"

#: Namespace for deterministic chunk ids. Changing it re-keys every chunk, so
#: it is a constant on purpose.
CHUNK_ID_NAMESPACE = uuid.UUID("6f2a1c34-9d1e-4f7b-8c25-1b0e5d9a7f43")


class IndexRequest(BaseModel):
    """What to index. The caller decides; the indexer never scans on its own."""

    document_ids: list[uuid.UUID]
    force_reembed: bool = False


class IndexResult(BaseModel):
    """Outcome of one indexing run."""

    chunks_written: int
    chunks_skipped: int  # unchanged content_hash
    embed_tokens: int


def chunk_id_for(document_id: uuid.UUID | str, seq: int) -> uuid.UUID:
    """Deterministic chunk id: same document + position → same row id."""
    return uuid.uuid5(CHUNK_ID_NAMESPACE, f"chunk:{document_id}:{seq}")


@dataclass(frozen=True, slots=True)
class _PriorChunk:
    """What the database already has for one ``(document_id, seq)``."""

    content_hash: str
    has_embedding: bool
    meta: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _Pending:
    """A chunk that still needs a vector and a write."""

    document_id: uuid.UUID
    chunk: Chunk


@dataclass(slots=True)
class Indexer:
    """Dependency-injected indexer. ``indexing.index`` builds the default one."""

    session_factory: async_sessionmaker[AsyncSession]
    embedder: Embedder
    chunker: Chunker = DEFAULT_CHUNKER
    #: Chunks per embedding request/transaction. Capped at ``embedder.max_batch_size``.
    batch_size: int | None = None
    #: Trace sink; defaults to the database when one is configured.
    event_sink: EventSink | None = None
    #: Optional M4 hook: resolve entity-registry ids for a document's chunks.
    entity_resolver: Callable[[SourceDocument], Sequence[str]] | None = None

    def __post_init__(self) -> None:
        # Dimension mismatch is a configuration error, not a per-chunk failure:
        # the column is vector(EMBEDDING_DIMENSIONS) and changing it needs a
        # migration, so fail once, loudly, before touching any row.
        if self.embedder.dimensions != EMBEDDING_DIMENSIONS:
            raise IndexingConfigError(
                f"embedder {self.embedder.model!r} returns {self.embedder.dimensions}-dimension "
                f"vectors but core.chunks.embedding is vector({EMBEDDING_DIMENSIONS}); "
                "reconfigure EMBEDDING_MODEL/EMBEDDING_DIMENSIONS and add a migration"
            )
        if self.batch_size is not None and self.batch_size < 1:
            raise IndexingConfigError("batch_size must be >= 1")

    @property
    def row_batch_size(self) -> int:
        requested = self.batch_size or self.embedder.max_batch_size
        return min(requested, self.embedder.max_batch_size)

    # -- public entry point ---------------------------------------------
    async def run(self, req: IndexRequest, *, trace_id: str | None = None) -> IndexResult:
        """Chunk, embed, and upsert the requested documents."""
        started = time.perf_counter()
        tracer = Tracer(self.event_sink or make_sink(self.session_factory), trace_id)
        document_ids = list(dict.fromkeys(req.document_ids))
        if not document_ids:
            return IndexResult(chunks_written=0, chunks_skipped=0, embed_tokens=0)

        documents = await self._load_documents(document_ids)

        superseded = 0
        active: list[Any] = []
        for document_id in document_ids:
            row = documents.get(document_id)
            if row is None:
                # Nothing to index and nothing to clean: no row, no chunks
                # (the FK would have cascaded). Caller asked for a stale id.
                logger.warning("indexer: document %s not found — skipped", document_id)
                continue
            if not row.is_current:
                superseded += await self._delete_document_chunks(document_id)
                continue
            active.append(row)
        indexed = len(active)

        pending: list[_Pending] = []
        skipped = 0
        for row in active:
            source_doc = source_document_from_row(row)
            content_tokens = estimate_tokens(source_doc.content)
            if content_tokens > OVERSIZED_DOCUMENT_TOKENS:
                logger.warning(
                    "indexer: document %s (%s) is oversized (~%d tokens); chunking anyway",
                    source_doc.source_id,
                    row.id,
                    content_tokens,
                )
            entity_ids = tuple(self.entity_resolver(source_doc)) if self.entity_resolver else ()
            chunks = self.chunker.chunk(source_doc, entity_ids=entity_ids)
            prior = await self._load_prior_chunks(row.id)
            doc_pending, doc_skipped = await self._plan_document(
                row.id, chunks, prior, force=req.force_reembed
            )
            pending.extend(doc_pending)
            skipped += doc_skipped
            # A document that shrank (or was emptied) must not leave chunks
            # whose text no longer exists in any source. Skipped when the
            # document did not get shorter, so a no-op run writes nothing.
            if prior and max(prior) >= len(chunks):
                await self._delete_trailing_chunks(row.id, keep=len(chunks))

        written, embed_tokens, failed_batches = await self._embed_and_write(pending)

        result = IndexResult(
            chunks_written=written,
            chunks_skipped=skipped,
            embed_tokens=embed_tokens,
        )
        await tracer.emit(
            AGENT,
            "indexer.completed",
            {
                "documents_requested": len(document_ids),
                "documents_indexed": indexed,
                "chunks_superseded": superseded,
                "chunks_written": written,
                "chunks_skipped": skipped,
                "failed_batches": failed_batches,
                "embed_tokens": embed_tokens,
                "embedding_model": self.embedder.model,
                "embedding_dimensions": self.embedder.dimensions,
                "batch_size": self.row_batch_size,
            },
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        logger.info(
            "indexer: %d docs, %d chunks written, %d skipped, %d superseded, %d tokens",
            indexed,
            written,
            skipped,
            superseded,
            embed_tokens,
        )
        return result

    # -- database reads --------------------------------------------------
    async def _load_documents(self, document_ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, Any]:
        async with self.session_factory() as session:
            rows = await session.scalars(
                select(DocumentRow).where(DocumentRow.id.in_(list(document_ids)))
            )
            return {row.id: row for row in rows}

    async def _load_prior_chunks(self, document_id: uuid.UUID) -> dict[int, _PriorChunk]:
        async with self.session_factory() as session:
            rows = await session.execute(
                select(
                    ChunkRow.seq,
                    ChunkRow.content_hash,
                    ChunkRow.embedding.is_not(None),
                    ChunkRow.meta,
                ).where(ChunkRow.document_id == document_id)
            )
            return {
                int(seq): _PriorChunk(
                    content_hash=str(content_hash), has_embedding=bool(has_embedding), meta=meta
                )
                for seq, content_hash, has_embedding, meta in rows
            }

    # -- planning --------------------------------------------------------
    async def _plan_document(
        self,
        document_id: uuid.UUID,
        chunks: list[Chunk],
        prior: dict[int, _PriorChunk],
        *,
        force: bool,
    ) -> tuple[list[_Pending], int]:
        """Decide per chunk: skip, refresh metadata only, or embed + write."""
        pending: list[_Pending] = []
        skipped = 0
        for chunk in chunks:
            existing = prior.get(chunk.seq)
            if (
                existing is None
                or force
                or existing.content_hash != chunk.content_hash
                or not existing.has_embedding
            ):
                pending.append(_Pending(document_id=document_id, chunk=chunk))
                continue
            if existing.meta != chunk.meta:
                # Text is identical but metadata moved (labels, heading path,
                # entity ids). Re-embedding would cost tokens for nothing, so
                # the row is refreshed in place. The chunk still counts as
                # skipped: ``chunks_skipped`` means "unchanged content_hash",
                # and ``embed_tokens`` is what reports embedding spend.
                await self._update_meta(document_id, chunk)
            skipped += 1
        return pending, skipped

    # -- embedding + writes ---------------------------------------------
    async def _embed_and_write(self, pending: list[_Pending]) -> tuple[int, int, int]:
        """Embed in batches; commit each batch; keep going after a failure."""
        if not pending:
            return 0, 0, 0

        # One embedding per distinct text: duplicates inside a run (and across
        # the documents in it) reuse the vector instead of paying twice.
        by_hash: dict[str, list[_Pending]] = {}
        for item in pending:
            by_hash.setdefault(item.chunk.content_hash, []).append(item)

        hashes = list(by_hash)
        written = 0
        tokens = 0
        failed_batches = 0
        for start in range(0, len(hashes), self.row_batch_size):
            batch_hashes = hashes[start : start + self.row_batch_size]
            texts = [by_hash[content_hash][0].chunk.text for content_hash in batch_hashes]
            try:
                result = await self.embedder.embed(texts)
            except EmbeddingError as exc:
                failed_batches += 1
                logger.error(
                    "indexer: embedding batch %d-%d failed (%s); %d chunks left unwritten",
                    start,
                    start + len(batch_hashes) - 1,
                    exc,
                    sum(len(by_hash[h]) for h in batch_hashes),
                )
                continue
            if len(result.vectors) != len(batch_hashes):  # defensive: protocol violation
                failed_batches += 1
                logger.error(
                    "indexer: embedder returned %d vectors for %d inputs — batch dropped",
                    len(result.vectors),
                    len(batch_hashes),
                )
                continue

            tokens += result.tokens
            vectors = dict(zip(batch_hashes, result.vectors, strict=True))
            rows = [
                _chunk_row(item, vectors[item.chunk.content_hash])
                for content_hash in batch_hashes
                for item in by_hash[content_hash]
            ]
            await self._upsert_chunks(rows)
            written += len(rows)
        return written, tokens, failed_batches

    async def _upsert_chunks(self, rows: list[dict[str, Any]]) -> None:
        """Write one batch in its own transaction (never half a chunk)."""
        if not rows:
            return
        statement = pg_insert(ChunkRow).values(rows)
        statement = statement.on_conflict_do_update(
            index_elements=["document_id", "seq"],
            set_={
                "text": statement.excluded.text,
                "embedding": statement.excluded.embedding,
                "meta": statement.excluded.meta,
                "content_hash": statement.excluded.content_hash,
                "embedded_at": statement.excluded.embedded_at,
            },
        )
        async with self.session_factory() as session, session.begin():
            await session.execute(statement)

    async def _update_meta(self, document_id: uuid.UUID, chunk: Chunk) -> None:
        async with self.session_factory() as session, session.begin():
            await session.execute(
                update(ChunkRow)
                .where(ChunkRow.document_id == document_id, ChunkRow.seq == chunk.seq)
                .values(meta=chunk.meta, content_hash=chunk.content_hash)
            )

    # -- cleanup ---------------------------------------------------------
    async def _delete_document_chunks(self, document_id: uuid.UUID) -> int:
        async with self.session_factory() as session, session.begin():
            result = cast(
                CursorResult[Any],
                await session.execute(delete(ChunkRow).where(ChunkRow.document_id == document_id)),
            )
        deleted = int(result.rowcount or 0)
        if deleted:
            logger.info(
                "indexer: superseded document %s — deleted %d stale chunks",
                document_id,
                deleted,
            )
        return deleted

    async def _delete_trailing_chunks(self, document_id: uuid.UUID, *, keep: int) -> None:
        async with self.session_factory() as session, session.begin():
            result = cast(
                CursorResult[Any],
                await session.execute(
                    delete(ChunkRow).where(
                        ChunkRow.document_id == document_id, ChunkRow.seq >= keep
                    )
                ),
            )
        if result.rowcount:
            logger.info(
                "indexer: document %s shrank — deleted %d trailing chunks",
                document_id,
                result.rowcount,
            )


# ---------------------------------------------------------------------------
# Row mapping
# ---------------------------------------------------------------------------


def source_document_from_row(row: Any) -> SourceDocument:
    """Project a ``core.documents`` row onto the chunker's input type."""
    acl = row.acl if isinstance(row.acl, dict) else {}
    raw_labels = acl.get("labels") or []
    labels = tuple(str(label) for label in raw_labels if isinstance(label, str))
    return SourceDocument(
        source=str(getattr(row.source, "value", row.source)),
        source_id=str(row.source_id),
        doc_type=str(getattr(row.doc_type, "value", row.doc_type)),
        title=str(row.title),
        content=str(row.content),
        document_id=str(row.id),
        parent_source_id=str(row.parent_source_id) if row.parent_source_id else None,
        author=str(row.author) if row.author else None,
        url=str(row.url) if row.url else None,
        labels=labels,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _chunk_row(item: _Pending, vector: list[float]) -> dict[str, Any]:
    chunk = item.chunk
    return {
        "id": chunk_id_for(item.document_id, chunk.seq),
        "document_id": item.document_id,
        "seq": chunk.seq,
        "text": chunk.text,
        "embedding": vector,
        "meta": chunk.meta,
        "content_hash": chunk.content_hash,
        "embedded_at": datetime.now(UTC),
    }


# ---------------------------------------------------------------------------
# Default wiring (settings + database)
# ---------------------------------------------------------------------------


_indexer: Indexer | None = None


def get_indexer(*, embedder: Embedder | None = None, kind: str = "auto") -> Indexer:
    """Return the process-wide indexer, building it on first use.

    ``kind`` selects the embedder (``auto`` / ``openai`` / ``fake``). Passing an
    explicit ``embedder`` sidesteps the cache — the caller owns its lifecycle.
    """
    if embedder is not None:
        factory = get_session_factory()
        if factory is None:
            raise IndexingConfigError("DATABASE_URL is not set: cannot index documents")
        return Indexer(session_factory=factory, embedder=embedder)

    global _indexer
    if _indexer is None:
        factory = get_session_factory()
        if factory is None:
            raise IndexingConfigError("DATABASE_URL is not set: cannot index documents")
        _indexer = Indexer(session_factory=factory, embedder=build_embedder(kind=kind))
    return _indexer


async def index(req: IndexRequest) -> IndexResult:
    """Public entry point: index the requested documents."""
    return await get_indexer().run(req)


async def aclose_indexer() -> None:
    """Release the cached embedder's transport (app shutdown / tests)."""
    global _indexer
    if _indexer is not None:
        await _indexer.embedder.aclose()
    _indexer = None
