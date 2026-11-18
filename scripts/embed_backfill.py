#!/usr/bin/env python
"""Backfill/sweep entrypoint: chunk + embed every current document.

Designed to be run **on the shared compute server**, not on a laptop::

    # from the project root on the server
    export DATABASE_URL=...        # never pass credentials on the command line
    .venv/bin/python scripts/embed_backfill.py --limit 5000 --page-size 100

Typical flow:

    rsync -av --exclude .venv --exclude .mypy_cache ./ shared-server:~/agentic_rag/
    ssh shared-server 'cd agentic_rag && .venv/bin/python scripts/embed_backfill.py'

Properties:

- **Resumable and idempotent.** The indexer skips chunks whose content hash and
  metadata are unchanged, so re-running only pays for what changed.
- **Bounded memory.** Documents are processed in pages (``--page-size``), and
  each embedding batch is committed on its own.
- **Observable.** One ``indexer.completed`` audit event per page, plus a JSON
  report with per-page totals if ``--report`` is given.
- **No secrets on the command line.** The database URL and provider key come
  from the environment only.

``--embedder fake`` writes deterministic hash vectors (no provider calls). Use it
to exercise the whole pipeline offline, or in a test database.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Make ``python scripts/embed_backfill.py`` work without installing the package:
# the project uses a src/ layout with an editable install, but a fresh clone on
# the compute server may not have one.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sqlalchemy import select, text  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker  # noqa: E402

from config import Settings, get_settings  # noqa: E402
from db.models import Document as DocumentRow  # noqa: E402
from db.session import dispose_engine, get_session_factory  # noqa: E402
from indexing.chunker import Chunker, SourceDocument, chunk_document  # noqa: E402
from indexing.embedder import build_embedder  # noqa: E402
from indexing.errors import EmbeddingError, IndexingConfigError  # noqa: E402
from indexing.indexer import Indexer, IndexRequest, source_document_from_row  # noqa: E402
from indexing.tokens import estimate_tokens  # noqa: E402

logger = logging.getLogger("embed_backfill")


@dataclass
class Totals:
    """Accumulated counters across all pages."""

    documents: int = 0
    chunks_written: int = 0
    chunks_skipped: int = 0
    embed_tokens: int = 0
    pages: int = 0
    started: float = field(default_factory=time.perf_counter)

    def add(self, written: int, skipped: int, tokens: int) -> None:
        self.chunks_written += written
        self.chunks_skipped += skipped
        self.embed_tokens += tokens
        self.pages += 1

    def as_payload(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "chunks_written": self.chunks_written,
            "chunks_skipped": self.chunks_skipped,
            "embed_tokens": self.embed_tokens,
            "pages": self.pages,
            "elapsed_s": round(time.perf_counter() - self.started, 2),
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk and embed current documents (run on the shared server).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=None, help="max documents to process")
    parser.add_argument(
        "--page-size", type=int, default=50, help="documents per index() call"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="chunks per embedding request (capped at the embedder's max)",
    )
    parser.add_argument(
        "--source", choices=("jira", "notion", "gmail"), help="restrict to one source"
    )
    parser.add_argument(
        "--document-id",
        action="append",
        default=None,
        metavar="UUID",
        help="index specific documents instead of sweeping (repeatable)",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="only documents that have no chunks yet (initial backfill)",
    )
    parser.add_argument("--force", action="store_true", help="re-embed unchanged chunks")
    parser.add_argument(
        "--embedder",
        choices=("auto", "openai", "fake"),
        default="auto",
        help="'fake' writes deterministic vectors without calling a provider",
    )
    parser.add_argument(
        "--sleep-between-pages",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="pause between pages to be gentle with the provider",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="count documents/chunks that would be processed, write nothing",
    )
    parser.add_argument("--report", type=Path, help="write a JSON report to this path")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


async def select_documents(
    factory: async_sessionmaker[AsyncSession],
    *,
    source: str | None,
    only_missing: bool,
    after: uuid.UUID | None,
    page_size: int = 500,
) -> list[uuid.UUID]:
    """One page of current document ids, ordered by id (keyset pagination)."""
    query = select(DocumentRow.id).where(DocumentRow.is_current.is_(True))
    if source:
        query = query.where(DocumentRow.source == source)
    if only_missing:
        query = query.where(
            text(
                "NOT EXISTS (SELECT 1 FROM core.chunks c WHERE c.document_id = core.documents.id)"
            )
        )
    if after is not None:
        query = query.where(DocumentRow.id > after)
    query = query.order_by(DocumentRow.id).limit(page_size)
    async with factory() as session:
        return list((await session.scalars(query)).all())


async def explicit_documents(
    factory: async_sessionmaker[AsyncSession], ids: list[str]
) -> list[uuid.UUID]:
    parsed: list[uuid.UUID] = []
    for raw in ids:
        try:
            parsed.append(uuid.UUID(raw))
        except ValueError as exc:
            raise SystemExit(f"invalid document id {raw!r}: {exc}") from exc
    async with factory() as session:
        rows = await session.scalars(
            select(DocumentRow.id).where(DocumentRow.id.in_(parsed)).order_by(DocumentRow.id)
        )
        found = list(rows)
    missing = set(parsed) - set(found)
    if missing:
        logger.warning("documents not found, skipping: %s", sorted(str(item) for item in missing))
    return found


async def dry_run(
    factory: async_sessionmaker[AsyncSession],
    document_ids: list[uuid.UUID],
    *,
    chunker: Chunker,
) -> dict[str, Any]:
    """Count what would be written, without writing or embedding anything."""
    documents = 0
    chunks = 0
    oversized: list[str] = []
    async with factory() as session:
        for document_id in document_ids:
            row = await session.get(DocumentRow, document_id)
            if row is None:
                continue
            documents += 1
            source_doc: SourceDocument = source_document_from_row(row)
            if estimate_tokens(source_doc.content) > 100_000:
                oversized.append(source_doc.source_id)
            chunks += len(chunk_document(source_doc, chunker=chunker))
    return {"documents": documents, "chunks": chunks, "oversized": oversized}


def build_indexer(
    factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    *,
    embedder_kind: str,
    batch_size: int | None,
) -> Indexer:
    embedder = build_embedder(settings, kind=embedder_kind)
    logger.info(
        "embedder: %s (model=%s, dimensions=%d)",
        embedder_kind,
        embedder.model,
        embedder.dimensions,
    )
    return Indexer(session_factory=factory, embedder=embedder, batch_size=batch_size)


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    factory = get_session_factory()
    if factory is None:
        raise IndexingConfigError(
            "DATABASE_URL is not set — export it in the environment (never on the command line)"
        )

    chunker = Chunker()
    totals = Totals()
    try:
        if args.document_id:
            pending: list[uuid.UUID] = await explicit_documents(factory, args.document_id)
            if args.limit:
                pending = pending[: args.limit]
        else:
            pending = await select_documents(
                factory,
                source=args.source,
                only_missing=args.only_missing,
                after=None,
            )

        if args.dry_run:
            summary = await dry_run(factory, pending, chunker=chunker)
            print(
                f"dry run: {summary['documents']} documents, {summary['chunks']} chunks, "
                f"no writes"
            )
            if summary["oversized"]:
                print(f"oversized documents (>100k tokens): {summary['oversized']}")
            return 0

        indexer = build_indexer(
            factory, settings, embedder_kind=args.embedder, batch_size=args.batch_size
        )
        logger.info(
            "backfill: embedder=%s batch=%d page_size=%d force=%s",
            args.embedder,
            indexer.row_batch_size,
            args.page_size,
            args.force,
        )

        processed = 0
        remaining = args.limit
        while pending:
            page = pending[: args.page_size]
            if remaining is not None:
                page = page[:remaining]
            if not page:
                break
            pending = pending[len(page) :]

            result = await indexer.run(IndexRequest(document_ids=page, force_reembed=args.force))
            totals.documents += len(page)
            totals.add(result.chunks_written, result.chunks_skipped, result.embed_tokens)
            processed += len(page)
            print(
                f"[{processed}] docs={len(page)} written={result.chunks_written} "
                f"skipped={result.chunks_skipped} tokens={result.embed_tokens}",
                flush=True,
            )

            if remaining is not None:
                remaining -= len(page)
                if remaining <= 0:
                    break
            if not pending and not args.document_id:
                # Keyset pagination: continue past the last id we processed.
                pending = await select_documents(
                    factory,
                    source=args.source,
                    only_missing=args.only_missing,
                    after=page[-1],
                )
            if args.sleep_between_pages:
                await asyncio.sleep(args.sleep_between_pages)

        await indexer.embedder.aclose()
        payload = totals.as_payload()
        print(json.dumps({"total": payload}, indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"report written to {args.report}")
        return 0
    except EmbeddingError as exc:
        # A provider outage is an operator problem, not a crash: report clearly.
        print(f"embedding failed: {exc}", file=sys.stderr)
        return 1
    finally:
        await dispose_engine()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
