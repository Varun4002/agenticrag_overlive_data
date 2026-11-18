"""Chunking, embedding, and hybrid-index maintenance (M1).

Public surface:

- :func:`~indexing.chunker.chunk_document` / :class:`~indexing.chunker.Chunker`
  — deterministic structure-aware chunking.
- :class:`~indexing.embedder.Embedder` (protocol) and its two implementations,
  ``DeterministicFakeEmbedder`` (offline/tests) and
  ``OpenAICompatibleEmbedder`` (production path).
- :class:`~indexing.indexer.IndexRequest` / ``IndexResult`` and
  ``async def index(req)`` — the frozen contract from
  ``docs/agents/02-indexer-agent.md``.

This package is the only writer of ``core.chunks``.
"""

from __future__ import annotations

from indexing.chunker import (
    REQUIRED_META_KEYS,
    Chunk,
    Chunker,
    SourceDocument,
    chunk_document,
    strip_email_quotes,
)
from indexing.embedder import (
    DeterministicFakeEmbedder,
    Embedder,
    EmbeddingResult,
    OpenAICompatibleEmbedder,
    build_embedder,
)
from indexing.errors import (
    EmbeddingAuthError,
    EmbeddingBadRequestError,
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingRateLimitError,
    EmbeddingResponseError,
    EmbeddingServerError,
    EmbeddingTimeoutError,
    IndexingConfigError,
    IndexingError,
)
from indexing.indexer import (
    Indexer,
    IndexRequest,
    IndexResult,
    aclose_indexer,
    chunk_id_for,
    get_indexer,
    index,
    source_document_from_row,
)
from indexing.tokens import estimate_tokens, sha256_text

__all__ = [
    "REQUIRED_META_KEYS",
    "Chunk",
    "Chunker",
    "DeterministicFakeEmbedder",
    "Embedder",
    "EmbeddingAuthError",
    "EmbeddingBadRequestError",
    "EmbeddingDimensionError",
    "EmbeddingError",
    "EmbeddingRateLimitError",
    "EmbeddingResponseError",
    "EmbeddingResult",
    "EmbeddingServerError",
    "EmbeddingTimeoutError",
    "IndexRequest",
    "IndexResult",
    "Indexer",
    "IndexingConfigError",
    "IndexingError",
    "OpenAICompatibleEmbedder",
    "SourceDocument",
    "aclose_indexer",
    "build_embedder",
    "chunk_document",
    "chunk_id_for",
    "estimate_tokens",
    "get_indexer",
    "index",
    "sha256_text",
    "source_document_from_row",
    "strip_email_quotes",
]
