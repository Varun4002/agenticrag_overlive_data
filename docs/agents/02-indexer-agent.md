# Agent 02 — Indexer Agent

> Read with: `docs/02-architecture.md`, `docs/05-data-model.md`.

## Mission

Turn canonical documents into searchable chunks: chunk → embed → upsert into
`core.chunks` (vector + FTS). You are the only writer of `core.chunks`.

## Contract

```python
class IndexRequest(BaseModel):
    document_ids: list[UUID]      # explicit; the caller decides what to index
    force_reembed: bool = False

class IndexResult(BaseModel):
    chunks_written: int
    chunks_skipped: int           # unchanged content_hash
    embed_tokens: int
```

Public API: `async def index(req: IndexRequest) -> IndexResult`.
Triggered by Connector Agent after upserts, and by a re-embed sweep job.

## Chunking rules (deterministic, unit-tested)

- Target 300–500 tokens, overlap ~15%.
- Respect structure first: Jira issue fields (summary, description, comments
  chunk separately, comment = its own document anyway); Notion heading
  boundaries; email body vs quoted reply (strip forwarded chains — they
  duplicate other chunks).
- Chunk `meta` jsonb records: heading path, doc_type, source, labels, and any
  entity ids from `mem.entity_registry` matches.
- `content_hash` per chunk — dedupe and staleness detection.

## Embedding

- Single embedder interface; config decides model (MVP: one API model,
  1536-dim unless configured otherwise — schema uses `vector(1536)`).
- Batched requests (≤ 128 inputs/batch), async, cached by content hash so
  unchanged chunks never re-embed.
- Bulk embedding runs on the shared server (`ssh shared-server`) — provide a
  `scripts/embed_backfill.py` entrypoint suitable for remote execution; never
  run multi-thousand-chunk backfills on the laptop.

## Indexing

- Upsert into `core.chunks` keyed by `(document_id, seq)`.
- HNSW index on `embedding` (cosine), GIN on `fts` — per `05-data-model.md`.
- On document supersession (`is_current=false`), delete-or-supersede its
  chunks so stale content can't be retrieved.

## Failure modes

| Failure | Handling |
|---|---|
| Embedding API failure mid-batch | retry with backoff; write completed batches, report failed ones; never half-write a chunk (transaction per batch) |
| Dimension mismatch vs configured model | hard config error at startup, not per-chunk |
| Oversized document | chunk it; log if > 100k tokens |

## Tests

- Chunker: golden tests on fixture docs (Jira issue, Notion page with
  headings, email with quoted reply) — assert boundaries, overlap, meta.
- Embedding: fake transport; assert batching, cache hits, token accounting.
- Upsert idempotency: index twice → same chunk ids, second pass all skipped.
