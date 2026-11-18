# 05 — Data Model

PostgreSQL 16, `pgvector` + `pg_trgm`. All tables live in one database.
Schema names: `core` (docs/chunks), `mem` (memory), `audit`.

## core.documents — canonical source items

One row per ingested item (Jira issue, issue comment, Notion page section,
email).

| Column | Type | Notes |
|---|---|---|
| `id` | uuid PK | |
| `source` | enum | `jira` / `notion` / `gmail` |
| `source_id` | text | native ID (issue key, page id, message id) |
| `parent_source_id` | text nullable | issue key for a comment; thread id for an email |
| `doc_type` | enum | `issue` / `issue_comment` / `page` / `page_section` / `email` |
| `title` | text | |
| `content` | text | raw normalized text (pre-PII-scrub; never leaves the DB unredacted) |
| `url` | text | deep link back to the source |
| `author` | text nullable | source-visible identity (for ACL/registry only) |
| `created_at` / `updated_at` | timestamptz | **source** timestamps |
| `fetched_at` | timestamptz | our sync time |
| `acl` | jsonb | mirrored permissions: `{users:[], groups:[], labels:[]}` |
| `content_hash` | text | sha256 of content — change detection |
| `is_current` | bool | superseded rows kept for audit |

Unique: `(source, source_id, is_current)`.

## core.chunks

| Column | Type | Notes |
|---|---|---|
| `id` | uuid PK | |
| `document_id` | uuid FK → documents |
| `seq` | int | position within the document |
| `text` | text | chunk text (pre-redaction) |
| `embedding` | vector(768) | dims match `nomic-embed-text` (local Ollama, migration 0002) |
| `fts` | tsvector | generated — for `pg_trgm`/FTS hybrid |
| `meta` | jsonb | heading path, doc_type, labels, entity ids |
| `embedded_at` | timestamptz | enables re-embed sweeps |
| `content_hash` | text | dedupe / staleness |

## mem.answer_cache

| Column | Type | Notes |
|---|---|---|
| `id` | uuid PK | |
| `question_normalized` | text | normalized query text |
| `answer` | text | final synthesized answer |
| `citations` | jsonb | citation list as delivered |
| `evidence_hashes` | text[] | content_hash of every chunk used |
| `source_versions` | jsonb | per-source last-updated watermark at answer time |
| `created_at` | timestamptz | |
| `expires_at` | timestamptz | TTL or next sync watermark |

Invalidation rule: cache row is *stale* if any source watermark advanced
beyond `source_versions` or any `evidence_hashes` no longer matches current
chunk hashes.

## mem.entity_registry

| Column | Type | Notes |
|---|---|---|
| `id` | uuid PK | |
| `canonical_name` | text | e.g. `Project Phoenix` |
| `entity_type` | enum | `project` / `task` / `person` / `team` / `page` |
| `aliases` | text[] | `PHX`, `Phoenix`, page titles… |
| `source_refs` | jsonb | `{"jira":"PHX","notion":"page:abc","gmail":"label:phoenix"}` |
| `confidence` | real | registry entries below 0.5 don't auto-resolve |
| `updated_at` | timestamptz | |

Built by matching names/keys across ingested docs; corrections are stored as
rows with confidence boost.

## mem.session_turns

| Column | Type | Notes |
|---|---|---|
| `id`, `session_id`, `role`, `content`, `created_at` | | last-N-turns working memory |
| `entity_hits` | jsonb | entities resolved in this turn (planner priming) |

## audit.redactions

| Column | Type | Notes |
|---|---|---|
| `id`, `trace_id` | | which request caused it |
| `document_id`, `chunk_id` | | where it happened |
| `pii_type` | enum | `person` / `email` / `phone` / `address` / `api_key` / `other` |
| `placeholder` | text | e.g. `[PERSON_1]` |
| `original_hash` | text | salted hash of the original value — debugging join, NOT reversible to the value |
| `created_at` | | |

## audit.events

| Column | Type | Notes |
|---|---|---|
| `trace_id`, `agent`, `event` | | e.g. `retrieval.ran`, `guardrail.blocked` |
| `payload` | jsonb | decision details, scores, token counts |
| `latency_ms` | int | |
| `created_at` | | |

Every agent writes here. This is the compliance and debugging backbone —
records *what data was used to answer whom*, which the guardrail layer requires.

## core.sync_state

| Column | Type | Notes |
|---|---|---|
| `source` | text PK | |
| `watermark` | timestamptz | last sync cursor (e.g. Jira `updated`) |
| `last_run_at`, `last_status`, `error` | | |

## Indexes (required)

- `documents (source, updated_at desc)` — incremental sync
- `chunks USING hnsw (embedding vector_cosine_ops)` — ANN search
- `chunks USING gin (fts)` — FTS
- `answer_cache (question_normalized)`; entity_registry GIN on `aliases`
- `audit.events (trace_id)`, `(created_at desc)`

## Migration & seed strategy

Alembic migrations from M0. A `fixtures/` corpus (50–100 synthetic
documents resembling Jira/Notion/Gmail output, PII included on purpose) seeds
all dev/test environments — agents never depend on live credentials to build
or test.
