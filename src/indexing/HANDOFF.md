# Indexer Agent — Hand-off (component 02)

Status: **implemented and verified** on this machine (2026-11-17), including the
local-first change (768-dim local embeddings via Ollama).

Read with: `docs/agents/02-indexer-agent.md`, `docs/05-data-model.md`,
`docs/04-orchestration.md`.

---

## 1. What is implemented

| File | Contents |
|---|---|
| `src/indexing/tokens.py` | `estimate_tokens` (deterministic, dependency-free), `sha256_text`. |
| `src/indexing/chunker.py` | Structure-aware chunker: `Chunker`, `SourceDocument`, `Chunk`, `chunk_document`, `strip_email_quotes`, `REQUIRED_META_KEYS`. |
| `src/indexing/embedder.py` | `Embedder` protocol, `EmbeddingResult`, `DeterministicFakeEmbedder`, `OpenAICompatibleEmbedder`, `build_embedder`. |
| `src/indexing/errors.py` | Typed embedding/indexing errors (retryable vs terminal split). |
| `src/indexing/indexer.py` | `IndexRequest`/`IndexResult`, `Indexer`, `async def index(req)`, `get_indexer`, `chunk_id_for`, `source_document_from_row`. |
| `scripts/embed_backfill.py` | Remote-executable backfill/sweep CLI. |
| `tools/smoke_embeddings.py` | Manual, opt-in live-provider check (not in pytest). |
| `src/evals/fixtures.py` | Replaced the M0 stub: real corpus loader (`load_fixture_corpus`), `FixtureDocument`/`FixtureCorpus`, deterministic `seed_fixture_corpus`. |
| `fixtures/corpus/*.json` | The synthetic corpus I own (44 documents) + `manifest.json` + `README.md`. |
| `tests/indexing/` | 4 test modules + scratch-DB conftest (141 tests). |

### Contract (unchanged, frozen)

```python
class IndexRequest(BaseModel):
    document_ids: list[UUID]
    force_reembed: bool = False

class IndexResult(BaseModel):
    chunks_written: int
    chunks_skipped: int
    embed_tokens: int

async def index(req: IndexRequest) -> IndexResult
```

`index()` builds the process-wide `Indexer` from settings. Tests and the
backfill script construct `Indexer(session_factory=…, embedder=…)` directly,
which is why nothing in the test suite needs an API key, a network, or a
credential file.

### Chunker (`src/indexing/chunker.py`)

Deterministic and pure: no clock, no randomness, no env, no DB. Same document →
same text, same `seq`, same `content_hash`, byte for byte.

- **Sizes**: `max_tokens=500` hard ceiling per chunk (breadcrumb included),
  `min_tokens=300` used to fold a tiny trailing fragment back in when it fits,
  `overlap_ratio=0.15` → 75 tokens of trailing paragraph/sentence carried into
  the next chunk. Overlap never duplicates the whole previous chunk.
- **Structure first**: section boundaries are never merged, even when a section
  is short. A 40-token Notion section stays its own chunk; merging would hide
  the heading that gives it meaning.
- **Jira issue**: the leading `Label: value` block (requires ≥2 fields and at
  least one recognised label from `JIRA_CARD_FIELDS`) becomes its own *field
  card* chunk with `meta["fields"]` and `meta["section"]="fields"`; the
  description becomes `section="description"` chunks. A normalizer that emits a
  different layout still works — it falls back to paragraph chunking.
- **Jira comment**: one document per comment (per the data model), chunked as
  `section="comment"`, breadcrumbed with the *issue key*
  (`PHX-150 · comment · <title>`) because the comment id is opaque.
- **Notion**: markdown `#`..`######` headings define sections;
  `meta["heading_path"]` is level-aware (`("Q4 objectives", "Ship Ledger v2…")`).
  Content before the first heading becomes a `preamble` section. Heading lines
  are removed from the chunk body (they are already in the breadcrumb and in
  metadata). Headings with no content produce no chunk.
- **Email**: `strip_email_quotes()` cuts quoted replies (`On … wrote:`),
  `-----Original Message-----`, `---------- Forwarded message ---------`,
  Gmail's `From:/Sent:` header block, `>` quoted lines, and signatures after a
  `-- ` marker or `Sent from my <device>`. Also public, because the PII and
  guardrail agents need the same "new text in this message" rule.
- **Breadcrumb**: every chunk starts with one context line
  (`PHX-150 — Cut over billing webhooks to Ledger v2`, or
  `Email · Re: …`, or `<title> · <heading> · <subheading>`). Without it a
  comment like "agreed, I'll pick it up" is unembeddable and unfindable by FTS.
  It is counted against the token budget and truncated past 25% of it.
- **`meta` jsonb** per chunk: `source`, `source_id`, `doc_type`, `title`,
  `heading_path`, `section`, `part`, `labels`, `token_count`, `content_hash`,
  `entity_ids` (empty hook for M4), plus `document_id`, `parent_source_id`,
  `author`, `url`, `created_at`, `updated_at`, and `fields` when the chunk is a
  Jira field card.
- **`content_hash`** is `sha256(chunk.text)` — content-addressed, so identical
  text anywhere maps to one embedding-cache key. It is also inside `meta` for
  retrieval/evidence convenience.
- **Empty document** → zero chunks (the indexer then deletes any leftovers).

### Embedding (`src/indexing/embedder.py`)

- `Embedder` protocol: `model`, `dimensions`, `max_batch_size`, `embed(texts)`,
  `aclose()`. `embed()` splits arbitrarily long input into requests of at most
  `max_batch_size` (default and hard cap 128).
- `DeterministicFakeEmbedder`: `sha256(text + counter)` bytes → floats in
  `[-1, 1]`, L2-normalized. Stable across processes/machines, configurable
  dimension (default 768 = schema), records `calls` for batching assertions.
- `OpenAICompatibleEmbedder`: async `httpx`, `POST {base_url}/embeddings`,
  connect/read timeouts, bounded exponential backoff with **full jitter**,
  `Retry-After` support, error classification (401/403 auth-terminal, 429 rate
  limit, 408/409/425/5xx + transport retryable, other 4xx terminal), response
  validation (count, shape, numeric, **dimension**), and token accounting from
  `usage.prompt_tokens` with an estimate fallback when a provider omits usage.
  `dimensions` is only sent when `send_dimensions=True` (gateways reject
  unknown fields); it defaults to off because local `nomic-embed-text` is
  768-dim natively.
- `build_embedder(settings, kind="auto"|"openai"|"fake")`: model and dimensions
  from `settings.embedding_model` / `settings.embedding_dimensions`, endpoint
  and key from `LLM_BASE_URL` / `LLM_API_KEY`. **A key is not required** — the
  local-first Ollama endpoint has none; a remote provider without a key fails
  with `EmbeddingAuthError` on the first request.

### Indexing (`src/indexing/indexer.py`)

- **Only writer of `core.chunks`** (per the brief).
- Reads `core.documents` by id; missing ids are logged and skipped.
- **Idempotent upsert** keyed on `(document_id, seq)` via
  `INSERT … ON CONFLICT DO UPDATE`, and chunk ids are deterministic
  (`uuid5` of document id + seq), so re-indexing keeps the same row ids.
- **Per-chunk decision**: no row / changed hash / no vector / `force_reembed` →
  embed and write; unchanged hash + vector + unchanged meta → *skipped* (no
  write, no token); unchanged hash + changed meta (labels, heading path,
  entity ids) → metadata-only `UPDATE`, no re-embedding.
- **Intra-run dedupe**: identical chunk text is embedded once and the vector
  reused across all its chunks.
- **Transaction per embedding batch**: each batch is embedded, then committed
  in its own transaction. A failing batch (after the client's retries) is
  logged, counted, and skipped; earlier and later batches still land. A chunk is
  never half-written.
- **Supersession**: if `is_current=false`, all chunks of that document are
  deleted (the schema has no `is_current` on chunks, so deletion is the only way
  stale text becomes unretrievable). Trailing chunks are also deleted when a
  document shrinks.
- **Dimension guard**: an embedder whose `dimensions` differs from
  `db.models.EMBEDDING_DIMENSIONS` (768) raises `IndexingConfigError` at
  `Indexer` construction, before any row is touched.
- **Oversized documents**: `>100k` estimated tokens is logged and chunked anyway.
- **Audit**: one `indexer.completed` event per run with counts, token total,
  failed batches, model, dimensions, batch size, latency. A `trace_id` is
  generated when the caller does not supply one (`Indexer.run(trace_id=…)`).
- **`IndexResult` semantics** (important for the Lead): `chunks_written` = rows
  inserted/updated with new content; `chunks_skipped` = chunks whose
  `content_hash` is unchanged (per the frozen comment), which includes
  metadata-only refreshes — those cost zero `embed_tokens`. Deleted chunks
  (supersession / shrink) are **not** represented in the contract and are
  logged instead; if the Lead wants them reported, `superseded: int = 0` is the
  natural addition.

### Backfill (`scripts/embed_backfill.py`)

Built for the shared compute server, not the laptop.

```bash
# on the server (DATABASE_URL in the environment, never on the command line)
cd ~/agentic_rag
.venv/bin/python scripts/embed_backfill.py --limit 5000 --page-size 100 \
    --batch-size 64 --report backfill.json

# initial backfill only (documents with no chunks)
.venv/bin/python scripts/embed_backfill.py --only-missing

# re-embed everything after a model change (needs a migration if dims change)
.venv/bin/python scripts/embed_backfill.py --force

# offline structural check, no provider calls, no writes
.venv/bin/python scripts/embed_backfill.py --embedder fake --dry-run
```

- Keyset pagination over current document ids, `--page-size` documents per
  `index()` call, resumable and idempotent (re-running pays only for changes).
- `--source`, `--document-id` (repeatable), `--only-missing`, `--force`,
  `--embedder fake|openai|auto`, `--sleep-between-pages`, `--dry-run`,
  `--report` (JSON totals).
- The database URL comes from the environment only; there is deliberately no
  `--database-url` flag so a password cannot end up in shell history or `ps`.

Copying the project: `rsync -av --exclude .venv --exclude .mypy_cache ./ shared-server:~/agentic_rag/`,
then run it there with the server-reachable `DATABASE_URL`. **No multi-thousand
chunk backfill was run on this laptop** — the only local runs were the 44-
document scratch corpus with the fake embedder plus a 2-document live check.

---

## 2. Fixture corpus (mine)

`fixtures/corpus/` — 44 documents, hand-written, deterministic, fictional:

| File | Contents |
|---|---|
| `jira_phoenix.json` | 12 PHX issues + 9 PHX comments |
| `jira_atlas.json` | 5 ATL issues + 1 ATL comment (unrelated project) |
| `notion_pages.json` | 7 pages (goals/OKR, weekly status, vendor notes, team directory, Alice Migration plan, onboarding runbook, Atlas charter) + 2 page sections (webhook replay, incident escalation) |
| `gmail_threads.json` | 8 emails in 3 threads (Phoenix cutover ×5 with quoted history, Nimbus vendor ×2, Atlas maintenance ×1) |
| `manifest.json` | machine-readable index: projects/aliases, clock (`today=2026-11-17`), PII planted-in list, injection list, contradictions, false-positive names, overdue + due-this-week tables |
| `README.md` | how to load/seed it |

Coverage the other agents asked for:

- **Two projects**: `Project Phoenix` (`PHX`, billing/Ledger v2) and
  `Project Atlas` (`ATL`, warehouse ETL — no Ledger v2 text anywhere).
- **Tasks with owners/statuses/due dates**: overdue vs 2026-11-17 (`PHX-142`
  due 11-10, `PHX-131` due 11-14, `ATL-88` due 11-12), due this week
  (`PHX-150`, `PHX-155` 11-20, `PHX-147` 11-23, `ATL-91` 11-19), later
  (`PHX-160` 12-04, `PHX-163` 12-18, `PHX-166` 12-11).
- **Goals/OKRs, status report, email thread**: the OKR page says GA 2026-12-15;
  the weekly status says GA 2027-01-15 and cutover 2026-12-04; the thread carries
  the 11-20 → 11-27 → 12-04 progression.
- **PII (deliberate, fictional)**: 8 person names, 9 addresses
  (`*.example`), 2 phone numbers in the 555 range, one home/office address, and
  an API-key-shaped token (`sk-phx-…`) planted twice (Jira comment `PHX-144#c10390`,
  Notion section `Runbook: webhook replay`).
- **Prompt injection**: a Notion page (`Vendor Integration Notes — Nimbus
  Systems`, section "Assistant instructions", including an HTML-comment variant)
  and a Jira comment (`PHX-131#c10452`, "Ignore all previous instructions…"),
  both naming the API key as the exfiltration target.
- **Person-like name, not a person**: `Alice Migration` (`PHX-131` epic + the
  cutover plan page both state it is a codename, "not named after a person").
- **Contradictions for the Evidence Agent**: PHX-150 cutover date 2026-11-20
  (Jira field) vs 2026-11-27 (comment) vs 2026-12-04 (status page + email);
  Phoenix GA 2026-12-15 (OKR) vs 2027-01-15 (status); PHX-147 owner
  Ravi (Jira field) vs Ines (status page).

`load_fixture_corpus()` validates schema **and** manifest integrity: duplicate
ids, wrong `doc_type` for a source, missing `parent_source_id`, naive
timestamps, `updated_at < created_at`, empty content, and manifest references to
non-existent documents all raise `FixtureCorpusError`. `content_hash` is
computed by the loader (never stored in the JSON). `seed_fixture_corpus()` uses
`uuid5(source:source_id)` ids, so seeding twice updates the same rows and keeps
chunk ids stable.

Corpus chunking: **94 chunks** for 44 documents (measured via
`scripts/embed_backfill.py --dry-run`).

---

## 3. How to run

```bash
cd /home/varun/projects/agentic_rag_over_live_data

# unit tests only (no DB, no network)
.venv/bin/python -m pytest tests/indexing -q

# with the scratch DB (indexer + seeding tests)
set -a; source ~/.config/agentic-rag/dev.env; set +a
.venv/bin/python -m pytest tests/indexing -q

# lint / types
.venv/bin/ruff check
.venv/bin/mypy src

# live provider check (manual; needs Ollama up)
.venv/bin/python tools/smoke_embeddings.py
```

### Scratch database (indexer tests)

Provisioned once, already done here:

```bash
sudo -u postgres createdb -O rag agentic_rag_idx_test
sudo -u postgres psql -d agentic_rag_idx_test \
  -c "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_trgm;"
```

`tests/indexing/conftest.py` derives the URL from `DATABASE_URL`
(`…/agentic_rag_idx_test`), or from `INDEX_TEST_DATABASE_URL`, **refuses** any
database name not ending in `_idx_test`, runs `alembic upgrade head` (follows
migration 0002 automatically), and truncates `core.documents` (cascades to
chunks) + `audit.events` before each test. The dev database is never written.
Without `DATABASE_URL` the DB tests skip with an explicit reason.

---

## 4. Local-first delta (2026-11-17, Lead sign-off)

1. **Dimensions 1536 → 768.** Nothing in my code hardcodes 1536 any more:
   - `DeterministicFakeEmbedder` default is 768 (matches
     `db.models.EMBEDDING_DIMENSIONS`; a test asserts the two stay equal).
   - `Indexer.__post_init__` compares against `EMBEDDING_DIMENSIONS` and raises
     `IndexingConfigError` naming `vector(768)`.
   - Model/dimension still come from `settings.embedding_model` /
     `settings.embedding_dimensions` — unchanged, as instructed.
   - Scratch DB was re-migrated to head (`0002_local_embedding`) and the column
     is `vector(768)` with the rebuilt HNSW index.
2. **Local provider exercised for real** (manual, opt-in — `pytest` never
   touches the network). `tools/smoke_embeddings.py` output:

   ```
   endpoint : http://localhost:11434/v1
   model    : nomic-embed-text
   expected : 768 dimensions
   latency  : 14954 ms for 3 inputs (1 request)   # cold start: model load
   tokens   : 51 (provider usage, else estimate)
   dims     : [768]
   first 5  : [-0.002672, 0.032464, -0.159653, -0.011499, 0.013296]
   l2 norms : [1.0, 1.0, 1.0]
   cosine(same text)      : 1.000000
   cosine(different text) : 0.572556
   OK: live embedding endpoint matches the configured model and schema dimension
   ```

   A second, warmer run indexed two real corpus documents through
   `Indexer` against Ollama into the scratch DB: 8 chunks written / 946 tokens,
   immediate re-run 8 skipped / 0 tokens, `fts @@ to_tsquery('cutover')` → 7
   rows, and `ORDER BY embedding <=> …` returned neighbours at cosine
   1.0000 / 0.8540 / 0.7261. So the full path (chunk → HTTP embed → vector(768)
   write → FTS + ANN read) is verified against the real endpoint.
3. **`build_embedder(kind="auto")` no longer requires `LLM_API_KEY`** (Ollama
   has none). It only requires a non-empty `LLM_BASE_URL`. A remote provider
   without a key now fails with `EmbeddingAuthError` on first use instead of a
   config error at construction. This is a deliberate behaviour change for
   local-first.

---

## 5. Deviations from the brief (for Lead review)

1. **Token counting is an estimate, not a tokenizer.** `estimate_tokens` counts
   words + punctuation marks. No `tiktoken` dependency (it would pin the chunker
   to one provider and is a large install). Deterministic and stable, which is
   what golden chunk tests need; absolute numbers are ±20% of the provider's
   count. Documented in `tokens.py`.
2. **Overlap is applied only when a structural unit overflows.**
   `docs/agents/02-indexer-agent.md` says 300–500 tokens with ~15% overlap;
   applying a floor of 300 to every section would merge across headings, which
   the same brief forbids ("respect structure first"). So short sections stay
   short and overlap only appears where a long section is split.
3. **Chunk breadcrumbs are part of the chunk text** (and therefore of the
   `content_hash` and the token count). Trade-off: cross-document dedupe of
   byte-identical bodies no longer happens, but embedding quality and FTS
   findability for short comments/sections improve substantially.
4. **Per-run embedding dedupe as well as the cross-run cache.** The brief asks
   for "cached by content hash so unchanged chunks never re-embed"; I also
   dedupe identical text *within* one run (same vector reused). Cheap, and it
   makes duplicated Notion sections cost one embedding.
5. **Embeddings reuse `LLM_BASE_URL` / `LLM_API_KEY`.** There is no
   `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` in `config.py` and I did not add
   them (config.py is shared with other agents mid-flight). If embeddings ever
   come from a different vendor than completions, the Lead should add those two
   fields; `build_embedder` is the single place that would change.
6. **`chunks_skipped` includes metadata-only refreshes** (unchanged
   `content_hash`, changed `meta`). Follows the frozen comment literally; the
   row is still updated in place. `embed_tokens` is the signal for spend.
   Deleted chunks (supersession/shrink) are logged, not returned — the frozen
   `IndexResult` has no field for them.
7. **No `dedupe` of chunks across documents** (same text in two documents is
   stored twice, with two ids). Retrieval-side dedupe is agent 03's call; the
   schema keys chunks per document.
8. **Retry/backoff logic is duplicated** between `llm/openai_compatible.py` and
   `indexing/embedder.py` (~40 lines). Deliberate: the two have different error
   taxonomies and I did not want to refactor the Lead's M0 client from under
   other agents. If the Lead wants one implementation, a shared
   `http_retry.py` is a clean follow-up.
9. **No schema change was made.** All writes fit migration 0002 as-is.
10. **No new dependencies.** `httpx`, `pydantic`, `sqlalchemy`, `pgvector` were
    already in `pyproject.toml`; no additions were needed.

---

## 6. Uncertainty / risks a reviewer should know

- **`core.chunks.fts` is generated from `text` only.** Because the breadcrumb is
  inside `text`, issue keys are searchable; but FTS ranking sees the breadcrumb
  words too, so a query for "cutover" matches every chunk of the cutover issue.
  Acceptable for hybrid retrieval (agent 03 fuses with vectors), worth knowing.
- **Overlap duplicates text across chunks**, so naive "count how often a fact
  appears" logic in an eval will double-count boundary facts. Dedupe by
  `content_hash` won't catch it (different text); the eval harness should count
  claims, not chunks.
- **`estimate_tokens` vs the real model.** `nomic-embed-text` truncates at 2048
  tokens (a chunk of ~500 estimated tokens is ~650 real ones, comfortably
  inside). Chunks are not silently truncated.
- **Metadata-only refresh does a write per chunk** (one round trip each, no
  batching). For a 5k-chunk document with relabelled ACLs that is 5k small
  transactions. Rare, but if it becomes common, batch the `UPDATE`s.
- **Supersession deletes chunks.** There is no historical chunk trail after a
  document is superseded; the audit value lives in `core.documents`
  (`is_current=false` rows are kept) and in `audit.events`, not in chunks. That
  matches the brief ("delete-or-supersede"), but it is irreversible for chunks.
- **Concurrency**: two indexers running on the same document can race on the
  trailing-chunk `DELETE` (both compute `keep` independently). The upsert itself
  is safe. Run one backfill at a time.
- **Live endpoint latency**: the first embedding request after idle took ~15 s
  (Ollama loading the model). The client's default `LLM_TIMEOUT_S=120` covers
  it; a cold `--only-missing` backfill on the server will be slow for the first
  batch only.
- **`tools/smoke_embeddings.py` depends on `nomic-embed-text` being pulled and
  Ollama running.** It is manual by design; nothing in CI checks the real
  provider.
- **Unverified**: behaviour under a hosted embedding provider (no API key here),
  and any provider that returns fewer/more vectors than inputs (handled with a
  typed error, only exercised against `httpx.MockTransport`).

---

## 7. Verified gates (real output, 2026-11-17)

```
$ .venv/bin/ruff check
All checks passed!

$ .venv/bin/mypy src
Success: no issues found in 47 source files

$ .venv/bin/mypy src/indexing src/evals/fixtures.py
Success: no issues found in 7 source files

$ env -u DATABASE_URL .venv/bin/python -m pytest tests/indexing -q
119 passed, 22 skipped, 1 warning in 0.16s
    # the 22 skips are the scratch-DB indexer tests, skipped with an explicit
    # "DATABASE_URL is not set" reason.

$ env -u DATABASE_URL .venv/bin/python -m pytest -q          # no DB configured
281 passed, 39 skipped, 1 warning in 0.68s

$ set -a; source ~/.config/agentic-rag/dev.env; set +a
$ .venv/bin/python -m pytest tests/indexing -q
141 passed, 1 warning in 4.19s

$ .venv/bin/python -m pytest -q                              # full suite + DB
315 passed, 5 skipped, 1 warning in 5.23s

$ .venv/bin/python tools/smoke_embeddings.py                   # live Ollama
OK: live embedding endpoint matches the configured model and schema dimension
```

The 5 skips in the full run are all in `tests/connectors/test_db_store.py`
(agent 01's suite; it needs `CONNECTOR_TEST_DATABASE_URL`). Nothing in
`tests/indexing` is skipped once `DATABASE_URL` is set.

---

## 8. What I did **not** do (scope)

- No retrieval, no PII scrubbing, no evidence/synthesis, no orchestration.
- No connector work; `fixtures/jira/`, `fixtures/golden.json`, and any metric
  definitions in `fixtures/` belong to agents 01/09 and were not touched.
- No schema migration, no new dependency, no change to `src/orchestrator/`,
  `src/api/`, or `src/connectors/`.
- No git operations (git is not installed on this machine).
