# HANDOFF — M0 Skeleton

> **Scope note (2026-11-17):** this document describes **M0 only** and is kept
> for its reasoning about the skeleton, schema, tracing and LLM client. It is
> no longer a status report: connectors, indexing, retrieval, PII, evidence,
> synthesis and evals have since landed. Current status lives in `README.md`
> ("Status"), each component's `src/*/HANDOFF.md`, and `docs/09-operator-dashboard.md`.


Status: **complete and verified** on this machine (2026-11-17). All three gates
pass: `pytest`, `ruff check`, `mypy src`.

Read with: `docs/02-architecture.md`, `docs/03-mvp-plan.md` (M0),
`docs/05-data-model.md`.

---

## 1. What is implemented

### Project layout (`src/`, per `02-architecture.md`)

All component packages exist with docstrings describing their milestone:

```
api/ orchestrator/ connectors/ indexing/ retrieval/ pii/ evidence/
synthesis/ guardrails/ memory/ llm/ db/ evals/    + config.py
```

Only `api/`, `orchestrator/contracts.py`, `db/`, `llm/`, and `evals/fixtures.py`
have code in M0. The rest are importable placeholders with no behavior — nothing
there pretends to work.

### HTTP API (`src/api/`)

| Route | Behavior |
|---|---|
| `POST /ask` | Validates `AskRequest`, generates a `trace_id`, writes three `audit.events` rows (`api.request.received`, `orchestrator.placeholder`, `api.request.completed` with latency), returns the frozen `AskResponse` with a placeholder answer, empty citations, `refusal=null`. |
| `GET /health` | Liveness plus `database.configured` / `database.reachable` (live `SELECT 1` probe). |

- `src/api/main.py` — `create_app()` + module-level `app` for uvicorn; lifespan
  logs config and disposes the engine on shutdown.
- `src/api/dependencies.py` — `get_event_sink` / `get_tracer`. Tests override
  `get_event_sink` (see `tests/conftest.py`), so endpoint tests need no database.
- `src/api/routes.py` — the two routes.
- `src/orchestrator/contracts.py` — frozen `AskRequest`, `AskResponse`,
  `Citation`, `Refusal` exactly as specified in `docs/agents/00-orchestrator.md`.
  No extra fields were added (the freeze rule in `04-orchestration.md`).

**Degradation behavior (verified, not assumed):** with `DATABASE_URL` pointing
at a dead port, `/ask` still returns 200 (trace writes are logged as warnings,
never raised) and `/health` reports `database.reachable=false`.

### Config (`src/config.py`)

`pydantic-settings` `Settings`, no env prefix (so field `llm_model` reads
`LLM_MODEL`), cached via `get_settings()`. Fields:

- app: `APP_ENV`, `APP_VERSION`, `LOG_LEVEL`, `API_HOST`, `API_PORT`
- db: `DATABASE_URL` (optional — unset means DB features are off), `DB_ECHO`,
  `DB_CONNECT_TIMEOUT_S`
- llm: `LLM_PROVIDER` (`openai_compatible`), `LLM_BASE_URL`, `LLM_API_KEY`
  (`SecretStr`, never in `repr`), `LLM_MODEL`, `LLM_TIMEOUT_S`,
  `LLM_MAX_RETRIES`, `LLM_BACKOFF_BASE_S`
- embeddings: `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`
- sync: `JIRA_SYNC_INTERVAL_S`, `NOTION_SYNC_INTERVAL_S`, `GMAIL_SYNC_INTERVAL_S`,
  `SYNC_STALE_AFTER_S`

`.env.example` documents every variable. `.env` is git-ignored and no `.env`
was created.

### Database schema + migrations (`src/db/`)

`src/db/migrations/versions/0001_initial_schema.py` creates, per
`05-data-model.md`:

- schemas `core`, `mem`, `audit`; extensions `vector`, `pg_trgm`
- enum types `core.source_type`, `core.doc_type`, `mem.entity_type`,
  `audit.pii_type`
- `core.documents`, `core.chunks` (vector(1536) + generated stored `tsvector`),
  `core.sync_state`
- `mem.answer_cache`, `mem.entity_registry`, `mem.session_turns`
- `audit.redactions`, `audit.events`
- required indexes: `(source, updated_at DESC)`, HNSW `vector_cosine_ops` on
  embeddings, GIN on `fts`, `answer_cache(question_normalized)`, GIN on
  `entity_registry.aliases`, `audit.events(trace_id)`, `audit.events(created_at DESC)`

ORM models in `src/db/models.py` mirror the migration (for M1 queries and
autogenerate). `src/db/base.py` sets deterministic constraint naming.

### Tracing (`src/db/tracing.py`)

`Tracer` (trace id + `emit`/`span`) over an `EventSink` protocol:
`DatabaseEventSink` (writes `audit.events`, swallows failures),
`NullEventSink` (no DB), `InMemoryEventSink` (tests). `read_trace()` reloads a
trace for debugging/eval. Payloads are `jsonb`; `latency_ms` is milliseconds.

### LLM client (`src/llm/`)

- `base.py` — `LLMClient` protocol, `Message`/`CompletionRequest`/`Usage`/
  `CompletionResponse` models, `TokenLedger` (per-request token accounting,
  `as_payload()` for trace payloads).
- `errors.py` — typed errors with an explicit retryable/terminal split.
- `openai_compatible.py` — async `httpx` client for
  `POST {base_url}/chat/completions`: connect+read timeouts, bounded exponential
  backoff with full jitter, `Retry-After` support, error classification
  (401/403 → auth, 429 → rate limit, 5xx/408/409/425 + transport failures →
  retryable, other 4xx → bad request), response validation, usage accounting
  into the ledger. Provider is injected via `httpx.AsyncBaseTransport` for tests.

**The client is not called by any endpoint** (per brief). It is exercised only
by unit tests with `httpx.MockTransport`.

### Fixtures plan (not fixtures)

`src/evals/fixtures.py` documents the planned synthetic corpus layout,
document shape, PII/injection content plan, and determinism rules.
`load_fixture_corpus()` raises `FixtureCorpusNotBuiltError` (a
`NotImplementedError`) — no fixture data exists.

---

## 2. What is stubbed / missing

| Item | State |
|---|---|
| Real retrieval, evidence, synthesis, guardrails, memory, connectors | Not started (M1–M5). Packages are empty placeholders. |
| Orchestrator state machine | Not started; only the frozen contracts exist. `/ask` does not route through an orchestrator yet. |
| Synthetic fixture corpus | Plan only (`src/evals/fixtures.py`). |
| Presidio / PII pipeline | Not started (M3). Not a M0 dependency, not installed. |
| Worker entrypoint (`worker.py`), sync scheduler | Not started (M1). |
| Auth on `/ask` | Not implemented (single-user local dev, as scoped). |
| Embeddings | No embedder yet; `core.chunks.embedding` exists but is never written. Dimension is pinned at 1536. |

---

## 3. How to run

```bash
cd /home/varun/projects/agentic_rag_over_live_data

# 1. venv + install (editable, so `uvicorn api.main:app` and `alembic` work)
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"

# 2. run the tests / lint / types
.venv/bin/python -m pytest -q
.venv/bin/ruff check
.venv/bin/mypy src

# 3. API (no DB: /health reports database.reachable=false, traces are dropped)
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000

# 4. API with a database
DATABASE_URL="postgresql+psycopg://rag@localhost:5432/agentic_rag" \
  .venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### Database provisioning on this machine (done, reproducible)

Postgres **18.6** was already installed and running locally; `pgvector` was
missing and installed with apt. Docker is not available, so the `docker compose`
step in `docs/03-mvp-plan.md` was replaced by a local cluster — the only
deviation from the brief, and a forced one.

Already executed here:

```bash
sudo apt-get install -y postgresql-18-pgvector        # 0.8.1
sudo -u postgres psql -c "CREATE ROLE rag LOGIN PASSWORD '<set separately>'"
sudo -u postgres createdb -O rag agentic_rag
# Extensions are not trusted: must be created by a privileged role, once.
sudo -u postgres psql -d agentic_rag -c \
  "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_trgm;"
```

Then migrations run as the unprivileged app role:

```bash
DATABASE_URL="postgresql+psycopg://rag@localhost:5432/agentic_rag" \
  .venv/bin/alembic upgrade head
```

Notes:
- The `rag` role has **no password in any file**. Supply it via `PGPASSWORD` (or
  a full URL in `DATABASE_URL`) when running commands. It is a local dev role
  with rights only on `agentic_rag`.
- `CREATE EXTENSION` requires superuser on this install, so the migration's
  `CREATE EXTENSION IF NOT EXISTS` is a no-op for the app role once extensions
  exist (documented in the migration itself). A fresh cluster needs the
  one-time privileged step above.
- `alembic downgrade base` was verified to drop everything cleanly and re-upgrade.

### Verified commands and real output

```
$ .venv/bin/ruff check
All checks passed!

$ .venv/bin/mypy src
Success: no issues found in 28 source files

$ .venv/bin/python -m pytest -q                     # no DATABASE_URL
23 passed, 6 skipped

$ DATABASE_URL=… PGPASSWORD=… .venv/bin/python -m pytest -q
29 passed
```

The 6 skipped tests are the database smoke tests; they skip with an explicit
message (`DATABASE_URL is not set …` / `DATABASE_URL is unreachable …`), never
fail.

End-to-end check against the running server (uvicorn + real Postgres):

```
$ curl -s localhost:8123/health
{"status":"ok",…,"database":{"configured":true,"reachable":true},…}

$ curl -s -X POST localhost:8123/ask -d '{"question":"What is due this week in Project Phoenix?",…}'
{"answer":"M0 placeholder: …","citations":[],"refusal":null,"trace_id":"1a3898ac-…"}

$ psql -c "SELECT agent, event, latency_ms FROM audit.events WHERE trace_id='1a3898ac-…'"
 api          | api.request.received     |
 orchestrator | orchestrator.placeholder |
 api          | api.request.completed    |         28
```

That is the M0 acceptance criterion: `POST /ask` returns a placeholder response
with a full trace.

---

## 4. Deviations from the brief

1. **Docker compose → local Postgres 18 cluster** (forced: docker is not
   installed; `pgvector` installed via apt). Postgres 18 instead of the
   documented 16 — the schema uses nothing version-specific beyond
   `gen_random_uuid()` (13+) and pgvector 0.8 (HNSW).
2. **`pgcrypto` was dropped** from the migration: `gen_random_uuid()` is built
   into Postgres 13+, so the extension was dead weight and one more privileged
   operation.
3. **`pgvector` Python package** is used for the SQLAlchemy `Vector` type
   (brief allowed "pgvector python pkg or raw sql"). The migration itself writes
   the chunk table as raw DDL because Alembic's column DSL cannot express a
   parameterised vector type or a generated `tsvector` cleanly.
4. **A few indexes/constraints not listed in `05-data-model.md` were added**
   because the listed ones are insufficient for the documented queries — flag
   for Lead review:
   - `uq_chunks_document_id_seq` — dedupe/staleness requires one row per
     `(document, seq)`.
   - `uq_mem_entity_registry_canonical_name` — otherwise aliases resolve
     ambiguously.
   - `ix_mem_session_turns_session_id_created_at` — the "last N turns" query
     needs it.
   - `ix_audit_redactions_trace_id` — redaction lookups are by trace.
   Remove any of these if the Lead prefers a strictly literal schema.
5. **`sqlalchemy[asyncio]`** (pulls `greenlet`) — required for SQLAlchemy's
   async engine on Python 3.14; found by running the tests, not assumed.
6. **`pytest-asyncio`** added as a dev dependency to test the async client and
   the async trace path. `uvicorn[standard]` (not bare `uvicorn`) so `--reload`
   and uvloop work as `DEVELOPMENT.md` describes.

No other dependencies were added. No LangChain/LlamaIndex. No git operations
(git is not installed). No secrets in code, tests, or docs.

---

## 5. Uncertainty / things a reviewer should know

- **Mypy is `strict = true` for `src/` with one narrow override**: Alembic
  migration scripts (`db.migrations.*`) keep dynamic typing, since
  `op.execute`/`context` are inherently `Any`-heavy. Everything else is fully
  annotated. `pgvector` has an `ignore_missing_imports` override (no typed stubs).
- **`/health` performs a live DB probe per call** (no caching). Fine for a dev
  endpoint; if it becomes a load-balancer target, cache the result for a few
  seconds.
- **`core.chunks.embedding` is fixed at `vector(1536)`.** Changing
  `EMBEDDING_MODEL`/`EMBEDDING_DIMENSIONS` requires a new migration; the
  migration does not read settings (deliberately — migrations must be
  deterministic).
- **`AskResponse` has no `degraded` flag**, although
  `docs/agents/00-orchestrator.md` mentions marking degradation in response
  metadata. That would change a frozen contract, so it was left for the Lead to
  decide in M1.
- **Postgres runs with whatever auth `pg_hba.conf` already had**; no system auth
  config was modified. The app role uses password auth over localhost TCP.
- **`pytest` emits one third-party deprecation warning** (Starlette's
  `TestClient` recommending `httpx2`). Not actionable in M0; it is not a
  `src/` warning and does not fail the suite.
- **Unverified:** behavior under a non-local Postgres, under load, and with a
  real LLM provider (no API key is configured; the client has only been tested
  against `httpx.MockTransport`).

---

## 6. Files created

```
pyproject.toml  alembic.ini  .env.example  .gitignore  HANDOFF.md
src/config.py
src/api/{__init__,main,routes,dependencies}.py
src/orchestrator/{__init__,contracts}.py
src/db/{__init__,base,models,session,tracing}.py
src/db/migrations/env.py, script.py.mako
src/db/migrations/versions/0001_initial_schema.py
src/llm/{__init__,base,errors,openai_compatible}.py
src/evals/{__init__,fixtures}.py
src/{connectors,indexing,retrieval,pii,evidence,synthesis,guardrails,memory}/__init__.py
tests/conftest.py
tests/test_config.py
tests/api/test_endpoints.py
tests/llm/test_openai_compatible.py
tests/db/test_migrations.py
```
