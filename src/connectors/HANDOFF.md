# HANDOFF — Connector Agent (component 01, wave 1 / M1)

Status: **M1 complete for Jira** (2026-11-17). Notion and Gmail are typed stubs
that raise `NotImplementedError` and make no network call.

Gates on this machine, real output:

```
$ .venv/bin/python -m pytest -q
281 passed, 39 skipped, 1 warning in 0.72s

$ CONNECTOR_TEST_DATABASE_URL=<scratch db> .venv/bin/python -m pytest tests/connectors -q
75 passed, 1 warning in 0.59s

$ .venv/bin/python -m pytest tests/connectors -q     # no scratch DB: 5 DB tests skip
70 passed, 5 skipped, 1 warning in 0.17s

$ .venv/bin/ruff check
All checks passed!

$ .venv/bin/mypy src
Success: no issues found in 46 source files
```

Frozen interfaces untouched: `src/orchestrator/contracts.py` and migration
`0001_initial_schema` were not edited. No new dependencies (httpx/SQLAlchemy/
pydantic were already in `pyproject.toml`).

---

## Update — 2026-11-18: Gmail connector implemented over IMAP

Supersedes the Gmail row of §2 and the "Gmail is a stub" claims in the status
block above; the rest of this document is unchanged. The adapter was built by a
separate agent — this addendum records what landed.

**What replaced the stub.** `src/connectors/gmail.py` is now a real read-only
adapter instead of the `NotImplementedError` stub. `GmailConnector` does the
label-scoped sync and `fetch_targeted`, behind a narrow `ImapConnection`
protocol (`select` / `uid_search` / `uid_fetch` / `logout`). Production is
`ThreadedImapConnection` (`imaplib.IMAP4_SSL` on a worker thread, client opened
lazily so construction never dials out); tests inject an in-memory fake.
`service.sync(source="gmail")` and `service.build_connector("gmail")` dispatch to
it, and `describe_settings` reports `gmail_configured` / `gmail_address` /
`gmail_labels` without the app password.

**Binding deviations** (Lead ruling #12 in `docs/08-decisions.md`):

- **IMAP, not the Gmail REST API.** This deployment has an app password and no
  OAuth client; IMAP gives the same label-scoped read access using the standard
  library — no new dependency, no token-refresh machinery.
- **No deletion sweep.** Unlike Jira, a full sync does not retire documents that
  vanished upstream: IMAP exposes no reliable deleted-signal, so mail is treated
  as append-mostly.
- **Cross-mailbox dedupe.** Gmail shows one message under every label it
  carries, so a message sitting in two allowlisted labels is written **once**
  per run (processed ids are tracked across mailboxes) instead of twice with a
  drifting ACL.
- **Stable ids**, in fallback order: `X-GM-MSGID` → RFC 822 `Message-ID` →
  content hash. A server without the Gmail extension is retried without
  `X-GM-MSGID` and falls back to `Message-ID`.
- **Read-only by protocol.** Every mailbox is opened with `EXAMINE`
  (`select(readonly=True)`) and bodies are fetched with `BODY.PEEK[]`, so a sync
  never marks a message read, and never mutates, moves, or sends anything.

**New environment variables** (`connectors.settings`, all present in
`.env.example`): `GMAIL_ADDRESS` (also the ACL identity of every mail document),
`GMAIL_APP_PASSWORD` (`SecretStr`, never logged), `GMAIL_IMAP_HOST` (default
`imap.gmail.com`), `GMAIL_IMAP_PORT` (default 993), `GMAIL_PAGE_SIZE` (default
50), `GMAIL_MAX_MESSAGES` (default 200, the per-label cap on a run) — plus the
existing `GMAIL_LABELS` allowlist (default `INBOX`).

**Tests:** `tests/connectors/test_gmail_sync.py`, 29 tests against a fake IMAP
connection — no network, no database. Service dispatch, config reporting and
targeted-fetch validation live in `tests/connectors/test_service.py`.

**Unverified:** no live IMAP login has been attempted — there are no Gmail
credentials on this machine. Everything above is exercised against the fake
connection only, so the first real sync is the integration step: confirm login,
label selection, `X-GM-MSGID` availability, and that the per-label cap behaves
on a real inbox.

---

## 1. What is implemented

### `base.py` — shared async HTTP framework

- Read-only by construction: only `GET` exists on the base class.
- Auth is supplied by the subclass from environment-sourced settings; nothing is
  logged (the Jira token is a `SecretStr`, header built at construction time).
- Retries: bounded exponential backoff **with full jitter**, `Retry-After`
  honoured (delta-seconds form, capped by `backoff_max_s`), retryable =
  `408/409/425/429/5xx` + transport failures/timeouts.
- Typed errors, retryable/terminal split exactly like `src/llm/errors.py`:
  `ConnectorError` → `ConnectorConfigError` / `ConnectorAuthError` /
  `ConnectorNotFoundError` / `ConnectorBadRequestError` /
  `ConnectorRateLimitError` / `ConnectorServerError` / `ConnectorTimeoutError` /
  `ConnectorResponseError` / `ConnectorRateBudgetExhausted`.
- `RateBudget`: fixed-window per-run request counter. Every **attempt** spends
  budget (a retry is a real request); exhaustion raises
  `ConnectorRateBudgetExhausted`, which the Jira adapter turns into a partial
  `SyncResult` instead of hammering the API.
- `ConnectorStats` (requests / retries / Retry-After hits) surfaced into
  `SyncResult`.
- `paginate()`: offset pagination helper (`startAt`/`maxResults`/`total`),
  reusable by Notion/Gmail-adjacent endpoints that page the same way.

### `jira.py` — Jira Cloud REST API v3 adapter

- **Full sync** of the allowlisted projects via JQL
  (`project IN ("PHX") [AND (...) ] ORDER BY updated ASC`).
- **Incremental sync** from the `updated` cursor in `core.sync_state`
  (`updated >= "<stored watermark, minute precision>"`). JQL formatting is
  minute-resolution and `>=`, so the boundary issue is re-fetched and then
  skipped by hash — deliberately inclusive rather than lossy.
- **Doc mapping**: one `issue` doc + one `issue_comment` doc per comment.
  Issue doc content renders summary, project, type, status, priority,
  resolution, assignee, reporter, due date, labels, created/updated, deep-link
  URL and the ADF description flattened to text. Comment docs carry the parent
  issue key/summary, author, timestamps, a `?focusedCommentId=` deep link and the
  body (code blocks flattened).
- **ADF → text**: paragraphs, headings, bullet/ordered lists, list items,
  code blocks, blockquotes, panels, tables, mentions, emoji, inline/block cards,
  media placeholders, link marks (`text <href>`), hard breaks. Plain-string
  bodies (older instances) also accepted.
- **ACL mirror** (`acl` jsonb, documented identity conventions):
  `users` = `account:<accountId>` and/or `email:<address>` for
  assignee + reporter + comment author; `groups` = `jira:project:<KEY>`;
  `labels` = `[]`. Lists are sorted/de-duplicated so ACL drift detection is a
  plain equality check.
- **Change detection**: `content_hash` = sha256 over title + content; unchanged
  content *and* unchanged ACL → the row is skipped (`UpsertOutcome.unchanged`).
  An ACL-only change updates the row.
- **Supersession**: changed content retires the outgoing row
  (`is_current=false`) and inserts a new current row — never a hard delete.
- **Deletion detection**: a full sync sweeps the source scope and retires
  documents missing from the result *and their comments*; an incremental sync
  only retires comments that vanished from an issue it just fetched. Items that
  errored during the run are excluded from the sweep, so a transient failure
  never looks like a deletion.
- **Per-item isolation**: every issue+comments is one SAVEPOINT
  (`DocumentStore.unit_of_work`), so one failure rolls back alone; all other
  items commit. Failures become `ErrorReport`s (stage
  `fetch`/`normalize`/`persist`), and the run continues.
- **Cursor safety**: if an item fails, the new watermark is clamped to one second
  before that item's `updated` so its next change cannot be skipped. A failed
  item therefore retries on the next run (by design; a permanently malformed
  item keeps the cursor pinned there and reports an error each run).
- **Rate budget exhaustion** → partial `SyncResult` with an `ErrorReport`
  (per the brief's failure table). **Auth failure** → state recorded as `error`
  and `ConnectorAuthError` re-raised (a human must fix credentials).
- **Allowlist enforcement on the payload**, not only in JQL: an issue whose
  `fields.project.key` is not allowlisted is skipped and counted
  (`documents_skipped`), never ingested.
- **Pagination**: search follows whichever shape the tenant serves — the classic
  offset form (`startAt`/`total`) or the new token form
  (`nextPageToken`/`isLast`) — because Jira Cloud is mid-migration between them.
  Comments use the offset helper.
- `fetch_targeted(source_id=...)` (single issue + comments) and
  `fetch_targeted(query=...)` (JQL fragment ANDed with the project allowlist,
  bounded by `limit`). Pure fetch: **writes nothing to the database**.
- Guards: project keys and issue keys are regex-validated before being
  interpolated into JQL/paths (`JIRA_PROJECTS='PHX") OR ("1'` is rejected).

### `normalizer.py` — canonical model + persistence

- `CanonicalDoc` (Pydantic v2, `extra="forbid"`), owned here, 1:1 with
  `core.documents`; `content_hash` auto-filled; all timestamps
  `AwareDatetime` (naive values are rejected).
- `Acl` model matching the documented `{users, groups, labels}` jsonb shape.
- `jira_issue_to_docs()` / `jira_comment_source_id()` /
  `parse_jira_timestamp()` / `adf_to_text()`.
- Source-id convention (relied on by the indexer/retrieval agents):
  - issue: `PHX-101` (plain native key), `parent_source_id = NULL`
  - comment: `PHX-101/comment/20101`, `parent_source_id = 'PHX-101'`
- `DocumentStore` port + `apply_document()` / `retire_missing()` holding all the
  decision logic, and two implementations:
  - `SqlAlchemyDocumentStore` — real `core.documents` / `core.sync_state`,
    savepoints, `ON CONFLICT` sync-state upsert. Owns no transaction: the caller
    commits.
  - `InMemoryDocumentStore` — shipped in the package (not in tests) so the
    semantics have one executable definition; the tests drive the sync logic
    through it, and a Postgres test asserts the SQL store behaves identically.

### `service.py` — thin orchestration

- `async def sync(req, *, connector=None, transport=None, session_factory=None, settings=None) -> SyncResult`
- `async def fetch_targeted(source, source_id=None, *, query=None, limit=10, transport=None, settings=None) -> list[CanonicalDoc]`
- Owns the session lifecycle for Jira (creates the session, commits on success,
  commits the recorded failure state before re-raising a `ConnectorError`,
  rolls back on anything else, always closes the HTTP transport).
- `build_connector(source, ...)`, `describe_settings()` (non-secret diagnostics),
  `SUPPORTED_SOURCES`. Unknown source → `ConnectorConfigError`.

### `scripts/sync_jira.py`

Manual entrypoint: `--full`, `--json`, `--source {jira,notion,gmail}`,
`--verbose`. Exit codes: `0` clean, `1` partial (collected errors), `2` run
failed (auth/config/database), `3` not configured / not implemented. Prints a
summary; never prints credentials.

### Fixtures `fixtures/jira/*.json`

Hand-built Jira v3 payloads (**not recorded from a live tenant** — stated in
`fixtures/jira/README.md`): issue with comments, issue with no due date /
unassigned, issue with PII (name, e-mail, phone), issue whose comment contains an
embedded instruction ("Ignore all previous instructions…"), an updated revision
(with a deleted comment), a single-issue GET payload, a malformed-issue page.

### Tests `tests/connectors/` — 75 tests, no live API calls

`httpx.MockTransport` + fixtures throughout; `test_db_store.py` is the only
module that touches Postgres and it skips unless a scratch DB is provided.

Coverage: full sync upsert, offset **and** token pagination, incremental
watermark advance (asserts the JQL cursor string), unchanged-hash skip,
comment docs, ACL content, supersession (1st and 3rd revision), comment
retirement, last-comment deletion, full-sync deletion sweep (issue + children),
incremental runs never retiring the scope, allowlist skip, malformed item →
`ErrorReport` with the run continuing, failed item not treated as deleted,
watermark clamping, auth error → raised + state recorded, budget exhaustion →
partial result, `fetch_targeted` by id/query (+ injection guard, + no writes),
Gmail label-scoping fail-closed behaviour and stub no-fetch behaviour, Notion
stub, settings/secret redaction, plus the framework suite (retries, jitter,
`Retry-After`, terminal errors, budget, pagination).

---

## 2. What is stubbed / not done

| Item | State |
|---|---|
| Notion connector | Stub (`NotionConnector`), raises `NotImplementedError` (M2). Only the configured `NOTION_DATABASE_IDS` is stored. |
| Gmail connector | **Superseded — implemented over IMAP (2026-11-18); see the Update at the top of this file.** |
| Persisted rate budget | In-memory per run. `core.sync_state` has no budget column (schema frozen). See §6. |
| Worker / scheduler | Not started (M1 scope was the adapter + manual entrypoint). `scripts/sync_jira.py` is the runnable path. |
| Audit events from the connector | Not written. The connector returns structured `SyncResult`/`ErrorReport`; wiring `audit.events` needs a `trace_id` policy the orchestrator owns. See §6. |
| Notion section docs, Gmail threads | Not implemented (M2/M5). |

---

## 3. How to run

### Environment (environment variables only — never commit values)

| Variable | Meaning |
|---|---|
| `JIRA_BASE_URL` | e.g. `https://acme.atlassian.net` |
| `JIRA_EMAIL` | account e-mail for HTTP Basic auth |
| `JIRA_API_TOKEN` | API token (`SecretStr`, never logged) |
| `JIRA_PROJECTS` | **required allowlist**, comma-separated keys (`PHX,OPS`) |
| `JIRA_JQL_EXTRA` | optional JQL fragment ANDed with the project scope |
| `JIRA_PAGE_SIZE` | 1–100, default 50 |
| `JIRA_MAX_RETRIES` / `JIRA_BACKOFF_BASE_S` / `JIRA_REQUEST_TIMEOUT_S` | retry policy |
| `JIRA_RATE_BUDGET_CALLS` / `JIRA_RATE_BUDGET_WINDOW_S` | per-run request budget |
| `JIRA_SEARCH_PATH` | default `/rest/api/3/search/jql` (override for tenants that only serve `/rest/api/3/search`) |
| `GMAIL_LABELS` | allowlisted labels (used by the M5 adapter; scoping helper is live) |
| `NOTION_DATABASE_IDS` | allowlisted Notion databases (M2) |
| `DATABASE_URL` | required for `sync()` (persistence) |

These are **not** in `.env.example` / `src/config.py` yet: both files are outside
this component's write scope. Recommended follow-up for the Lead (§6).

### Commands

```bash
cd /home/varun/projects/agentic_rag_over_live_data

# credentials + database for local dev
set -a; source ~/.config/agentic-rag/dev.env; set +a

# tests (no network, no database required)
.venv/bin/python -m pytest tests/connectors -q

# lint / types
.venv/bin/ruff check src/connectors scripts tests/connectors
.venv/bin/mypy src

# manual sync
.venv/bin/python scripts/sync_jira.py            # incremental
.venv/bin/python scripts/sync_jira.py --full     # full resync (+ deletions)
.venv/bin/python scripts/sync_jira.py --json
```

### Postgres-backed tests (scratch database, never the dev DB)

```bash
# one-time
sudo -u postgres createdb -O rag agentic_rag_conn_test
sudo -u postgres psql -d agentic_rag_conn_test \
  -c "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_trgm;"
set -a; source ~/.config/agentic-rag/dev.env; set +a
DATABASE_URL="postgresql+psycopg://rag@localhost:5432/agentic_rag_conn_test" \
  .venv/bin/alembic upgrade head

# run
set -a; source ~/.config/agentic-rag/dev.env; set +a
CONNECTOR_TEST_DATABASE_URL="postgresql+psycopg://rag@localhost:5432/agentic_rag_conn_test" \
  .venv/bin/python -m pytest tests/connectors -q      # 75 passed
```

`test_db_store.py` requires **`CONNECTOR_TEST_DATABASE_URL`** (a separate
variable, on purpose) and refuses to run if it names the same database as
`DATABASE_URL`; it truncates `core.documents` / `core.sync_state`. Without the
variable the five DB tests skip with instructions — never fail.

The scratch DB `agentic_rag_conn_test` is left in place for reruns.

### Verified end to end (real Postgres, fake Jira over HTTP)

`scripts/sync_jira.py` was run against a throwaway local HTTP server serving the
fixtures and the scratch database:

```
source: jira
mode: full
documents upserted: 6
documents unchanged: 0
documents retired: 0
documents skipped: 0
watermark: 2026-11-16T14:22:00+00:00
requests: 5 (retries: 0)
errors: 0

# second (incremental) run: documents_upserted 0, documents_unchanged 6
```

and the rows landed as expected (`PHX-101`, `PHX-101/comment/20101`,
`PHX-101/comment/20102`, `PHX-102`, `PHX-103`, `PHX-103/comment/20103`;
`core.sync_state` = `watermark 2026-11-16 19:52+05:30 / status ok`).

---

## 4. Deviations from the brief (Lead approval requested)

1. **Connector settings live in `src/connectors/settings.py`, not
   `src/config.py`.** `src/config.py` and `.env.example` are outside this
   component's write scope while other agents edit the tree, so a second
   `BaseSettings` class (`ConnectorSettings`, no prefix, `SecretStr` secrets)
   reads `JIRA_*` / `GMAIL_*` / `NOTION_*` with the same conventions. Fold it
   into `Settings` and `.env.example` when convenient; nothing else changes.

2. **Superseded rows get a `#history-<8 hex of row id>` suffix on `source_id`.**
   The frozen schema has `UNIQUE (source, source_id, is_current)`, i.e. at most
   **one** non-current row per item. Retaining a second revision is then
   impossible without either hard-deleting history or freeing the slot.
   Resolution chosen: on retirement, `is_current=false` **and** `source_id`
   becomes `PHX-101#history-1a2b3c4d` in one `UPDATE`. Consequences:
   - only the **current** row always carries the native id, so lookups that
     filter `is_current` are unaffected (all of ours are);
   - unlimited revision history is retained, no row is ever deleted;
   - `source_id` of superseded rows is no longer the native id — tell me if any
     later component wants to query history by native id and I will add a
     dedicated column instead.

   **Preferred real fix (schema change, Lead decision):** replace the unique
   constraint with a partial unique index
   `CREATE UNIQUE INDEX uq_documents_source_source_id_current
    ON core.documents (source, source_id) WHERE is_current` — then history rows
   can keep the native `source_id` and the suffix hack can be removed.

3. **Per-run rate budget is not persisted.** The brief says "per-source rate
   budget tracked in `core.sync_state`"; that table has no column for it and the
   migration is frozen. The budget is therefore a per-process fixed-window
   counter. If it must survive restarts, add e.g.
   `core.sync_state.budget_json jsonb` (new migration).

4. **`SyncResult` has extra fields** beyond `source / documents_upserted /
   watermark / errors`: `documents_unchanged`, `documents_retired`,
   `documents_skipped`, `full`, `started_at`, `finished_at`, `requests`,
   `retries`. The four required fields are unchanged and are all the frozen
   contract needs; drop the rest if the Lead prefers a strictly literal shape.

5. **Jira search endpoint is `/rest/api/3/search/jql`** (current Jira Cloud
   endpoint) with adaptive pagination; the classic `/rest/api/3/search` shape is
   also parsed, and `JIRA_SEARCH_PATH` overrides the path. Neither shape has been
   verified against a live tenant (no credentials, no live calls allowed).

6. **`InMemoryDocumentStore` ships in `src/connectors/normalizer.py`**, not in
   tests only, so the supersession/retirement semantics have exactly one
   definition that both stores follow. It is not used by production code paths.

7. **Errors are treated asymmetrically**: per-item failures and budget
   exhaustion return a partial `SyncResult`; auth failures raise. The brief only
   specified the auth case ("surfaced, not retried blindly"); a raised error
   seemed right since a partial result from dead credentials is noise.

---

## 5. Uncertainty / risks a reviewer should know

- **No live Jira call was ever made** (no credentials on this machine, and the
  brief forbids live calls in tests). Everything is verified against hand-built
  payloads over `httpx.MockTransport` and one local fake server. The shapes most
  likely to differ in reality: the exact `search/jql` response envelope, comment
  pagination fields, and whether `description`/`comment.body` arrive as ADF
  (assumed; plain strings also accepted).
- **JQL date timezone**: JQL date literals are interpreted in the *user's* Jira
  timezone. The adapter writes UTC strings; if the syncing account is not on UTC,
  the incremental window can be off by the account offset in one direction. It
  is safe today because `>=` is inclusive and the boundary issue is re-fetched,
  but an account offset *ahead* of UTC would widen the window (extra work, not
  lost data). Fix would be to read the account's timezone from
  `/rest/api/3/myself` — deliberately not done (extra request, unverifiable).
- **Watermark granularity is minutes** (JQL limitation). Combined with `>=` and
  content hashing this is safe, at the cost of re-fetching the boundary item.
- **Deletion detection only runs on full syncs** for whole issues (incremental
  runs cannot prove absence); deleted *comments* are detected on both. A
  permanently failing item keeps the watermark pinned before it — intended, but
  it means one broken item makes every subsequent run re-scan from that point.
- **Superseded rows keep their chunks.** When a document is superseded, its old
  `core.documents` row stays (audit) and any `core.chunks` rows that reference it
  stay too — the indexer (component 02) must re-chunk the *current* row and must
  filter joins on `documents.is_current`. Flagging this because the two
  components meet here and I cannot edit `src/indexing/`.
- **`core.sync_state.last_status`** uses `ok` / `partial` / `error` (free-form
  column); agree the vocabulary with the Lead if anything reads it.
- `apply_document` compares the full ACL jsonb for equality: a Jira-side rename
  of a person's display name changes nothing (only `account:`/`email:`
  identifiers are mirrored), but an e-mail change does rewrite both the issue and
  its comments.
- **Mypy strict** passes; the only non-obvious typing escape hatches are
  `# type: ignore`-free — `SqlAlchemyDocumentStore` uses real SQL expressions
  (`func.concat`, `cast(..., Text)`) rather than raw SQL strings.
- Performance: one comment request per issue (Jira has no bulk comment endpoint
  on search results). For 5k issues that is 5k+ requests; the budget/backoff
  machinery handles throttling, but a full sync of a large tenant will take a
  while and should run on the shared server, not the laptop.

---

## 6. Requests to the Lead

1. **Schema decision needed** on
   `uq_documents_source_source_id_is_current` (see deviation 2) — either keep the
   suffix convention or replace the constraint with a partial unique index. My
   preference is the partial index; the code change is then deleting one
   `func.concat` argument.
2. **Fold connector config** into `src/config.py` + `.env.example` (the variable
   table in §3 is the list to add).
3. **Decide whether sync runs should emit `audit.events`** (agent
   `connector`, events e.g. `connector.sync.started` / `.completed` /
   `connector.item.failed`, payload = `SyncResult` counters). The `Tracer` in
   `src/db/tracing.py` is importable; I did not wire it because `trace_id`
   semantics for background syncs (vs per-question traces) belong to the
   orchestrator/Lead.
4. **Indexer coordination**: chunks must be regenerated for superseded documents
   and retrieval must filter on `documents.is_current` (see §5).
5. If you want a **worker loop** (`src/worker.py`), say so — this component
   exposes exactly what it needs (`service.sync`, `service.fetch_targeted`, and
   `SYNC_STALE_AFTER_S` for freshness checks).

---

## 7. Files created / changed

```
src/connectors/__init__.py        (module map)
src/connectors/base.py            framework: retries, budget, errors, pagination
src/connectors/settings.py        ConnectorSettings / JiraCredentials
src/connectors/contracts.py       SyncRequest / SyncResult / ErrorReport / Connector
src/connectors/normalizer.py      CanonicalDoc, ADF, ACL, stores, upsert/retire logic
src/connectors/jira.py            Jira Cloud v3 adapter
src/connectors/notion.py          stub (M2)
src/connectors/gmail.py           stub (M5) + label-scoping helper
src/connectors/service.py         sync() / fetch_targeted() / build_connector()
src/connectors/HANDOFF.md         this file
scripts/sync_jira.py              manual entrypoint
fixtures/jira/README.md           "hand-built, not recorded" notice + file map
fixtures/jira/search_page_1.json
fixtures/jira/search_page_2.json
fixtures/jira/search_updated_page.json
fixtures/jira/comments_phx_101.json
fixtures/jira/comments_phx_101_updated.json
fixtures/jira/comments_phx_102.json
fixtures/jira/comments_phx_103.json
fixtures/jira/issue_phx_102.json
fixtures/jira/malformed_issue_page.json
tests/connectors/conftest.py
tests/connectors/jira_fixtures.py       (fixture loader + fake Jira server)
tests/connectors/test_base.py           (16 framework tests)
tests/connectors/test_normalizer.py     (14 normalization tests)
tests/connectors/test_jira_sync.py      (24 sync-behaviour tests)
tests/connectors/test_service.py        (16 service/stub/config tests)
tests/connectors/test_db_store.py       (5 Postgres tests, skip without scratch DB)
```

No files outside `src/connectors/`, `scripts/`, `fixtures/jira/`,
`tests/connectors/` were modified.
