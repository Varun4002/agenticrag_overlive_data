# 09 — Operator dashboard

A single-file web UI for seeing what the system is doing and triggering the work
that already exists. It is a **development/operations tool**, not a product UI:
no build step, no framework, no external assets, no auth beyond an environment
flag.

```
http://127.0.0.1:8000/ui        # the page
http://127.0.0.1:8000/admin/*   # the JSON API it calls
```

`/` redirects to `/ui`. The OpenAPI docs at `/docs` cover the admin routes too.

## What it shows

| Tab | Content |
|---|---|
| **Ask** *(landing page)* | A question box at the top. One click calls `POST /ask` and then runs the **real** retriever (`retrieval.search.retrieve`) on the same question, so you see both the pipeline's answer and the evidence it would cite. `user_id` selects the ACL identity, `top_k` the pack size, the source checkboxes scope the search, and *scrub PII* runs the **real** scrubber (`pii.scrubber.scrub`) and marks redacted spans. Enter submits, Shift+Enter is a newline. |
| **Overview** | Documents (current/superseded), chunks and how many are embedded, traces and events, PII redactions, memory tables; per-source counts; `core.sync_state`; runtime config (models, endpoint, thresholds); ACL users found in `documents.acl`; the last 12 audit events. Polls every 15 s. |
| **Documents** | Filter by source, search title/id/content, include superseded rows. Click a row to read the document text and every chunk (seq, chars, embedding presence, heading path, section, hash). |
| **Traces** | Recent traces with agent/latency roll-up; click one for the ordered event timeline with payloads. Any trace id can be opened directly. |
| **Actions** | Seed the fixture corpus, run a connector sync, run a bounded embedding backfill, clear the answer cache. The sync card is source-agnostic: the dropdown selects Jira or Gmail (Gmail when `GMAIL_ADDRESS`/`GMAIL_APP_PASSWORD` are set; Notion answers 501). A completed sync — Jira or Gmail — and any backfill that writes chunks bump the answer-cache version, so repeated questions re-run the pipeline instead of serving stale answers. The cache-clear button wraps `POST /admin/cache/invalidate`. Results are shown as JSON, with a link into the trace when one was produced. |

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/admin/overview` | Counts, health, non-secret config, ACL users, recent events |
| `GET` | `/admin/documents` | `source`, `q`, `include_history`, `limit`, `offset` |
| `GET` | `/admin/documents/{id}` | Document + chunks (`full=true` for untruncated text) |
| `GET` | `/admin/traces` | Recent trace summaries (`limit`) |
| `GET` | `/admin/traces/{trace_id}` | Full ordered timeline |
| `POST` | `/admin/search` | `{query, user_id, top_k, sources, scrub, known_person_names}` |
| `POST` | `/admin/sync` | `{source, full}` — same code path as `scripts/sync_jira.py` |
| `POST` | `/admin/backfill` | `{limit, only_missing, force, source, embedder}` — `embedder="fake"` for offline |
| `POST` | `/admin/cache/invalidate` | Drops every cached answer and bumps the data version. With `REDIS_URL`: `{invalidated: true, cleared, version}`; without it: `{invalidated: false, reason: "no answer cache configured (REDIS_URL is unset)"}`. A Redis failure is a 502. Needs no database; the Actions tab's **Clear cache** button calls it. |
| `POST` | `/admin/seed` | `{confirm: true}` — upserts `fixtures/corpus` (idempotent, refused in prod) |

Error semantics are deliberate: **400** = fix your configuration (e.g. Jira
credentials missing), **501** = connector not implemented at this milestone,
**503** = no database, **504** = the job exceeded its wall-clock cap (sync 180 s,
backfill 240 s) and the caller should re-run later — both jobs are idempotent.

## Safety

- **`ADMIN_ENABLED`** gates `/ui` and every `/admin/*` route. Unset means *on in
  dev/test, off in staging/prod*; set it explicitly to override. Disabled routes
  return 403, and `/ask` / `/health` are unaffected either way.
- **Mutating routes are serialized** (one sync and one backfill at a time per
  process) and bounded by a timeout, so a stuck Jira call or a missing Ollama
  cannot hang a browser request forever.
- **No secrets are ever returned.** The overview reports booleans and endpoints,
  never tokens; `Settings.llm_api_key` and `JIRA_API_TOKEN` are `SecretStr` and
  are not referenced by any admin response. A test asserts a planted key never
  appears in the payload.
- **Seed is refused in prod** and requires `confirm: true`, because it rewrites
  documents that share the corpus's deterministic ids.
- The dashboard binds wherever uvicorn binds. Keep it on localhost, or put it
  behind your own auth proxy if you expose it — there is no user system yet.

## Not wired yet

`/ask` now runs the full pipeline (retrieve → scrub → verify → synthesize) and
the **Ask** tab shows the real answer with its citations and trace. Still
missing, and visible as such in every trace: guardrails (M3 — the plan event
lists `guard_input`/`guard_output` as skipped), session turns and the entity
registry (M4 — the Redis answer cache is implemented and optional via
`REDIS_URL`, ruling #14), and the Notion connector. Gmail is implemented over
IMAP with an app password (ruling #12); it needs `GMAIL_ADDRESS` and
`GMAIL_APP_PASSWORD` before the sync route will run it. The Ask tab's evidence
list below the answer shows exactly what the model was given, scrubbed.

## Run it

```bash
cd /home/varun/projects/agentic_rag_over_live_data
set -a; source ~/.config/agentic-rag/dev.env; set +a

DATABASE_URL="postgresql+psycopg://rag@localhost:5432/agentic_rag" \
  .venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000/ui
```

With no `DATABASE_URL` the page still loads and states plainly that there is no
database; every data endpoint answers 503 with that reason.

## Verification

```bash
.venv/bin/python -m pytest tests/api -q     # 38 tests (2026-11-18), no database needed
.venv/bin/ruff check && .venv/bin/mypy src
```

Verified by hand against a scratch database (`agentic_rag_retrieval_test`, 44
documents / 94 chunks) with the live local embedder: overview counts, document
list and detail, `/ask` + trace timeline, retrieval with `scrub=true` returning
ranked chunks with redaction types, ACL denial for a user outside every
`acl.users`, fixture seeding (idempotent, same 44 rows), and a forced
`fake`-embedder backfill (3 chunks written, 417 tokens). Layout in a real
browser is unverified — the HTML/JS is checked by a JS syntax pass only.
