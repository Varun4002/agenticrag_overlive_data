# Agent 01 — Connector Agent

> Read with: `docs/02-architecture.md`, `docs/05-data-model.md`, `docs/03-mvp-plan.md` (M1, M2, M5).

## Mission

Pull live data from Jira, Notion, and Gmail into the canonical `core.documents`
schema — incrementally, rate-limit-aware, and read-only. Nothing else.

## Scope rules

- **Read-only.** No API call that writes/mutates a source system, ever.
- **Allowlisted scope only** (config): Jira projects, Notion pages/databases,
  Gmail labels. Skip + log everything else.
- Credentials from env only (Jira API token, Notion integration token, Gmail
  OAuth refresh token). No secrets in code, tests, or logs.

## Contract

```python
class SyncRequest(BaseModel):
    source: Literal["jira", "notion", "gmail"]
    full: bool = False          # initial sync vs incremental

class SyncResult(BaseModel):
    source: str
    documents_upserted: int
    watermark: datetime         # new core.sync_state watermark
    errors: list[ErrorReport]   # partial failures, not raised
```

Public API:
- `async def sync(req: SyncRequest) -> SyncResult` — incremental unless `full`.
- `async def fetch_targeted(source, source_id | query) -> list[CanonicalDoc]`
  — on-demand fetch used by the planner for freshness ("due today").

## Normalization (per source)

| Source | Item → doc | Incremental cursor | ACL mirror |
|---|---|---|---|
| Jira REST v3 | issue (+ 1 doc per comment) → `issue`/`issue_comment` | `updated` timestamp | project key + assignee/reporter/user in `acl.users` |
| Notion | page (+ section docs for long pages) → `page`/`page_section` | `last_edited_time` | workspace + explicit shared users |
| Gmail | message (+ thread context in `parent_source_id`) → `email` | `after:` date query | `acl.labels` — **label-scoped only** (MVP: `INBOX` + one project label) |

All three produce `CanonicalDoc` → `core.documents` rows with `content_hash`
change detection (skip re-index when hash unchanged), source timestamps
preserved, deep-link `url` set.

## Behavior details

- Exponential backoff on 429/5xx; per-source rate budget tracked in `core.sync_state`.
- Deletion detection: docs missing from a resync get `is_current=false`
  (never hard-delete; audit trail).
- Errors on one item must not abort the run — collect, continue, report.

## Failure modes

| Failure | Handling |
|---|---|
| Auth expired | typed `AuthError` → surfaced to config owner, not retried blindly |
| Rate limited | backoff; if budget exhausted, return partial `SyncResult` with errors listed |
| Malformed item | skip + log, never crash the sync |

## Tests

- Unit tests against recorded fixture payloads (`fixtures/jira/*.json` etc.) —
  **no live API calls in tests**.
- Incremental logic: watermark advances, unchanged content skipped via hash.
- Gmail label scoping: a message outside allowed labels never becomes a doc.
