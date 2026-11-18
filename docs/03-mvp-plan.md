# 03 — MVP Plan

Build order: one vertical slice first (Jira end-to-end with no guardrails),
then harden sideways (more sources, more policies, memory). Every milestone
ends with the system **runnable** and evaluable.

## Milestones

### M0 — Skeleton (week 1)
- Repo, venv, FastAPI app + `/ask` and `/health` endpoints, config via env vars
  (`pydantic-settings`), Postgres + pgvector running.
  *As built:* local Postgres 18 cluster (docker is not installed on this
  machine); `postgresql-18-pgvector` via apt. Credentials live outside the repo
  in `~/.config/agentic-rag/dev.env`.
- Alembic migrations for the full schema in `05-data-model.md`.
- Provider-agnostic LLM client with retries/timeouts/token accounting.
- Structured tracing to `audit_events`.
- **Done when:** `POST /ask` returns a placeholder response with a full trace.

### M1 — Vertical slice: Jira only, no guardrails (week 2)
- Connector Agent: Jira adapter (issues + comments + status), normalizer →
  canonical document schema, initial full sync, incremental sync by `updated`.
- Indexer Agent: chunking, embedding, hybrid index.
- Retrieval Agent: vector + FTS search, recency boost.
- Synthesizer: single-pass answer with naive citations.
- **Done when:** CLI/`/ask` question "What tasks are due this week in project
  X?" returns a correct cited answer from synced data. This proves the
  riskiest end-to-end path early.

### M2 — Evidence & retrieval hardening (week 3)
- Evidence Agent: claim extraction, claim→chunk mapping, sufficiency score,
  refusal path ("not enough evidence").
- Notion connector (pages + databases, incremental by `last_edited_time`).
- ACL stub (single-user, but enforced in the query path).
- Orchestrator planner step (LLM): source selection + live-refresh decision.
- **Done when:** eval on 20 golden questions → ≥ 80% groundedness, refusals
  fire on out-of-corpus questions instead of hallucination.

### M3 — PII & guardrails (week 4)
- PII Scrubber Agent: Presidio pipeline on evidence packs, placeholder
  replacement, audit trail of redactions.
- Guardrail Agent: input checks, prompt-injection scanning of retrieved
  content, output re-scan (PII + citation integrity + policy).
- **Done when:** eval suite shows 0 PII leaks; injected-document attack
  (fake "ignore instructions" text planted in a Notion page) is detected and
  quarantined in tests.

### M4 — Memory & freshness (week 5)
- Memory Agent: session scratchpad, answer cache keyed to data version,
  entity registry (cross-system project/task aliases).
- Sync freshness policy + targeted on-demand fetch in the planner.
- **Done when:** repeat query hits cache with < 1s latency and invalidates
  correctly after a source change; "Project Phoenix" resolves to `PHX`
  regardless of which name the user used.

### M5 — Gmail + evals + polish (week 6)
- Gmail connector: read-only, label-scope-limited (`INBOX`, one project
  label), email → canonical doc with date-based recency.
- Eval harness (`09-eval-observability-agent.md`): groundedness scorer, PII
  leak scan, refusal correctness, latency budget report.
- **Done when:** MVP acceptance criteria in `01-vision-scope.md` met.

## MVP cut lines

**In MVP:** Jira, Notion, Gmail (read-only, label-scoped); hybrid retrieval;
PII redaction; evidence verification + refusal; 3 guardrail layers (input,
injection, output); session memory + answer cache + entity registry; CLI +
HTTP API; eval suite.

**Out of MVP (post-MVP backlog):**
- Slack/Drive/Calendar connectors
- Write-back actions (create/update issues, send mail)
- Reversible redaction with per-user unredact rights
- Multi-tenant + SSO + per-team ACL mirroring
- Web UI / streaming responses
- Multi-hop agentic planning beyond single planner step
- Alternative vector stores, reranker models (add only if evals demand)

## Risk register

| Risk | Mitigation |
|---|---|
| Gmail data is messy (threads, signatures, PII-dense) | Scoping MVP Gmail to one label; email-specific chunking rules; PII scrubber exercised hardest on this source |
| Rate limits on API pulls | Incremental sync + exponential backoff + per-connector rate budget |
| Prompt injection in synced content | Injection scanner at ingestion *and* retrieval (defense in depth) |
| Presidio false negatives (missed PII) | Output-layer re-scan is a second net; regex catch-list; eval asserts 0 leaks |
| Embedding cost/latency at scale | Batch on shared server; cache by content hash; dedupe chunks |
| Planner over-fetching (LLM decides to re-sync too often) | Hard rules cap live fetches (staleness threshold + per-request budget) |
