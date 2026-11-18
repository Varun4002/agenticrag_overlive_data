# 02 — Architecture

## Component map

```
                        ┌─────────────────────────────────────────┐
                        │              Orchestrator               │
                        │  (state machine + LLM planning steps)   │
                        └───┬──────────┬──────────┬──────────┬────┘
                            │          │          │          │
        ┌───────────────────┘   ┌──────┘    ┌─────┘     ┌────┘
        ▼                       ▼           ▼           ▼
┌──────────────┐   ┌─────────────────┐  ┌──────────┐ ┌──────────┐
│ Connector    │   │ Retrieval Agent │  │ Memory   │ │ Guardrail│
│ Agent        │──▶│ + Indexer Agent │  │ Agent    │ │ Agent    │
│ (live fetch) │   │ (hybrid search) │  └──────────┘ └──────────┘
└──────┬───────┘   └────────┬────────┘
       │                    │ evidence pack (chunks + metadata)
       ▼                    ▼
┌──────────────┐   ┌─────────────────┐   ┌──────────────────────┐
│ Postgres:    │   │ PII Scrubber    │──▶│ Evidence Agent       │
│ raw docs,    │◀─▶│ Agent           │   │ (groundedness check, │
│ chunks,      │   └─────────────────┘   │  claim→citation map) │
│ vectors,     │                         └──────────┬───────────┘
│ memory, audit│                                    ▼
└──────────────┘                         ┌──────────────────────┐
                                         │ Synthesizer Agent    │
                                         │ (cited answer gen)   │
                                         └──────────┬───────────┘
                                                    ▼
                                         Output guardrail re-scan → user
```

**Operator surface (implemented).** `src/api/admin.py` plus a single static page
(`src/api/static/index.html`) serve `/ui` backed by `/admin/*`: document and
chunk browsers, trace timelines, a retrieval + scrub preview, fixture seed,
Jira/Gmail sync, bounded embedding backfill and cache invalidation. It is a
development/operations tool, not a product surface — enabled by `ADMIN_ENABLED`
(default off outside dev/test), no user system of its own. Ruling #11 in
`docs/08-decisions.md`, details in `docs/09-operator-dashboard.md`.

## Request lifecycle

1. **Question in** → Guardrail Agent runs input checks (scope, injection
   heuristics).
2. **Orchestrator** checks Memory Agent: (a) cached answer still fresh?
   (b) resolved entities for names in the question? Either can short-circuit
   or augment planning. What exists today is Tier 2 only: a Redis answer cache
   keyed by `sha256(question | user_id | data version)`, invalidated by a version
   bump after any sync/backfill/seed write (ruling #14). The watermark /
   `content_hash` freshness rule in `07-memory-design.md` is not built, and
   there is no entity resolution yet.
3. **Planner (LLM step)** produces a small plan: which sources to query,
   which entity filters to apply, whether a live re-fetch is needed
   (e.g., "what's *due today*" → re-sync Jira first if data is stale).
4. **Retrieval Agent** runs hybrid search (pgvector + `pg_trgm` full-text,
   recency boost, ACL filter) and optionally triggers Connector Agent for
   targeted live fetches.
5. **PII Scrubber** redacts personal data from every chunk in the evidence
   pack. Redaction events go to the audit table.
6. **Evidence Agent** (LLM step) extracts atomic claims, maps each to ≥1 cited
   chunk, drops unsupported material, computes an evidence sufficiency score.
7. **Synthesizer** (LLM step) writes the answer restricted to verified claims,
   inline citations `[1]`, and the refusal path if sufficiency < threshold.
8. **Output guardrails**: re-scan answer for PII, verify citation integrity,
   block policy violations → response.
9. **Memory Agent** writes the session turn, updates the answer cache
   (keyed to data version), updates entity registry hits.

## Key design decisions (and trade-offs)

| Decision | Rationale | Trade-off accepted |
|---|---|---|
| **No LangChain/LlamaIndex** | The agentic loop here is small and domain-specific (6 states). A hand-rolled state machine is debuggable, avoids dependency churn, and every step is observable. | We write our own retry/tracing glue; revisit LangGraph if the planner grows to conditional multi-hop territory. |
| **Postgres for everything** (docs, chunks, vectors, memory, audit) | One datastore, one migration story, `pgvector` is adequate at this scale (< 1M chunks). | Not as fast as a dedicated vector DB; fine until ~10M chunks, at which point revisit. |
| **Pull sync, not webhooks, for MVP** | Webhooks need public endpoints + per-vendor verification work. Scheduled incremental polling is simpler and reliable. | Data freshness bounded by sync interval (default 15 min); questions trigger targeted on-demand fetches to cover the gap. |
| **Irreversible PII redaction at retrieval time** | Simplest correct policy: the model can never leak what it never saw. Redaction happens per evidence pack, not on stored data. | Same person mentioned across chunks is redacted inconsistently unless the entity registry links placeholders — accepted for MVP, handled by registry in `07-memory-design.md`. |
| **Evidence Agent as a separate LLM pass** | Groundedness is checkable before generation, and refusal decisions get an auditable trace. | +1 LLM call (~1–2s latency). Worth it — this is the anti-hallucination core. |
| **ACL mirrored at ingestion, enforced at query time** | Storing per-chunk ACL lets filtering happen in the SQL WHERE clause — fast and hard to bypass. | ACL drift between source and mirror; refresh each sync run. |
| **Fail closed when the audit trail is unavailable** | An answer that cannot be audited cannot be investigated. `AUDIT_STRICT` (default on) makes a failed trace write a 503, and makes prod-without-DB a startup-level refusal. | Availability depends on the DB; dev must explicitly opt out. Guardrail spec: `06-guardrails-pii.md`, tests: `tests/api/test_audit_policy.py`. |

## Directory layout (target)

```
agentic_rag_over_live_data/
├── docs/                      # this doc suite
├── src/
│   ├── api/                   # FastAPI routes, auth
│   ├── orchestrator/          # state machine, planner
│   ├── connectors/            # jira.py, notion.py, gmail.py, base.py, normalizer
│   ├── indexing/              # chunker, embedder, indexer
│   ├── retrieval/             # hybrid search, ranker, evidence pack
│   ├── pii/                   # presidio wrapper, policies
│   ├── evidence/              # claim extraction, citation mapper
│   ├── synthesis/             # answer generation, refusal
│   ├── guardrails/            # input/retrieval/output rules
│   ├── memory/                # scratchpad, cache, entity registry
│   ├── llm/                   # provider-agnostic client
│   ├── db/                    # models, alembic migrations
│   └── evals/                 # golden set, harness
└── tests/
```

## Concurrency & execution model

- Sync jobs and embedding batches run as background workers (asyncio tasks
  started from a `worker.py` entrypoint; heavy batch embedding runs on the
  shared server via a `sync-remote` script).
- Question answering is async end to end; each LLM call carries a timeout and
  a typed failure that the orchestrator maps to a user-visible message.

## Observability

Every request gets a `trace_id`. Each agent emits structured events
(agent name, latency, token counts, decision + reason) to the `audit_events`
table. Eval failures and guardrail blocks are first-class trace outcomes.
