# Agentic RAG over Live Data

Retrieval-augmented question answering over **live** work data — Jira, Notion,
Gmail — that answers questions about project goals, tasks, deadlines, and
reports. Every answer is grounded in cited evidence, personal data is removed
before it reaches the model, and a layered guardrail system decides what the
system may see, say, and remember.

## One-line flow

```
Question → Plan → Fetch live data (Jira/Notion/Gmail) → Chunk/Index
        → Retrieve (hybrid + ACL) → Strip PII → Verify evidence
        → Synthesize answer with citations → Output guardrails → Reply
                                              ↘ Memory (re-access cache)
```

## Doc index

| Doc | Purpose |
|---|---|
| `docs/01-vision-scope.md` | What this is, who it's for, example queries, non-goals |
| `docs/02-architecture.md` | Components, data flow, stack, key design decisions |
| `docs/03-mvp-plan.md` | MVP scope, milestones M0–M5, cut lines, acceptance criteria |
| `docs/04-orchestration.md` | How the build is orchestrated — agent assignments, build DAG, hand-off contracts |
| `docs/05-data-model.md` | Postgres schema: documents, chunks, embeddings, memory, audit |
| `docs/06-guardrails-pii.md` | Full guardrail catalog + PII policy |
| `docs/07-memory-design.md` | Working / episodic / semantic memory design |
| `docs/08-decisions.md` | Lead rulings on every sub-agent deviation (binding) |
| `docs/09-operator-dashboard.md` | The `/ui` dashboard: what it shows, its API, and its safety gates |

## Sub-agent briefs (`docs/agents/`)

Each file is a complete brief for one sub-agent. An agent gets **only** its own
brief plus `02-architecture.md` and `05-data-model.md`; the interfaces in those
docs are frozen so agents can build in parallel.

| Agent | Brief |
|---|---|
| `00-orchestrator.md` | Planner/dispatcher — the only component that calls the LLM for multi-step reasoning |
| `01-connector-agent.md` | Jira / Notion / Gmail adapters, normalization, incremental sync |
| `02-indexer-agent.md` | Chunking, embedding, hybrid index (pgvector + FTS), ACL metadata |
| `03-retrieval-agent.md` | Hybrid retrieval, recency/freshness ranking, evidence pack assembly |
| `04-pii-scrubber-agent.md` | PII detection + redaction on evidence before it reaches the LLM |
| `05-evidence-agent.md` | Groundedness verification: every claim maps to a cited chunk, else dropped/refused |
| `06-synthesizer-agent.md` | Final answer generation with citations, refusal messaging |
| `07-guardrail-agent.md` | Input/retrieval/output guardrails incl. prompt-injection defense |
| `08-memory-agent.md` | Session scratchpad, answer cache, project entity registry |
| `09-eval-observability-agent.md` | Golden question set, eval harness, tracing, PII-leak tests |

## Stack (assumed defaults — change here if wrong)

- **Backend:** Python 3.14 + FastAPI, Pydantic v2, SQLAlchemy 2.0 + Alembic
- **Store:** PostgreSQL 16 + `pgvector` + `pg_trgm` (hybrid search, no extra DB)
- **Cache:** Redis (`redis>=5.0`, optional via `REDIS_URL`) for the answer cache;
  Postgres stays the source of truth and the cache is never required (ruling #14)
- **Connectors:** `httpx` against the Jira/Notion REST APIs, plus Gmail over IMAP
  with an app password (no OAuth client — ruling #12); **no** LangChain/LlamaIndex for the MVP; the orchestrator is a deterministic state machine with LLM planning steps (rationale in `02-architecture.md`)
- **Operator dashboard:** single-file static HTML at `/ui`, backed by `/admin/*`
  (dev/ops tool, gated by `ADMIN_ENABLED` — `docs/09-operator-dashboard.md`)
- **PII:** Microsoft Presidio (analyzer + anonymizer)
- **LLM + embeddings:** local-first via Ollama's OpenAI-compatible API
  (`http://localhost:11434/v1`): `qwen3.5:4b` for completions,
  `nomic-embed-text` (768 dims) for embeddings. Same client works against any
  OpenAI-compatible provider by changing env vars.
- **LLM:** provider-agnostic thin client (direct API calls behind one interface)

## Status

**The request path is wired**: `POST /ask` runs retrieve → PII scrub → evidence
verification → synthesis under a wall-clock budget, and returns cited answers or
typed refusals (`insufficient_evidence`, `out_of_scope`, `guardrail_blocked`,
`error`). The operator dashboard is at `http://127.0.0.1:8000/ui`, backed by
`/admin/*`: corpus/chunk browser, Ask tab, trace timelines, retrieval + scrub
preview, seed / Jira / Gmail sync / embedding backfill, and cache invalidation
(`docs/09-operator-dashboard.md`).

Second live eval (golden set, lexical judge, local Ollama): **PII leaks 0** and
refusal correctness 0.90 pass their gates; groundedness 0.20 still fails, measured
with the labelled lexical proxy rather than an entailment judge (ruling #13 in
`docs/08-decisions.md`). The eval must run as a corpus ACL identity
(`--user-id …`); the default identity sees nothing.

Remaining: guardrails (M3), the planner's live fetch plus session/entity memory
(M4), and the Notion connector (M5). Gmail is implemented over IMAP with an app
password (ruling #12) and the answer cache is Redis behind an optional
`REDIS_URL` (ruling #14). See `docs/08-decisions.md` wave 3 for the integration
rulings and `src/orchestrator/HANDOFF.md` for the state machine's contract.

