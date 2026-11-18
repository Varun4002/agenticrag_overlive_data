# 04 — Build Orchestration

How the work is decomposed into sub-agents, the order they build in, and the
contracts that let them work in parallel without stepping on each other.

## Roles

- **Lead (this agent):** owns `02-architecture.md` + `05-data-model.md`
  interfaces, reviews every hand-off, resolves cross-agent conflicts, runs the
  final integration.
- **Sub-agents:** each owns one component, builds from its brief in
  `docs/agents/`, and delivers code + tests + a hand-off note. Sub-agents do
  **not** edit shared docs or other agents' components.

## Interface freeze rule

Before M0 completes, the following are frozen (changes require Lead sign-off
and are announced to every active agent):

1. Canonical document/chunk schemas (`05-data-model.md`).
2. The typed contract between agents: each agent exposes one class with a
   Pydantic `*Request`/`*Response` pair defined in its brief.
3. The audit-event schema and `trace_id` propagation convention.

Everything *behind* each contract is free to change.

## Build DAG

```
M0: skeleton ── schema, config, LLM client, tracing   [Lead]
        │
        ├───────────────┬──────────────────┐
        ▼               ▼                  ▼
   01-connector    02-indexer        09-eval-agent (golden set first)
   (jira first)    (chunk/embed)          │
        └───────┬───────┘                  │
                ▼                          │
        03-retrieval ◀── depends on chunks │
                │                          │
        ┌───────┴────────┐                 │
        ▼                ▼                 ▼
   04-pii-scrubber   05-evidence      integration test of slice
        │                │
        └───────┬────────┘
                ▼
          06-synthesizer
                │
   ┌────────────┼───────────────┐
   ▼            ▼               ▼
07-guardrail  08-memory    (notion/gmail waves of 01)
   │
   └────▶ final eval pass by 09
```

Parallelization windows:
- **Wave 1 (M1):** 01-connector, 02-indexer, 09-eval run in parallel —
  02 depends only on the frozen chunk schema, not on live Jira data (use a
  fixture corpus).
- **Wave 2 (M2–M3):** 04-pii, 05-evidence, 06-synthesizer build in parallel
  against the frozen `EvidencePack` / `VerifiedClaims` contracts.
- **Wave 3 (M4–M5):** 07-guardrail, 08-memory, and the Notion/Gmail connector
  waves.

## Hand-off contract (every sub-agent delivers)

1. Code under `src/<component>/`, matching the layout in `02-architecture.md`.
2. Tests under `tests/<component>/` — unit tests for pure logic, contract
   tests asserting the Pydantic request/response models round-trip.
3. `HANDOFF.md` in the component dir: what's implemented, what's stubbed, how
   to run its tests, any deviations from the brief (Lead must approve).
4. All config read via `pydantic-settings` from env — **no hard-coded
   credentials, no secrets in code or tests** (connectors use env-stored
   OAuth tokens / API keys only).

## Acceptance gate per milestone

The Lead runs:
1. `pytest` (full suite) — must be green.
2. `ruff check` + `mypy` — clean.
3. Slice test: run the golden-question subset for the current milestone
   against a fixture corpus.
4. Trace review: audit events for a sample query must show every agent
   decision recorded.

Only then does the next wave spawn.

## Working conventions

- Python 3.14, asyncio, type annotations everywhere (`mypy` strict for
  `src/`), Pydantic v2 models for all cross-boundary data.
- One dependency per real need; adding a library requires justification in
  the hand-off note (current approved list in `02-architecture.md`).
- Heavy work (bulk embedding, full eval runs) goes to the shared server
  (`ssh shared-server`), not the laptop.
- Agents mark uncertainty explicitly in `HANDOFF.md` — never silently stub
  behavior and claim it works.
