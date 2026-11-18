# Agent 00 — Orchestrator

> Read with: `docs/02-architecture.md`, `docs/05-data-model.md`.

## Mission

Own the request lifecycle: receive a question, apply input guardrails, decide
(plan) how to answer, drive the pipeline states, and return the final answer.
You are a **deterministic state machine with two LLM steps** (planning,
memory-compaction). You never generate answer content yourself.

## State machine

```
INTAKE → GUARD_INPUT → MEMORY_CHECK → PLAN → RETRIEVE ⇄ FETCH_LIVE
       → SCRUB → VERIFY → SYNTHESIZE → GUARD_OUTPUT → RESPOND
                                  ↘ REFUSE (from VERIFY / GUARD_OUTPUT)
```

- Transitions are explicit code, not LLM whims. The LLM only fills in plan
  details (sources, entity filters, freshness needs) inside the `PLAN` step,
  and only from a fixed choice set with hard budgets (max 1 live fetch per
  request, max 2 LLM calls before forced synthesis).
- Every transition emits `audit.events` with `trace_id`.

## Contract

```python
class AskRequest(BaseModel):
    question: str
    session_id: str
    user_id: str

class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    refusal: Refusal | None   # set when refused, with reason
    trace_id: str
    degraded: bool = False    # budget exceeded -> answered on partial evidence
```

The orchestrator class exposes only `async def ask(req: AskRequest) -> AskResponse`.

## Failure modes

| Failure | Handling |
|---|---|
| Planner LLM timeout / bad JSON | 1 retry with stricter schema prompt → fall back to default plan (hybrid retrieval, all sources, no live fetch) |
| Any downstream agent raises typed error | map to user-visible message, never raw stack traces |
| Budget exceeded (latency/tokens) | skip remaining optional steps, force synthesis on evidence gathered so far, mark `degraded` in response metadata |

## Tests

- State machine unit tests with all agents mocked (order + refusal paths).
- Budget enforcement tests (fake slow agents trigger degradation).
