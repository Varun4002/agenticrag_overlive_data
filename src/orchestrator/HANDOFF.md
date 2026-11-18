# HANDOFF — Orchestrator (component 00, integration wave)

Status: **implemented and verified** (2026-11-18). `/ask` now runs
retrieve → scrub → verify → synthesize and returns cited answers; the M0
placeholder is gone. Gates: `pytest` 421+ passed / 59 skipped (DB tests skip
without `DATABASE_URL`), `ruff` clean, `mypy src` clean.

Read with: `docs/agents/00-orchestrator.md`, `docs/04-orchestration.md`,
`docs/08-decisions.md` (wave-3 rulings are binding context for everything below).

---

## 1. What is implemented

| File | Contents |
|---|---|
| `src/orchestrator/run.py` | `Orchestrator.ask(req) -> AskResponse`: the state machine, stage injection, refusals, budget, audit events. `Stages` (the four callables), `Plan`, `default_stages()`. |
| `src/orchestrator/pipeline.py` | Lead-owned frozen hand-off shapes (`EvidencePack` → `ScrubbedPack` → `VerifiedClaims` → `SynthesisResult`) — unchanged apart from the additive `SynthesisResult.refusal_reason` (ruling #20). |
| `src/orchestrator/contracts.py` | Frozen HTTP contract — untouched. |
| `src/api/routes.py` | `/ask` calls the orchestrator; 503 on misconfiguration; `api.request.failed` on failure; audit-strict semantics unchanged. |
| `src/config.py` | `orchestrator_top_k` (12), `orchestrator_budget_s` (120). |
| `tests/orchestrator/` | `orch_fakes.py` + 14 state-machine tests: stage order, refusal taxonomy, budget, tracing, contract round-trip. |
| `tests/api/` | Boundary tests updated: no-database 503 (not a refusal), full failure trace, unique trace ids. |

### Flow and failure mapping

```
INTAKE → PLAN → RETRIEVE → SCRUB → VERIFY → SYNTHESIZE → RESPOND
                            ↘ refusal: guardrail_blocked (scrub failure)
                     ↘ 503: RetrievalConfigError (no database)
                  ↘ refusal: error (LLM failure, budget exhaustion)
               ↘ refusal: insufficient_evidence (sufficiency or draft cleanup)
```

- **Refusal taxonomy**: `insufficient_evidence` (evidence says no),
  `guardrail_blocked` (scrub failed), `out_of_scope` (blank question),
  `error` (LLM/budget). The user-visible detail never contains provider
  messages, only the error *type* — no stack traces leak. Synthesis refusal
  reasons are propagated into `refusal.detail` (`SynthesisResult.refusal_reason`
  first, then `VerifiedClaims.insufficiency_reason`, then the generic string —
  ruling #20), so a truncated draft reports `"answer truncated by the token
  limit"` rather than a misleading fallback.
- **Budget** (`orchestrator_budget_s`, wall clock): checked before each LLM
  step; exhaustion or stage timeout refuses with `degraded=true`. It never
  forces synthesis on unverified claims (ruling #2 in docs/08).
- **Skipped stages are explicit in every trace**: `orchestrator.plan` records
  `skipped_stages: ["guard_input", "memory_check", "guard_output"]` — the guard
  (M3) and memory (M4) agents are not written, and the trace says so.
- **Trace events**: `orchestrator.intake/plan/completed/refused/failed`, plus
  every downstream agent's own events under the same `trace_id`. A full trace
  for one answered question: received → intake → plan → retrieval.ran →
  pii.scrubbed → evidence.extracted → evidence.sufficient → synthesis.completed
  → orchestrator.completed → api.request.completed.

### Stage injection

`Orchestrator(settings=…, sink=…, stages=Stages(...), session_factory=…)`.
`default_stages()` binds the real components **lazily** — `pii` pulls
Presidio + spaCy and must not slow app startup or break `/health`. Tests inject
`tests/orchestrator/orch_fakes.py::FakeStages`; no test needs a database,
network, or model. (The fakes module is named `orch_fakes`, not `fakes`, because
`tests/retrieval/fakes.py` already owns the bare name.)

## 2. How to run

```bash
cd /home/varun/projects/agentic_rag_over_live_data
.venv/bin/python -m pytest tests/orchestrator tests/api -q   # no DB/network
.venv/bin/ruff check && .venv/bin/mypy src

set -a; source ~/.config/agentic-rag/dev.env; set +a
DATABASE_URL=… .venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question":"When is the PHX-150 cutover?","session_id":"s","user_id":"priya.raghunathan@northwind.example"}'
```

The golden-set eval must run as an identity the corpus grants:

```bash
.venv/bin/python -m evals.run --suite golden --judge lexical \
  --user-id priya.raghunathan@northwind.example
```

## 3. Deviations from the agent-00 brief (Lead rulings in docs/08, wave 3)

1. **No LLM planner.** Deterministic plan until a decision exists that
   determinism gets wrong. The brief's "LLM fills in plan details" is deferred,
   not dropped: `Plan` is the seam.
2. **Budget exhaustion refuses** instead of "force synthesis on gathered
   evidence" — that would violate the VerifiedClaims-only contract.
3. **Blank question → `out_of_scope`**; the brief has no rule for it.
4. **`GUARD_INPUT`/`MEMORY_CHECK`/`GUARD_OUTPUT` skipped and declared** in the
   trace rather than silently absent.
5. The route maps `RetrievalConfigError` to 503 — the brief's failure table
   covers agent errors, not deployment misconfiguration.

## 4. Known gaps (honest)

- **The guardrail agent (M3) is not written.** The evidence validator drops
  injection-shaped claims and prompts wrap content in `<untrusted_content>`,
  but there is no input or output guard. The planted-injection corpus question
  is answered through claim filtering alone.
- **Memory is Tier-2 only now**: the Redis answer cache is live
  (`src/memory/answer_cache.py`, optional via `REDIS_URL`, per-user keys,
  version invalidation — ruling #14), but the entity registry and session
  turns are still unwritten. The cache version is bumped by
  `connectors.service.sync`, `POST /admin/backfill`, `POST /admin/seed` and
  `POST /admin/cache/invalidate`.
- **No live fetch**: `Plan.live_fetch` is always false; fetched documents would
  need same-request indexing to be citable.
- **Synthesis quality is model-bound**: a live run answered a contradiction
  question with the latest value only, and one extraction returned non-JSON
  (fail-closed refusal, correct behavior, retry-able). Decorative
  `Citations:`-dump lines and duplicated reference lines are now dropped
  deterministically.
- **Not verified**: behaviour under concurrent load; anything but the local
  Ollama models; the route's 500 path has no test (it is a `logger.exception`
  plus generic detail).

## 5. Files created / changed (this wave)

```
src/orchestrator/run.py            new — state machine
src/orchestrator/__init__.py       exports
src/api/routes.py                  /ask wired; placeholder removed
src/config.py                      orchestrator_top_k, orchestrator_budget_s
tests/orchestrator/orch_fakes.py    stage fakes
tests/orchestrator/test_state_machine.py  14 tests
tests/api/test_endpoints.py        rewritten for the wired boundary
tests/api/test_audit_policy.py     one test's intent preserved under new semantics
src/synthesis/synthesizer.py       marker-merge + marker-only refusal (bug found live)
tests/synthesis/test_synthesizer.py 2 regression tests
src/evidence/sufficiency.py        keyword weight 0.25 + provenance context (via edit agent)
src/evidence/service.py            provenance map
src/evals/run.py                   timeout 180 s, progress lines, ACL flag
src/evals/schemas.py               EvalResult.user_id
docs/09-operator-dashboard.md      Ask tab now answers
.opencode/opencode.jsonc           project model policy (GLM-5.3 orchestrates, DeepSeek edits)
```
