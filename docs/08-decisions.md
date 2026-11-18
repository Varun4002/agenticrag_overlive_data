# 08 — Lead decisions log

Every deviation a sub-agent requested, with the ruling. Rulings are binding:
agents must conform, or revisit with new evidence. New decisions go at the top
of their section (newest first).

## Wave 3 — Integration (orchestrator + dashboard, 2026-11-18)

Rulings made while wiring the request path end to end. Evidence for the
behavioural ones: the first live golden-set run
(`evals.run --judge lexical --user-id <corpus ACL user>`), which refused 6 of 8
answerable questions and scored groundedness 0.00 with pii_leaks 0.

| # | Decision | Rationale |
|---|---|---|
| 1 | **No LLM planner yet.** The plan is deterministic: hybrid retrieval, all sources, `orchestrator_top_k`, no live fetch; `orchestrator.plan` records `planner: "deterministic"`. | An LLM planner has no measurable job at this milestone: there are no sources to choose between (only Jira has data) and live fetch is M4. Skip it until a decision exists that determinism gets wrong. |
| 2 | **Budget exhaustion refuses with `degraded=true`, it does not "force synthesis on partial evidence"** (overriding the agent-00 brief's failure table). | Synthesis's contract is "VerifiedClaims only". Forcing synthesis on unverified evidence would break the grounding invariant the whole pipeline exists to enforce. A budget refusal is honest; a hallucination with citations is not. |
| 3 | **A PII scrub failure refuses with `guardrail_blocked`**; verify/synthesize are never called. | The scrubber is the only path from retrieval to an LLM. A scrub failure means the pack cannot safely leave the process. |
| 4 | **`known_person_names` is NOT sourced from `mem.entity_registry` yet.** The registry is empty; feeding a partial roster would narrow PERSON redaction to corroborated names and *leak* everyone missing. | Over-redaction is the safe failure direction until M4 builds a complete registry. The hook exists; the data does not. |
| 5 | **`/ask` without a database is 503, not a refusal.** | Extends agent 03's ruling: a refusal here would be indistinguishable from "no evidence found". The route emits `api.request.failed` and the trace shows `orchestrator.failed` at stage `retrieve`. |
| 6 | **A blank question refuses with `out_of_scope`** (intake guard). | Contract validation only enforces `min_length=1`; `" "` would otherwise reach retrieval. |
| 7 | **Sufficiency keywords weigh 0.25 and coverage accepts cited-chunk provenance** (`src/evidence/sufficiency.py`). Aspects like "Phoenix" are satisfied by the title/source_id/labels of a chunk a surviving claim actually cites; scaffolding words (give, list, show, tell, summarize, each) are stopworded. | The live eval refused "Which Project Phoenix tasks are overdue?" at coverage 0.22 although a claim citing PHX-142 fully answered it. The scorer conflated provenance with phrasing. Ticket keys (3.0) still dominate, so a genuinely-missing PHX-150 still refuses. |
| 8 | **Trailing citation-only fragments merge into the previous sentence; a marker-only answer refuses** (`src/synthesis/synthesizer.py::_clean_answer`). | A live request answered `"[1]"` — the model had written `"... because Nimbus slipped. [1]"`, so the real sentence looked uncited and was dropped. |
| 9 | **Multi-chunk claims cite the best-supporting chunk, not the first listed** (`_citations_for`). | A live answer stated 2026-12-04 and cited a chunk saying 2026-11-20. The lexical judge caught it; selection is now by claim-text word overlap with deterministic first-id tie-break. |
| 10 | **The eval harness must run as an ACL identity that the corpus grants** (`--user-id`). Default `eval-user` sees nothing (SQL ACL, fail closed) and the report flags `acl_identity_sees_nothing` instead of presenting 10/10 refusals as a system result. Per-question timeout raised 60 s → 180 s (local 8B extraction runs 30–60 s; the orchestrator budget is 120 s) and progress lines print per question. | The first run looked like a total pipeline failure; it was a harness identity + timeout problem plus the sufficiency defect. |
| 11 | **The operator dashboard ships enabled in dev/test only** (`ADMIN_ENABLED`, default off in staging/prod). Local-first, no auth — keep it bound to localhost or behind a proxy. | `docs/09-operator-dashboard.md`. |
| 12 | **Gmail connects over IMAP with an app password, not the Gmail REST API** (deviation from the M5 plan). Labels are IMAP mailboxes; scope = the allowlisted mailbox only, fail closed via the existing `is_label_allowed`. No deletion sweep — IMAP has no reliable deleted-signal; Gmail is treated as append-mostly. Cap: `gmail_max_messages` per label per run. | The user has an app password and no OAuth client; IMAP gives the same read-only, label-scoped access with zero console setup. Revisit only if labels/threads metadata beyond IMAP's reach becomes necessary. |
| 13 | **The groundedness gate is currently measured with the labelled lexical proxy.** 0.90 is the target for a real entailment judge; the lexical proxy (≥60% token overlap + verbatim numerals) under-scores correctly *reasoned* answers (e.g. "overdue" computed from dates, which appears in no chunk verbatim). | Run 2 scored 0.20 judged / 0.62 structural. Wiring the LLM judge through `src/llm` is the remaining step before the gate number means entailment; calibrate thresholds, never hand-tune answers. |
| 14 | **The answer cache is Redis (`REDIS_URL`, optional), keyed by `sha256(normalized question \| user_id \| data version)`.** Per-user by mandate: retrieval enforces `acl.users` in SQL, so a question-only key would serve one user's evidence to another. Hashed keys keep question text (which can contain names) out of Redis. Invalidation: a version counter bumped on any sync/backfill/seed write, plus a TTL backstop (900 s). Degraded and `reason="error"` outcomes are never cached; reads/writes fail open, version-read failure disables caching for that request, `bump_version`/`clear` fail loudly. Supersedes the `mem.answer_cache` Postgres table at the hot path (table stays, unused). | Token cost + M4 latency target ("repeat query < 1 s"). Redis over the existing table: a hit must not contend with retrieval on the same Postgres, TTL is native, and the cache must be optional infrastructure. New dependency `redis>=5.0` — the first added since M0. Cache hits emit `memory.cache_hit`, the event name `evals.trace_report.aggregate_stats` already counts. |

## Wave 2 (evidence / synthesis / PII) — 2026-11-17

Rulings from the cross-cutting decisions below apply; component hand-offs live in
`src/{evidence,pii,synthesis}/` (evidence and PII never delivered a HANDOFF —
their deviations are therefore unrecorded; flagged for completion).

## Wave 1 (connector / indexer / eval) — 2026-11-17

### Connector (agent 01)

| # | Deviation | Ruling |
|---|---|---|
| 1 | Connector config lives in `src/connectors/settings.py`, not `src/config.py` | **Approved as-is.** One settings module per bounded context is fine for a single-user MVP; `JIRA_*`/`NOTION_*`/`GMAIL_*` are documented in `.env.example`. Revisit if config surface multiplies. |
| 2 | Superseded rows get `source_id = "<native>#history-<id8>"` to satisfy `UNIQUE(source, source_id, is_current)` | **Rejected.** The constraint is wrong, not the model. Fix: migration `0003` replaces the constraint with a **partial unique index** `(source, source_id) WHERE is_current`, then remove the suffix hack. Only one *current* row per item; unlimited history. |
| 3 | Rate budget is per-process, not persisted in `core.sync_state` | **Approved for MVP.** Single worker process; revisit with the scheduler. |
| 4 | Per-run request budget not persisted | **Approved.** Same reasoning as #3. |
| 5 | Errors asymmetric: per-item/budget failures → partial result; auth failures → raise | **Approved.** Correct semantics: partial data is usable, missing credentials are not. |

### Indexer (agent 02)

| # | Deviation | Ruling |
|---|---|---|
| 1 | `estimate_tokens` (word/punct count) instead of a real tokenizer | **Approved.** Avoids pinning the chunker to one provider; determinism matters more than ±20% accuracy for chunk boundaries. |
| 2 | Overlap applied only when a structural unit overflows; no 300-token floor on short sections | **Approved.** Structure-first wins over a size target — the brief forbids merging across headings. |
| 3 | Breadcrumb prepended into chunk text (affects hash/token count) | **Approved.** Short comments and sections are otherwise unretrievable; the dedupe loss is acceptable. |
| 4 | Intra-run dedupe of identical chunk text | **Approved.** Free win. |
| 5 | Embeddings reuse `LLM_BASE_URL` / `LLM_API_KEY` | **Approved for local-first.** Add `EMBEDDING_BASE_URL`/`EMBEDDING_API_KEY` only when embeddings come from a different vendor than completions. |
| 6 | `chunks_skipped` includes metadata-only refreshes; deleted chunks logged, not returned | **Approved.** `embed_tokens` is the spend signal; do not change the frozen `IndexResult` for observability that `audit.events` already carries. |
| 7 | No cross-document chunk dedupe | **Approved.** Retrieval-side dedupe is agent 03's call. |
| 8 | Retry/backoff duplicated between `llm/` and `indexing/` | **Approved for now.** Extract a shared `http_retry.py` only if a third copy appears. |

### Eval (agent 09)

| # | Deviation | Ruling |
|---|---|---|
| 1 | `EvalResult.groundedness: float \| None` | **Approved.** The brief's own rule (no judge ⇒ null, never 1.0) requires it. |
| 2 | `golden.json` is an envelope with `schema_version` | **Approved.** Drift must fail loudly. |
| 3 | Category as a `[tag]` prefix in `notes` | **Approved.** Keeps `GoldenQuestion` exactly as specified. |
| 4 | `trace_report` is `async`; CLI `python -m evals.run` | **Approved.** Matches the async DB layer and the installed package layout. |

## Cross-cutting decisions

| Decision | Date | Rationale |
|---|---|---|
| **Structured-output steps use a non-reasoning model** (`LLM_JSON_MODEL`, default `dolphin-llama3:8b`) | 2026-11-17 | `qwen3.5:4b` spends its entire budget thinking: at `max_tokens=2048` it returned `finish_reason=length`, 6.7 kB of `reasoning`, and **empty `content`**. `/no_think` is ignored over Ollama's OpenAI-compatible `/v1`. A 4B reasoning model cannot do JSON extraction inside a usable latency budget. Verified: dolphin returns clean JSON in ~2 s. |
| **PII detection uses `en_core_web_md`, not `_sm`** | 2026-11-17 | The small model tags "Grace Adeyemi" as **ORG**, so PERSON policy silently leaked the name. Medium detects it correctly while still tagging "Nimbus Systems" as ORG (vendor names are not over-redacted). |

| Decision | Date | Rationale |
|---|---|---|
| Local-first models: Ollama `qwen3.5:4b` (completions) + `nomic-embed-text` 768-dim | 2026-11-17 | No paid API key available; identical OpenAI-compatible client. Migration `0002` moves `vector(1536) → vector(768)` and refuses to run on a non-empty `chunks` table. |
| Fail-closed audit (`AUDIT_STRICT`) | 2026-11-17 | An unauditable answer is refused with 503 (`docs/06-guardrails-pii.md`). |
| Lead owns internal pipeline contracts (`src/orchestrator/pipeline.py`) | 2026-11-17 | Prevents each agent inventing its own `EvidencePack`; PII scrubber is the only path from retrieval to any LLM. |

## Known integration gaps to close in Wave 2

1. **Retirement does not clean chunks.** Connector retires documents
   (`is_current=false`); the indexer deletes chunks for a non-current document
   **only when invoked for that id**. Nothing wires the two yet — the
   orchestrator (agent 00) must trigger index cleanup after a sync, or
   retrieval can return superseded text. Retrieval must also filter
   `documents.is_current` defensively.
2. **Connector emits no `audit.events`.** Sync decisions are invisible in
   traces today; add `connector.sync.completed`/`.failed` events in Wave 2.
3. **Eval metrics that depend on agent event names** (`memory.cache_hit`,
   `guardrail.injection_quarantined`, ...) stay `no-data` until those agents
   land. Rename with coordination, not unilaterally.
