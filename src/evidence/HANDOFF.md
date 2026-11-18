# Evidence Agent — Hand-off (component 05)

Status: **implemented**, tests pass on this machine (verified 2026-11-18), plus
one opt-in live extraction against the local model. Like the PII scrubber, this
component shipped in Wave 2 without a hand-off; this document was written after
the fact from the code and tests.

Read with: `docs/agents/05-evidence-agent.md` (brief),
`docs/06-guardrails-pii.md` (Layer 3), `docs/08-decisions.md` (ruling #7,
cross-cutting "structured-output steps use a non-reasoning model"),
`src/orchestrator/pipeline.py` (the frozen shapes).

---

## 1. What is implemented

| File | Contents |
|---|---|
| `src/evidence/service.py` | `verify(question, pack, *, …) -> VerifiedClaims`: one extraction call, at most one repair, deterministic validation, sufficiency scoring, audit events. |
| `src/evidence/parsing.py` | `ExtractedClaim`, `ExtractionResult`, `parse_extraction()`, `extract_json_object()`, `EvidenceParseError` — tolerant of model noise, strict about schema. |
| `src/evidence/validator.py` | `validate_claims()`, `ValidationOutcome`, `Contradiction`, `is_injection()`, `lexical_support()`, `tokenise()`, `DEFAULT_MIN_OVERLAP`. |
| `src/evidence/sufficiency.py` | `question_aspects()`, `score_sufficiency()`, `Aspect`, `SufficiencyResult`, `DEFAULT_THRESHOLD`. |
| `src/evidence/prompts.py` | `SYSTEM_PROMPT`, `build_system_prompt()`, `build_user_prompt()`, `build_messages()`, `MAX_OUTPUT_TOKENS`, `MAX_CHUNK_CHARS`. |
| `tests/evidence/` | 36 test items: 35 offline (all pass) + 1 opt-in live call (skipped without `LIVE_LLM`). |

### Contract (as implemented)

```python
async def verify(
    question: str,
    pack: ScrubbedPack,                     # the only data the model may see
    *,
    trace_id: str | None = None,
    client: LLMClient | None = None,        # injected in tests; built + closed when omitted
    sink: EventSink | None = None,          # defaults to a no-op sink
    settings: Settings | None = None,       # thresholds, model, endpoint
) -> VerifiedClaims
```

The brief's `VerifyRequest` wrapper is not implemented; see §4.1. The returned
`VerifiedClaims` is the frozen Lead shape: `question`, `claims`,
`dropped_claims`, `sufficiency`, `can_answer`, `insufficiency_reason`.

### The one call, and the one repair

```
verify()
 ├─ empty pack ─────────────────────────────► 0 claims, can_answer=False, no LLM call
 ├─ build_messages()  → <untrusted_content chunk_id="…"> blocks
 ├─ complete(model=LLM_JSON_MODEL, temperature=0.0, max_tokens=2048)
 ├─ parse_extraction() ── fails ───────────► one repair prompt (the error text is
 │                                           appended), one more call, parse again
 ├─ validate_claims()      citations → injection → lexical support → dedupe →
 │                         contradiction detection → renumbering
 ├─ score_sufficiency()    aspects × claim text + cited-chunk provenance
 └─ events
```

- **Prompt hardening:** every chunk is wrapped in `<untrusted_content
  chunk_id="…">`; any `</untrusted_content>` inside the text is rewritten to
  `[/untrusted_content]` so a document cannot end the data block early. Chunks
  longer than `MAX_CHUNK_CHARS` (2000) are truncated with `…[truncated]`.
- **Parsing tolerance:** `EvidenceParseError` only when no balanced JSON
  payload exists, the payload lacks `claims`, or a claim fails schema validation.
  `<think…>…</think>` blocks, code fences and surrounding prose are tolerated;
  a bare claim array is wrapped; `chunk_ids` accepts `"a, b"`; confidence accepts
  `"80%"` and is clamped instead of failing the claim.
- **Validation order matters:** a claim with no existing chunk id is dropped as
  dangling *before* the injection filter; a claim whose text is instruction-
  shaped is recorded separately in `injection_dropped`; a surviving claim needs
  ≥ `EVIDENCE_MIN_CLAIM_OVERLAP` (0.4) of its content tokens in the cited
  chunks; duplicates merge (union of chunk ids, higher confidence kept);
  contradictions are detected, **never resolved**; `claim_id` is renumbered
  0..k-1 in kept order.
- **Sufficiency is deterministic — no LLM.** Aspects: ticket keys (3.0),
  `[PLACEHOLDER_n]` (2.0), quoted phrases (2.0), capitalised names (1.5, with a
  question-word ignore list), content keywords (0.25, at most 8). An aspect is
  covered when it appears in a kept claim's text **or** in the provenance text
  (title, `source_id`, `meta.labels`, `meta.section`) of a chunk that claim
  cites. Uncited chunks never count. `can_answer = score >= threshold`
  (`EVIDENCE_SUFFICIENCY_THRESHOLD`, default 0.5). No claims or no aspects
  scores 0.0 (fails closed).

### Events (`audit.events`, agent `evidence`)

| Event | When | Payload |
|---|---|---|
| `evidence.extracted` | always, except a hard parse failure | `claims`, `dropped_dangling`, `dropped_unsupported`, `dropped_injection`, `duplicates_merged`, `contradictions` (subject/attribute/values), `chunks`, `calls`, `sufficiency`, `can_answer`, `usage`; `reason="empty_pack"` short-circuit variant |
| `evidence.repair_attempted` | first parse failed | `error` (≤500 chars), `question` |
| `evidence.parse_failed` | both parses failed | `error`, `calls`, `usage` |
| `evidence.injection_dropped` | any instruction-shaped claim survived parsing | `count`, up to 3 samples truncated to 120 chars |
| `evidence.sufficient` / `evidence.insufficient` | after scoring | `sufficiency`, `threshold`, `covered`, `missing` |

### Failure modes

| Failure | Behaviour |
|---|---|
| Empty pack | No LLM call; `insufficiency_reason="retrieval returned no evidence chunks"` |
| Invalid JSON twice | `evidence.parse_failed`, zero claims, `can_answer=False` |
| Model cites only non-existent chunk ids | all claims dropped as dangling, zero claims, refusal |
| Model returns valid JSON with `{"claims": []}` | **accepted as success** — no repair, zero claims, refusal |
| Instruction-shaped claim | dropped, audited; other claims unaffected |
| Contradictory values | both kept, contradiction reported in the trace; the synthesizer must present both |
| LLM error | propagates as `LLMError`; the orchestrator applies the wall-clock budget and maps both to a typed refusal |

---

## 2. How to run

```bash
cd /home/varun/projects/agentic_rag_over_live_data

# offline: fakes only, no DB, no network, no model
.venv/bin/python -m pytest tests/evidence -q

# the single opt-in live extraction (Ollama must be running with the deployed model)
set -a; source ~/.config/agentic-rag/dev.env; set +a
LIVE_LLM=1 .venv/bin/python -m pytest tests/evidence/test_live_ollama.py -q

.venv/bin/ruff check src/evidence tests/evidence
.venv/bin/mypy src/evidence
```

---

## 3. Tests

`tests/evidence/test_parsing.py` (10): fenced JSON · prose around the payload ·
decoy JSON inside a `think` block · braces inside strings · bare claim array ·
`"c1, c2"` and `"80%"` coercion · missing `claims` key · non-JSON · claim
without `text` · `extract_json_object` array wrapping.

`tests/evidence/test_validator.py` (7): dangling ids dropped · lexically
unsupported claim dropped · injection-shaped claim dropped · duplicates merge
with the union of citations · deterministic renumbering · contradictory values
kept and reported (no side picked) · `min_overlap=0.0` keeps weak claims.

`tests/evidence/test_sufficiency.py` (12): full coverage answers · partial
coverage refuses and names the missing aspect · no claims · no aspects ·
inclusive threshold · ticket keys outweigh keywords · keyword weight is 0.25 ·
project name satisfied by cited-chunk provenance · scaffolding words are not
aspects · generic keywords alone cannot sink a question · uncited chunks do not
count · a missing ticket key still refuses.

`tests/evidence/test_evidence_service.py` (6): valid extraction verified with
events emitted and the injected client left open · evidence wrapped as
untrusted, `temperature=0.0`, `max_tokens ≥ 1024` · malformed output triggers
exactly one repair and succeeds · double parse failure fails closed and emits
`evidence.parse_failed` · empty pack makes no LLM call · injection claim dropped
and audited.

`tests/evidence/test_live_ollama.py` (1, opt-in): a real document from
`fixtures/corpus` through the real model — claims are produced and every cited
`chunk_id` is one of the provided ids.

---

## 4. Deviations from the brief

1. **No `VerifyRequest` model; `verify(question, pack, **kwargs)` instead.** The
   brief's two request fields are the first two positional arguments; everything
   else is injection for tests/evals. No `VerifiedClaims` field was added or
   renamed (frozen Lead contract).
2. **Structured-output calls use `LLM_JSON_MODEL`** (`llm_json_model or
   llm_model`) — cross-cutting decision in `docs/08-decisions.md`: `qwen3.5:4b`
   spends its whole budget thinking and returns empty `content`; a non-reasoning
   model is required for JSON extraction.
3. **Keyword aspects weigh 0.25, coverage accepts cited-chunk provenance, and
   scaffolding words are stopworded** — ruling #7 in `docs/08-decisions.md`.
   Without it, a live request refused "Which Project Phoenix tasks are overdue?"
   at coverage 0.22 although the claim citing PHX-142 answered it. Ticket keys
   still dominate, so a genuinely missing key still refuses.
4. **No model self-report in sufficiency.** The brief says "heuristic + model
   self-report"; the score is purely deterministic and `confidence` from the
   model is carried on each `Claim` but never used in the decision.
5. **Claim texts are not written to the audit payload.** The brief says "all
   outputs audited with claim texts and chunk references"; the trace carries
   counts, dropped counts and contradiction values. No `audit.events` row holds a
   claim's text — claim texts exist only in the synthesizer's prompt. Rationale:
   payload size, and posting model output into the log is not obviously
   desirable.
6. **Citation mapping is "model cites + lexical overlap", not entailment
   search.** The brief's step 2 ("find the chunk ids that entail it") is
   implemented as: trust the ids the model cited, verify they exist, require
   0.4 token overlap with those chunks. A claim citing the wrong-but-existing
   chunk with enough shared vocabulary survives.
7. **Contradiction detection is an addition.** The brief only says "keep both";
   `_detect_contradictions` additionally groups by `subject` + `attribute` and
   reports conflicting values in the trace, so an operator can see *why* an
   answer presented two dates.

---

## 5. Uncertainty / risks

- **A valid-JSON zero-claim answer is not retried.** `{"claims": []}` parses, so
  no repair is attempted, and the result is a refusal. The same is true when
  every claim is dropped for a dangling id — the repair prompt only fires on
  *parse* errors, never on "the model ignored the provided ids".
- **Lexical support is a proxy for entailment.** A paraphrased claim ("slipped
  by two weeks" vs "delay of 10 business days") can fall below 0.4 and be
  dropped; a claim that reuses the chunk's vocabulary while inverting its
  meaning can pass. `EVIDENCE_MIN_CLAIM_OVERLAP` is the tuning knob; it has not
  been calibrated beyond the fixture corpus.
- **Injection filtering is regex-shaped.** A legitimate claim that quotes a
  document's instruction text ("do not follow the old runbook") is dropped whole,
  because `is_injection()` matches the claim text, not its provenance. Also,
  only the closing `</untrusted_content>` tag is neutralised — a document that
  contains an opening tag with a fake `chunk_id` is inside the wrapper as data,
  but the model can still be confused by it.
- **Contradiction detection compares raw lowercased strings.** `"2026-12-04"`
  and `"December 4, 2026"` are reported as two different values, so a single
  fact written two ways looks like a contradiction; conversely, if the model
  omits `subject`/`attribute`/`value` (the prompt allows it), no contradiction is
  detected at all and only the answer text can show the conflict.
- **One extraction call means the model's recall is the system's recall.** A
  second pass over the same chunks, or per-chunk extraction, would cost latency
  and tokens; the current design trades recall for a fixed two-call ceiling.
- **Prompt size is unbounded by the code.** `build_user_prompt()` includes every
  chunk up to 2000 chars; with `orchestrator_top_k=12` that is ~24k characters
  before the system prompt. Fine for the local model, unmeasured against a
  provider context limit.
- **`usage` is only reported when the client exposes a `ledger`.** The real
  `OpenAICompatibleClient` does; test fakes do not, so `usage` is `{}` in the
  unit tests and the orchestration is never cost-asserted.
- **Model-bound quality.** The live check is a smoke test that JSON comes back
  and cites real ids — not that the extraction was correct. A wrong-but-parsable
  claim is caught only if it lacks lexical support or contradicts another claim.
- **Not verified:** any provider other than the local Ollama models, concurrent
  requests against one model, very large packs, and the repair path against a
  model that fails twice for different reasons.

---

## 6. Verified gates (real output, 2026-11-18)

```
$ .venv/bin/python -m pytest tests/evidence -q
35 passed, 1 skipped, 1 warning in 0.05s
    # the skip is tests/evidence/test_live_ollama.py:
    #   "live model check: set LIVE_LLM=1 and have Ollama running"

$ set -a; source ~/.config/agentic-rag/dev.env; set +a
$ LIVE_LLM=1 .venv/bin/python -m pytest tests/evidence/test_live_ollama.py -q
1 passed, 1 warning in 10.38s
    # real extraction over PHX-150 from fixtures/corpus, local Ollama

$ .venv/bin/ruff check src/evidence tests/evidence
All checks passed!

$ .venv/bin/mypy src/evidence
Success: no issues found in 6 source files
```

The `StarletteDeprecationWarning` from `fastapi.testclient` is the one warning
everywhere in this repo (pre-existing).

---

## 7. Files

Read for this hand-off (no code changed):

```
src/evidence/__init__.py       docstring only
src/evidence/service.py        verify(), the call/repair/score wiring, events
src/evidence/parsing.py        model-output JSON extraction + schema coercion
src/evidence/validator.py      citation/injection/support/dedupe/contradiction pass
src/evidence/sufficiency.py    aspect extraction + coverage scoring
src/evidence/prompts.py        system/user prompts, untrusted-content wrappers
tests/evidence/conftest.py     fixture-corpus fixture
tests/evidence/helpers.py      ScrubbedPack builders + FakeClient
tests/evidence/test_parsing.py
tests/evidence/test_validator.py
tests/evidence/test_sufficiency.py
tests/evidence/test_evidence_service.py
tests/evidence/test_live_ollama.py
```
