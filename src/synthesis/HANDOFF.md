# Synthesis — Hand-off (component 06)

Status: **implemented**, tests pass on this machine (verified 2026-11-18). The
component shipped in Wave 2 without a hand-off; this document was written after
the fact from the code and tests, including the two bug fixes found on a live
run (rulings #8–#9).

Read with: `docs/agents/06-synthesizer-agent.md` (brief),
`docs/06-guardrails-pii.md` (Layer 3), `docs/08-decisions.md` (rulings #8, #9),
`src/orchestrator/pipeline.py` (`VerifiedClaims`, `SynthesisResult`).

---

## 1. What is implemented

| File | Contents |
|---|---|
| `src/synthesis/synthesizer.py` | `synthesize()` — one completion, deterministic cleanup, citation resolution, refusal paths. Helpers: `refusal_answer()`, `_clean_answer()`, `_merge_marker_only()`, `_is_decorative_line()`, `_dedupe_lines()`, `_line_key()`, `_renumber()`, `_best_supporting_chunk()`, `_citations_for()`. |
| `src/synthesis/prompts.py` | `SYSTEM_PROMPT`, `build_user_prompt()`, `build_messages()`, `MAX_OUTPUT_TOKENS` (1024). |
| `src/synthesis/__init__.py` | Docstring only; callers import `synthesis.synthesizer`. |
| `tests/synthesis/` | 19 tests in `test_synthesizer.py` + `synth_helpers.py`. |

### Contract (as implemented)

```python
async def synthesize(
    question: str,
    verified: VerifiedClaims,       # claims are the only permitted material
    pack: ScrubbedPack,             # citation metadata (title, url, quote text)
    *,
    trace_id: str | None = None,
    client: LLMClient | None = None,
    sink: EventSink | None = None,
    settings: Settings | None = None,
) -> SynthesisResult                # frozen: answer, citations, used_claim_ids, refused
```

The brief's `SynthesizeRequest` (with `session_context` and `refusal_reason`) is
not implemented; the refusal decision is derived from `verified` instead. See §4.

### Two paths

**Refusal — deterministic, no model involved.** Taken when
`not verified.can_answer or not verified.claims`. The text is
`refusal_answer(verified.insufficiency_reason)`:
`"I can't answer that from the evidence I could verify."` + the reason (when
present) + `"Try naming the project or ticket, or widening the time window."`
It emits `synthesis.refused` with the reason (≤500 chars) and returns
`citations=[]`, `used_claim_ids=[]`, `refused=True`.

**Answer — one completion, then deterministic surgery.**

1. Prompt: the numbered claims (`[1] … [n]`) and the question; the system prompt
   forbids outside knowledge, requires a marker on every factual sentence,
   requires placeholders to be repeated verbatim, requires conflicting claims to
   be presented with both citations, forbids headings/preamble, and forbids
   inventing a refusal.
2. `complete(model=LLM_JSON_MODEL or LLM_MODEL, temperature=0.0,
   max_tokens=1024)` — the JSON model is used for prose too, because a reasoning
   model spends the budget thinking.
3. `_clean_answer()` — per line: strip markers outside `1..len(claims)`; drop
   decorative lines (only markers and a `Citations:`/`References:` label);
   attach a citation-only fragment to the previous sentence (ruling #8); split
   sentences; keep marked sentences and record their markers; **drop** unmarked
   factual sentences (≥4 words, not a `:` heading); keep short/heading
   fragments; then drop duplicate lines by a marker/punctuation/case-insensitive
   key.
4. **Marker-only answers refuse.** If nothing but citation markers remains, the
   draft is discarded (ruling #8) — a live model answered `"[1]"`.
5. Renumber the surviving markers to dense `1..k` in order of first use, then
   build one `Citation` per used claim from `pack`:
   `chunk_id`, `document_id`, `source`, `title`, `quote=chunk.text[:200]`, `url`.
6. **Best-supporting chunk (ruling #9).** For a claim with several cited chunks,
   the citation is the chunk with the highest share of the claim's word set, not
   the first id listed; ties keep `chunk_ids` order, and zero overlap still
   yields a citation rather than dropping the claim.
7. If cleanup leaves no answer, no used marker, or no resolvable citation →
   refuse with `"The model's draft could not be tied back to the evidence."` and
   emit `synthesis.refused` with `dropped_sentences`.
8. Otherwise emit `synthesis.completed` (`answer_chars`, `citations`,
   `dropped_sentences`, `claims_available`) and return.

`used_claim_ids` is `[marker - 1 for marker in used]` — claim **positions**, not
`Claim.claim_id` (see §5).

---

## 2. How to run

```bash
cd /home/varun/projects/agentic_rag_over_live_data

.venv/bin/python -m pytest tests/synthesis -q      # fake client, no DB/network/model
.venv/bin/ruff check src/synthesis tests/synthesis
.venv/bin/mypy src/synthesis
```

The end-to-end path (live model + citations) is exercised through
`tests/orchestrator` (fake stages) and the eval harness, not here.

---

## 3. Tests

`tests/synthesis/test_synthesizer.py` (19): cited answer kept with citations
resolved and `used_claim_ids` · uncited factual sentence stripped · fully
uncited draft becomes a refusal + `synthesis.refused` · unanswerable evidence
refuses **without calling the model** and echoes the coverage reason · invalid
marker removed and the sentence dropped · markers renumbered to dense indices
only for claims actually used · placeholders passed through verbatim · trailing
`[1]` attached to the previous sentence · `"[1]"`-only answer refuses ·
multi-chunk claim cites the chunk that states the value (old id listed first on
purpose) · citation tie-break is first listed · single-chunk citation unchanged ·
`_best_supporting_chunk` returns `None` when no candidate is in the pack · a
`Citations: [1] [2] [3]` dump is not the answer · a duplicated reference line is
dropped · `_is_decorative_line` cases · `_line_key` ignores markers, punctuation
and case · `_dedupe_lines` keeps the first occurrence · zero-overlap claim still
cites its first candidate.

These are the two live bugs as regression tests: `"... because Nimbus slipped.
[1]"` and the wrong-chunk citation for the moved cutover date.

---

## 4. Deviations from the brief

1. **No `SynthesizeRequest`; the refusal decision moved in.** The brief gives the
   synthesizer `refusal_reason` and a `session_context`; here it reads
   `verified.can_answer` / `insufficiency_reason` and has no session context.
   The brief's `SynthesisResult` fields are the frozen ones, unchanged.
2. **`session_context` is absent** because Tier 1 memory is unbuilt (see
   `src/memory/HANDOFF.md`): nothing produces session turns to pass.
3. **The `Citation` shape is the frozen one**, `{chunk_id, document_id, source,
   title, quote, url}` — not the brief's `{source, source_id, title, url,
   updated_at}`. `quote` (first 200 characters of the chunk) is what
   `evals.scoring` feeds the groundedness judge; `source_id`/`updated_at` are not
   in the contract, so the API response cannot expose them.
4. **Marker-merge and marker-only refusal** (ruling #8). A model that writes
   `"... because Nimbus slipped. [1]"` otherwise loses the real sentence and can
   ship an answer that is only `"[1]"`.
5. **Best-supporting-chunk citations** (ruling #9). The first listed chunk id is
   not a priority order: a live answer stated 2026-12-04 while citing the chunk
   saying 2026-11-20.
6. **Decorative-line and duplicate-line cleanup** — same wave, same failure
   shape (a `Citations: [1] [2] [3]` dump as the entire answer; the model
   restating its own sentence after the marker). No separate ruling number; they
   are part of ruling #8's "clean the draft deterministically, then refuse if
   nothing is left".
7. **Structure is deliberately not question-type-aware.** The brief asks for
   task lists / deadline tables / status reports; the prompt forbids headings and
   asks for plain prose or a short list. Rationale: every extra structural
   instruction is another thing the local model can get wrong, and the
   deterministic cleaner discards structure it cannot verify.
8. **Refusal text does not say what was searched.** The brief asks for "state
   what was searched, what was found, and one concrete suggestion". The
   implementation states the sufficiency reason (which lists missing aspects) and
   one fixed suggestion; it does not enumerate sources or counts.

---

## 5. Uncertainty / risks

- **Model-bound quality is the main risk.** On a live run the synthesizer
  answered a contradiction question with only the latest value, although the
  prompt says to present both — the deterministic layer cannot detect a silently
  dropped claim. The same applies to a claim it paraphrases into a new number:
  numbers are not cross-checked against the claims.
- **Short unmarked sentences survive.** `_is_factual()` requires ≥4 words, so an
  unverified 1–3 word factual fragment ("Cutover moved.") is kept as a
  heading-like fragment. This is deliberate (headings must survive) and is a
  small hole in "no uncited factual sentence ships".
- **`used_claim_ids` and citations are positional.** `claims[marker - 1]` assumes
  the claim list is indexed by position; the evidence validator renumbers
  `claim_id` 0..k-1 so the invariant holds today, but a caller that supplies
  arbitrary `claim_id` values (the test helpers do) would get wrong citations.
  Keying on `claim.claim_id` would be stricter; the frozen shape allows either.
- **Citations are not deduplicated.** Two used claims that cite the same chunk
  produce two `Citation` entries with the same `chunk_id`, and two markers that
  resolve to the same quote. Not wrong, but an operator may read it as two
  sources.
- **`quote` is the head of the chunk, not the supporting span.** If the claim's
  evidence sits deeper than 200 characters, the delivered quote does not contain
  it. The eval's structural groundedness check operates on citations, so this
  can under-score otherwise-correct answers.
- **`finish_reason` is never inspected.** A truncated completion (1024-token
  cap) is cleaned and returned like any other; truncation shows up as a missing
  final sentence, not an error.
- **No retry inside synthesis.** A draft with no surviving citations refuses
  after one call; re-prompting (brief's spirit) is left to a future iteration.
- **Decorative-line detection is lexical.** A legitimate line consisting only of
  a label plus markers is dropped; conversely a one-word "answer" line survives
  as a fragment.
- **Not verified:** any model other than the local Ollama pair, long claim lists
  (renumbering and marker-stripping across many claims), non-English content, and
  concurrency.

---

## 6. Verified gates (real output, 2026-11-18)

```
$ .venv/bin/python -m pytest tests/synthesis -q
19 passed, 1 warning in 0.04s

$ .venv/bin/ruff check src/synthesis tests/synthesis
All checks passed!

$ .venv/bin/mypy src/synthesis
Success: no issues found in 3 source files
```

The `StarletteDeprecationWarning` from `fastapi.testclient` is the one warning
everywhere in this repo (pre-existing): the repo-root `tests/conftest.py`
imports `TestClient` for every run, so it shows up even in suites that never use
it.

Repo-wide context at the same moment: `pytest -q` → 490 passed, 64 skipped
(`REDIS_URL` unset; the suite was still moving while this was written — a
concurrent agent was adding answer-cache tests). The live quality numbers are in
`src/evals/HANDOFF.md` and `docs/08-decisions.md` (ruling #13): eval run 2 scored
groundedness 0.20 with `pii_leaks=0` and `refusal_correct=0.90`.

---

## 7. Files

Read for this hand-off (no code changed):

```
src/synthesis/__init__.py      docstring only
src/synthesis/synthesizer.py   synthesize(), cleanup, citation resolution, refusals
src/synthesis/prompts.py       system prompt + numbered-claim user prompt
tests/synthesis/synth_helpers.py   chunk/claim/pack/verified builders, FakeClient
tests/synthesis/test_synthesizer.py
```
