# Synthesis — Hand-off (component 06)

Status: **implemented**, tests pass on this machine (verified 2026-11-18). The
component shipped in Wave 2 without a hand-off; this document was written after
the fact from the code and tests, including the two bug fixes found on a live
run (rulings #8–#9) and deterministic citation for marker-less drafts (ruling
#21).

Read with: `docs/agents/06-synthesizer-agent.md` (brief),
`docs/06-guardrails-pii.md` (Layer 3), `docs/08-decisions.md` (rulings #8, #9,
#21), `src/orchestrator/pipeline.py` (`VerifiedClaims`, `SynthesisResult`).

---

## 1. What is implemented

| File | Contents |
|---|---|
| `src/synthesis/synthesizer.py` | `synthesize()` — one completion, deterministic cleanup, citation resolution, refusal paths. Helpers: `refusal_answer()`, `_clean_answer()`, `_implied_marker()`, `_merge_marker_only()`, `_is_decorative_line()`, `_dedupe_lines()`, `_line_key()`, `_renumber()`, `_best_supporting_chunk()`, `_citations_for()`. |
| `src/synthesis/prompts.py` | `SYSTEM_PROMPT`, `build_user_prompt()`, `build_messages()`, `MAX_OUTPUT_TOKENS` (2048). |
| `src/synthesis/__init__.py` | Docstring only; callers import `synthesis.synthesizer`. |
| `tests/synthesis/` | 37 tests in `test_synthesizer.py` + `synth_helpers.py`. |

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
) -> SynthesisResult                # frozen: answer, citations, used_claim_ids, refused,
                                    #         refusal_reason (additive, ruling #20)
```

The brief's `SynthesizeRequest` (with `session_context` and `refusal_reason`) is
not implemented; the refusal decision is derived from `verified` instead. See §4.
The *output* `SynthesisResult.refusal_reason` (additive, default `None`) carries
the deterministic reason whenever `refused` is true, so the orchestrator can put
it in `refusal.detail` instead of a generic fallback.

### Two paths

**Refusal — deterministic, no model involved.** Taken when
`not verified.can_answer or not verified.claims`. The text is
`refusal_answer(verified.insufficiency_reason)`:
`"I can't answer that from the evidence I could verify."` + the reason (when
present) + `"Try naming the project or ticket, or widening the time window."`
It emits `synthesis.refused` with the reason (≤500 chars) and returns
`citations=[]`, `used_claim_ids=[]`, `refused=True`,
`refusal_reason="insufficient evidence: " + reason`.

**Answer — one completion, then deterministic surgery.**

1. Prompt: the numbered claims (`[1] … [n]`) and the question; the system prompt
   forbids outside knowledge, requires a marker on every factual sentence,
   requires placeholders to be repeated verbatim, requires conflicting claims to
   be presented with both citations, forbids headings/preamble, forbids
   inventing a refusal, and bounds the shape: **at most 4 sentences**, at most
   **5** enumerated items (summarise the rest), markers immediately after the
   sentence they support, no `Citations:`/`References:` trailer.
2. `complete(model=LLM_JSON_MODEL or LLM_MODEL, temperature=0.0,
   max_tokens=2048)` — the JSON model is used for prose too, because a reasoning
   model spends the budget thinking.
3. `_clean_answer()` — per line: strip markers outside `1..len(claims)`; drop
   decorative lines (only markers and a `Citations:`/`References:` label);
   attach a citation-only fragment to the previous sentence (ruling #8); split
   sentences; keep marked sentences and record their markers; for a **factual**
   sentence (≥4 words, not a `:` heading) with no marker, try to match it to a
   verified claim (§1.1) and cite it, otherwise **drop** it; keep short/heading
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
   refuse with `"The model's draft could not be tied back to the evidence."`,
   set `refusal_reason="the draft answer could not be tied back to the evidence"`,
   and emit `synthesis.refused` with `dropped_sentences`.
8. Otherwise emit `synthesis.completed` (`answer_chars`, `citations`,
   `dropped_sentences`, `implied_citations`, `claims_available`) and return.

`used_claim_ids` is `[markers[marker].claim_id for marker in used]` — real
`Claim.claim_id` values, not claim positions or `marker - 1` (§5, ruling #20).

### 1.1 Marker-less sentences: deterministic claim matching (ruling #21)

The deployed 8B model writes a faithful answer and ignores "cite every sentence"
(twice-failed prompt instruction). Stripping its sentences turned a good answer
into a refusal, so cleanup now tries to **cite** an unmarked factual sentence.

- **Rule.** The candidate is the claim that shares the largest proportion of the
  sentence's lowercase alphanumeric word set:
  `overlap = |sentence tokens ∩ claim tokens| / |sentence tokens|`.
  Accept it only when `overlap >= MIN_IMPLIED_CITE_OVERLAP` (**0.7**); on
  acceptance the marker `[n]` is appended to the sentence (`n` = claim position,
  the numbering the prompt showed) and added to `used` exactly like a
  model-written marker. Ties keep claim order (strict `>`). The threshold is
  measured, not guessed: the live draft sentence scores **0.733** against its
  claim, while `"The cutover is 2026-11-20 and the vendor agreed."` against
  `"The cutover is 2026-11-20."` scores **0.667** and must not be cited.
- **Guards, in order** (all three must pass, and a failure drops the sentence
  rather than trying a weaker claim):
  1. **Overlap** ≥ 0.7 against the best claim.
  2. **Numeric.** If the sentence contains any numeric token (`2026`,
     `08:12:29`, `PHX-150` → `150`), every such token must also appear in the
     accepted claim's text. A number the claim does not state makes the sentence
     a contradiction, not a paraphrase — `2026-12-04` cited to a claim saying
     `2026-11-20` is plausible and wrong.
  3. **Polarity.** `_is_negated()` is true when a word is an explicit negation
     cue (`not`, `no`, `never`, `none`, `without`, `cannot`, `nobody`,
     `nothing`, …) or ends in `n't`. If `_is_negated(sentence)` differs from
     `_is_negated(claim.text)`, the sentence states the inverse of its claim, so
     it is dropped. Word splitting keeps the apostrophe (`isn't` is one word);
     `_TOKEN_RE` would split it into `isn` + `t` and hide the cue.
- **Precedence.** Model-written markers always win and are never replaced or
  overridden; at most one marker is attached per sentence, and a sentence that
  already carries a marker is never matched. No model call, no clock: the whole
  pass is a deterministic function of the draft text and the claims.
- **Observability.** `synthesis.completed.implied_citations` counts markers
  attached this way. Non-zero means the model failed to comply with the citation
  instruction even though its answer survived cleanup — a prompt regression
  signal, not an error.
- **Failure mode.** Best overlap below 0.7, a missing number, or a polarity
  mismatch → the sentence is dropped exactly as before, so a fully unmarked,
  poorly-matching draft still refuses.
- **Known limits.** Semantic negation has no lexical cue, so a claim saying
  `"delayed"` still backs a sentence saying `"on time"`. And matching is
  sentence-level: a clause whose words fall inside the 0.7 budget can still ride
  along on a genuinely supported head.

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

`tests/synthesis/test_synthesizer.py` (37): cited answer kept with citations
resolved and `used_claim_ids` (+ `refusal_reason is None`) · uncited factual
sentence stripped · fully uncited draft becomes a refusal + `synthesis.refused` ·
unanswerable evidence refuses **without calling the model** and echoes the
coverage reason · invalid
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

Marker matching (ruling #21) has its own six:

- `test_markerless_draft_is_cited_to_the_claim_it_restates` — the exact live
  incident (Google security-alert email, `18 Nov 2026 at 08:12:29 +0000`, no
  markers) now delivers the sentence with `[1]`, resolves the citation, and
  reports `implied_citations == 1`.
- `test_markerless_sentence_below_the_overlap_threshold_is_dropped` — best
  overlap under 0.7 → dropped, and with nothing left the result is a refusal.
- `test_markerless_sentence_with_a_contradicting_date_is_dropped` — the numeric
  guard: `2026-12-04` against a claim saying `2026-11-20` (word overlap 0.78) is
  dropped, not cited.
- `test_compliant_draft_keeps_its_markers_and_reports_no_implied_citations` —
  explicit markers survive untouched and `implied_citations == 0`.
- `test_each_markerless_sentence_gets_its_own_claim_marker` — two claims, two
  unmarked sentences, each cited to its own claim.
- `test_implied_citation_reports_the_claim_id_not_the_marker_position` —
  `used_claim_ids == [5]` for a single claim whose `claim_id` is 5 (the matching
  step produces a position; the reported id must not become one).
- `test_negated_sentence_is_not_cited_to_a_positive_claim` — `"The PHX-150
  cutover is not delayed."` (overlap 0.857) is dropped against a positive claim.
- `test_negated_claim_does_not_back_a_positive_sentence` — the inverse direction.
- `test_unsupported_tail_is_rejected_at_the_new_threshold` — `"The cutover is
  2026-11-20 and the vendor agreed."` (0.667) is dropped, so the unsupported
  clause cannot ride along.
- `test_live_draft_sentence_scores_above_the_threshold` — the incident's real
  paraphrase pair (0.733) is still cited with `[1]` and
  `implied_citations == 1`.
- `test_apostrophe_negation_counts_on_either_side` — `isn't` is recognised on
  both the sentence and the claim side.

The rest of the file covers `claim_id`s that are not `0..n`, two claims sharing
one chunk, dedupe order, a truncated completion, and the `refusal_reason` of all
three refusal paths (including `"answer truncated by the token limit"`) plus the
additive-field round-trip.

---

## 4. Deviations from the brief

1. **No `SynthesizeRequest`; the refusal decision moved in.** The brief gives the
   synthesizer `refusal_reason` and a `session_context`; here it reads
   `verified.can_answer` / `insufficiency_reason` and has no session context.
   The brief's `SynthesisResult` fields are the frozen ones, unchanged apart
   from the additive `refusal_reason` (ruling #20).
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
  dropped claim. The same applied to a claim paraphrased into a new number;
  implied matching now checks numbers against the matched claim, but a
  model-written marker (`"... moved to 2026-12-04 [2]"` where claim 2 says
  2026-11-20) is still trusted as written, because overruling a model marker
  would be a separate decision.
- **The 0.7 threshold is measured on two data points, not calibrated.** It
  recovers the live draft (0.733) and rejects the unsupported tail (0.667), but a
  more heavily paraphrased sentence can fall below it and be dropped — a
  conservative failure, since the answer then narrows or refuses instead of
  gaining an unverified claim. The residual holes are stated in §1.1: semantic
  negation (explicit cues are covered, `"delayed"` vs `"on time"` is not) and
  clause-level composition (0.7 is a sentence-wide average).
- **Short unmarked sentences survive.** `_is_factual()` requires ≥4 words, so an
  unverified 1–3 word factual fragment ("Cutover moved.") is kept as a
  heading-like fragment — the matching pass does not apply to it. This is
  deliberate (headings must survive) and is a small hole in "no uncited factual
  sentence ships".
- **A matched claim still has to resolve to a citation.** `_implied_marker` sees
  the claims, not the pack, so a claim whose `chunk_ids` are all missing from
  `pack.chunks` produces no citation and `_renumber` removes its marker — the
  sentence then ships uncited. This is the pre-existing behaviour for
  model-written markers, and the evidence layer validates claims against pack
  chunks, so it needs a malformed pack to trigger.
- **Markers are positions; `used_claim_ids` and citations are keyed by
  `claim_id`.** A `[n]` marker is the position `prompts.build_user_prompt`
  printed, so `_claim_markers` maps position → `Claim` and everything downstream
  reads `Claim.claim_id` (regression-tested with `claim_id=5`, ruling #20). Two
  claims sharing one chunk are one delivered citation, and a claim without any
  chunk in the pack yields no citation, so its marker is removed rather than
  left dangling.
- **`quote` is the head of the chunk, not the supporting span.** If the claim's
  evidence sits deeper than 200 characters, the delivered quote does not contain
  it. The eval's structural groundedness check operates on citations, so this
  can under-score otherwise-correct answers.
- **A truncated completion now refuses.** `finish_reason` is inspected; a
  length/token-limit stop returns the deterministic refusal
  (`synthesis.refused`, reason `"answer truncated by the token limit"`) instead
  of a cut-off answer. The synthesis cap is 2048 since ruling #20: 1024 cut off
  a real 6-claim answer after ~80 s, and the cleaner drops every uncited
  sentence, so extra room cannot become uncited content. The prompt also bounds
  the answer (≤4 sentences, ≤5 enumerated items, no `Citations:` trailer).
  The refusal now carries `refusal_reason` so `refusal.detail` reports the real
  reason instead of the generic "could not be tied back to the evidence".
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
37 passed, 1 warning in 0.08s

$ .venv/bin/ruff check
All checks passed!

$ .venv/bin/mypy src
Success: no issues found in 69 source files

$ .venv/bin/python -m pytest -q
522 passed, 66 skipped, 1 warning in 7.61s
```

The `StarletteDeprecationWarning` from `fastapi.testclient` is the one warning
everywhere in this repo (pre-existing): the repo-root `tests/conftest.py`
imports `TestClient` for every run, so it shows up even in suites that never use
it.

Repo-wide context: 522 passed / 66 skipped is the 511 / 66 baseline plus the
eleven matching tests (`REDIS_URL` unset skips the cache tests). Nothing outside
`src/synthesis` changed. The live quality numbers are in
`src/evals/HANDOFF.md` and `docs/08-decisions.md` (ruling #13): eval run 2 scored
groundedness 0.20 with `pii_leaks=0` and `refusal_correct=0.90`. Matching has
**not** been re-measured through the eval harness yet — the ruling #21 fix is
verified by unit tests only.

---

## 7. Files

Read for this hand-off; `synthesizer.py`, `test_synthesizer.py` and this
document were changed for ruling #21 (marker-less citation matching):

```
src/synthesis/__init__.py      docstring only
src/synthesis/synthesizer.py   synthesize(), cleanup, citation resolution, refusals
src/synthesis/prompts.py       system prompt + numbered-claim user prompt
tests/synthesis/synth_helpers.py   chunk/claim/pack/verified builders, FakeClient
tests/synthesis/test_synthesizer.py
```
