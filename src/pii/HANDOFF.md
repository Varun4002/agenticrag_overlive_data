# PII Scrubber — Hand-off (component 04)

Status: **implemented**, tests pass on this machine (verified 2026-11-18). This
hand-off was written after the fact from the code and tests — Wave 2 shipped the
component without one, which `docs/08-decisions.md` (Wave 2 section) flags.

Read with: `docs/agents/04-pii-scrubber-agent.md` (brief),
`docs/06-guardrails-pii.md` (the policy this implements),
`docs/05-data-model.md` (`audit.redactions`), `docs/08-decisions.md`
("PII detection uses `en_core_web_md`", ruling #3, ruling #4).

---

## 1. What is implemented

| File | Contents |
|---|---|
| `src/pii/scrubber.py` | `scrub()` — the only entry point. Policy filtering, overlap resolution, placeholder allocation, audit records, `pii.scrubbed` event. |
| `src/pii/analyzer.py` | Presidio analyzer/anonymizer seam: `SPACY_MODEL`, `DEFAULT_ENTITIES`, `DetectedSpan`, `Engines`, `get_engines()`, `reset_engines()`, `analyze_text()`, `analyze()`, `anonymize_text()`. |
| `src/pii/recognizers.py` | Five custom recognizers (`EMAIL_ADDRESS`, `API_KEY`, `PRIVATE_KEY`, `ADDRESS`, `DATE_OF_BIRTH`), `PersonRegistryRecognizer`, and the shape heuristics `person_like()`, `location_like()`, `shannon_entropy()`. |
| `src/pii/allowlist.py` | `AllowList`, `COARSE_GEO`, `load_allowlist_terms()`, `default_allowlist()`, `build_allowlist()`, `AllowListError`. |
| `src/pii/placeholders.py` | `PlaceholderRegistry`, `PLACEHOLDER_PATTERN`, `LABELS`, `LABEL_BY_ENTITY`, `PII_TYPE_BY_LABEL`, `normalize_value()`. |
| `src/pii/audit.py` | `RedactionRecord`, `salted_hash()`, `resolve_salt()`, `DEV_ONLY_SALT`, `RedactionWriter` protocol with `DatabaseRedactionWriter` / `NullRedactionWriter`, `make_redaction_writer()`. |
| `src/pii/errors.py` | `PiiError` → `PiiUnavailableError` / `PiiScrubError`. |
| `tests/pii/` | 17 tests in `test_scrubber.py` + `test_policy.py`, driven by the planted PII list in `fixtures/corpus/manifest.json`. |

### Contract (as implemented)

```python
async def scrub(
    pack: EvidencePack,                      # frozen Lead contract, never modified
    *,
    trace_id: str | None = None,             # audit rows + events; generated + warned when omitted
    known_person_names: list[str] | None = None,   # entity-registry stand-in
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    allowlist_terms: Iterable[str] | None = None,  # extra non-person terms
) -> ScrubbedPack
```

The brief's `ScrubRequest`/`scrub(req)` shape is not implemented; see §4.1.
`ScrubbedPack` carries `question`, `chunks`, `redaction_count`, `degraded_reason`
and each chunk gains `redacted_types` — all frozen fields, none added here.

### Behaviour, in the order it happens

1. **Policy assembly.** `PII_ENTITIES_ENABLED` (or `DEFAULT_ENTITIES`) is the
   entity set; `known_person_names` is de-duplicated; the allowlist is the
   configured one (`PII_ALLOWLIST_PATH`, else the repo corpus manifest) plus
   `allowlist_terms`; the salt comes from `PII_HASH_SALT`.
2. **Placeholder seeding.** Every existing `[LABEL_n]` in chunk text and titles
   raises the per-label counters (`PlaceholderRegistry.observe`), so a second
   pass over already-scrubbed text changes nothing and never re-redacts a
   placeholder. All existing placeholders are masked with spaces before analysis
   so spaCy cannot fold one into a new span (offsets stay valid).
3. **Analysis per field.** Chunk **text and title** are analysed separately.
   `analyze()` runs Presidio in a worker thread under one process-wide
   `threading.Lock` (spaCy pipelines are not thread-safe).
4. **Policy filtering.** For each detected span:
   - unknown entity class → `PiiScrubError` (fail closed): a detected span we
     cannot map is a mapping bug, and skipping it would be a silent leak;
   - overlaps an existing placeholder → skipped;
   - `PERSON`/`LOCATION`/`ADDRESS` overlapping an allowlisted term →
     suppressed (`allowlisted_spans`);
   - `PERSON` with no registry identity: when a registry was supplied the span
     is **declined** (registry is the precision signal); without one it is kept
     only if `person_like()` (2–3 capitalised alphabetic tokens);
   - `LOCATION` kept only if `location_like()` (1–4 capitalised tokens, no
     digits) — this is what stops `PHX-131` and `order_id` becoming places.
   Hard-signal classes (email, phone, card, keys) are never suppressed.
5. **Overlap resolution** (`ENTITY_PRIORITY`): private key < API key < email /
   card < phone < DOB < address < person < location. Without it a spaCy PERSON
   span over "Email tobias.lindqvist@…" would swallow the address.
6. **Allocation + audit.** Each accepted span gets a pack-scoped placeholder by
   `(label, normalised identity)`, so `Priya`, `Priya Raghunathan` and
   `priya.raghunathan@…` resolve consistently. Each redaction appends a
   `RedactionRecord` holding the **salted sha256** of the original value; the
   original is never stored, logged, or put in an event payload.
7. **Finish.** Redaction rows go through the writer, then one `pii.scrubbed`
   event (agent `pii`) carries: `chunks`, `redactions`, `by_type`,
   `chunks_with_redactions`, `placeholders`, `detected_spans`,
   `allowlisted_spans`, `declined_person_spans`, `declined_location_spans`,
   `person_registry_size`, `entities_enabled`,
   `audit_persisted`; the event itself carries `latency_ms`.

### Failure modes

| Failure | Behaviour |
|---|---|
| Presidio not importable | `PiiUnavailableError`; no pack |
| `en_core_web_md` missing / engine build fails | `PiiUnavailableError` naming the download command |
| `analyze()` raises (any reason) | `PiiUnavailableError`; no pack |
| Anonymizer raises, or a span cannot be mapped to its placeholder | `PiiUnavailableError` (fail closed rather than ship unmapped text) |
| Any other exception mid-scrub | `PiiScrubError`; no pack |
| `audit.redactions` insert fails | swallowed, `audit_persisted=false`, pack still returned (the `AUDIT_STRICT` refusal lives in `api/routes.py`) |
| No database configured | `NullRedactionWriter`: rows logged at debug, `audit_persisted=false` (truthful — nothing was persisted; see §5) |
| Empty chunk text | returned unchanged; no analyzer call |
| Called without `known_person_names` | warning logged once per call; PERSON falls back to shape heuristics |

---

## 2. How to run

```bash
cd /home/varun/projects/agentic_rag_over_live_data

.venv/bin/python -m pytest tests/pii -q         # needs the spaCy model, no DB/network
.venv/bin/ruff check src/pii tests/pii
.venv/bin/mypy src/pii
```

The `engines` fixture in `tests/pii/conftest.py` **skips** (never fails) when
Presidio or the model is unavailable, so a green run means the real model was
exercised. A scratch database (`agentic_rag_pii_test`, override
`PII_TEST_DATABASE_URL`) is provisioned by the same conftest — see §3 for why
nothing currently uses it.

---

## 3. Tests — and what they do not cover

`tests/pii/test_scrubber.py` (7): every planted value in
`manifest.pii.kinds` is absent from the scrubbed corpus pack (titles included)
· pack shape preserved field by field · placeholders stable across chunks and
mention forms, numbering order deterministic · `redaction_count` and
`redacted_types` agree with the records · titles scrubbed · first-name and
full-name mentions share one placeholder · without a registry, a name-shaped
person is still redacted.

`tests/pii/test_policy.py` (10): the `Alice Migration` codename survives (with
and without a registry) · role language, issue keys, statuses and due dates
survive · project names and coarse geo survive · a physical location is
redacted · issue keys are not locations · caller-supplied allowlist terms
survive · the corpus keeps `Project Phoenix`, `PHX-150`, `2026-11-20`,
`Ledger v2`, `EU incident`, `SEV-1` · injection text is not special-cased (the
scrubber redacts PII only) · an empty pack passes through.

**Coverage note:** the `audit.redactions` write path is now exercised by
`tests/pii/test_audit.py` (DB-gated on the existing `pii_session_factory` /
`clean_pii_db` fixtures, refusing any database not named `*_pii_test`): one row
per redaction with the right type/placeholder, a salted `original_hash`, and no
original value in any column. Fail-closed behaviour is covered by
`tests/pii/test_failures.py`.

**Still not covered:**

- Analyzer lock contention under concurrent requests (the lock is untested for
  contention).
- Idempotency of a second scrub pass (the seeding logic is implemented).

---

## 4. Deviations from the brief

1. **No `ScrubRequest` model; `scrub(pack, **kwargs)` instead.** The brief's
   `ScrubRequest.evidence` + `registry_persons` map directly onto the first
   argument and `known_person_names`; the extra keyword-only collaborators exist
   so tests and the dashboard can inject a trace id, a session factory and
   allowlist terms without a request wrapper.
2. **`en_core_web_md`, not `en_core_web_sm`** — cross-cutting decision in
   `docs/08-decisions.md` (2026-11-17): the small model tags "Grace Adeyemi" as
   ORG and silently leaked the name; the medium model tags it PERSON while still
   leaving "Nimbus Systems" as ORG. The model path is pinned in
   `analyzer.py::SPACY_MODEL`; it is not a pip dependency (`python -m spacy
   download en_core_web_md`).
3. **Salted-hash audit instead of storing originals** — the design the brief
   asked for (`docs/agents/04` §4). `salted_hash()` is sha256 over
   `salt \x00 normalised-value`; `PII_HASH_SALT` defaults to a documented
   dev-only constant and `resolve_salt()` warns once when it is in use. No
   ruling governs the salt specifically; `docs/06-guardrails-pii.md` (audit
   policy) is the requirement, and `AUDIT_STRICT` is the cross-cutting ruling.
4. **Presidio's email recognizer is removed and replaced.** The stock
   `EmailRecognizer` validates with `tldextract`, which does not know the
   RFC 2606 `.example` TLD — *every* corpus email was dropped. The replacement
   is a structural regex with no TLD lookup (which also keeps the scrubber
   offline). `DATE_TIME` is not in the enabled entity set at all: the module
   docstring records that Presidio's date recognizer redacts every deadline,
   which the policy forbids, so only explicit birth phrases are matched by
   `DateOfBirthRecognizer`.
5. **Registry augmentation is wired but unused in the live pipeline.** The
   orchestrator calls `scrub(pack, trace_id=…, session_factory=…)` with no
   `known_person_names` (`src/orchestrator/run.py`), per ruling #4: the entity
   registry is empty, so a partial roster would narrow PERSON redaction and leak
   everyone missing it. Only the dashboard's `/admin/search` supplies names
   (defaulting to the fixture corpus roster).
6. **The `PERSON`-without-registry path redacts shape, not identity.** With no
   registry the scrubber accepts any 2–3 token capitalised alphabetic span; the
   docstring and §5 record the precision cost. This is ruling #4's accepted
   failure direction (over-redaction over leakage).
7. **A scrub failure becomes a `guardrail_blocked` refusal, not a 5xx** — ruling
   #3, implemented in the orchestrator, not here.

---

## 5. Uncertainty / risks

- **Over-redaction of org names without a registry.** Every name-shaped PERSON
  becomes `[PERSON_n]` when no registry is supplied, and `en_core_web_md` does
  label some multi-word org/product strings PERSON. On a real tenant the
  allowlist must be pointed at a registry-shaped JSON (`PII_ALLOWLIST_PATH`);
  otherwise the default signal is the *fixture corpus manifest*, which knows
  nothing about real project names.
- **`audit_persisted` is truthful now.** `NullRedactionWriter.write()` returns
  `not records`, so a no-database deployment reports `audit_persisted: false`
  for rows it only logged. `AUDIT_STRICT` in the API layer is what actually
  refuses unauditable answers.
- **`errors.py` matches the code now.** Unmapped entity class ⇒
  `PiiScrubError` (fail closed); analyzer/anonymizer failure ⇒
  `PiiUnavailableError`. Both are covered in `tests/pii/test_failures.py`.
- **`PII_TYPE_BY_LABEL` is lossy against the frozen enum:** `LOCATION` audits as
  `address`, `CREDIT_CARD` and `DOB` audit as `other`. Per-class counts in
  `audit.redactions` therefore do not distinguish those three.
- **Placeholders are pack-scoped, not request-scoped.** `[PERSON_1]` means a
  different person in the next request; coreference only holds inside one pack.
  This is what the brief asked for, but a reviewer reading two traces should not
  compare placeholder numbers.
- **Detection is model-bound.** Person/org precision and location recall come
  from whichever spaCy model is installed; swapping the model silently changes
  the false-positive rate. There is no threshold tuning (`score_threshold=0.0`;
  policy filtering, not confidence, decides).
- **Latency:** two analyzer calls per chunk (text + title) under a global lock,
  so one request serialises all scrubbing in the process. Not measured here.
- **Over-redaction is not reported as a metric.** The counters live in the
  `pii.scrubbed` payload, but no `guardrail.over_redacted` event is emitted, so
  `evals.trace_report.aggregate_stats` still reports the over-redaction rate as
  "no data".
- **Not verified:** any real tenant corpus, any other spaCy model, behaviour
  under concurrent requests (the lock is untested for contention), and the
  allowlist file path (only the default manifest path is exercised).

---

## 6. Verified gates (real output, 2026-11-18)

```
$ .venv/bin/python -m pytest tests/pii -q
17 passed, 1 warning in 4.96s
    # the model was available: the `engines` fixture did not skip.

$ .venv/bin/ruff check src/pii tests/pii
All checks passed!

$ .venv/bin/mypy src/pii
Success: no issues found in 8 source files
```

One ad-hoc measurement against the fixture corpus (`tests/pii/pii_fixtures.py`,
44 chunks, registry = `manifest.pii.kinds.person`), no DATABASE_URL, dev salt:

```
chunks: 44 redactions: 148
by type: {'address': 6, 'api_key': 2, 'email': 13, 'person': 39, 'phone': 8}
chunks with redactions: 41
```

Three chunks needed no redaction at all. The corpus plants a fixed list, so
these are sanity-check counts on that list — not a recall measurement, and not a
bound on what a real tenant contains.

The third-party `StarletteDeprecationWarning` from `fastapi.testclient` is the
one warning everywhere in this repo (pre-existing).

---

## 7. Files

Read for this hand-off (no code changed):

```
src/pii/__init__.py            exports scrub(), errors, RedactionRecord, salted_hash
src/pii/scrubber.py            scrub() + policy pipeline
src/pii/analyzer.py            Presidio engine, threading lock, anonymizer seam
src/pii/recognizers.py         five PatternRecognizers + PersonRegistryRecognizer
src/pii/allowlist.py           terms that must never be redacted
src/pii/placeholders.py        [LABEL_n] allocation + entity→audit-type maps
src/pii/audit.py               salted hash, redaction writer protocol
src/pii/errors.py              typed failures
tests/pii/conftest.py          scratch-DB fixtures (unused), engines skip, corpus pack
tests/pii/pii_fixtures.py      pack/chunk builders from fixtures/corpus
tests/pii/test_scrubber.py     7 end-to-end scrub tests
tests/pii/test_policy.py       10 must-not-redact / must-redact tests
```

No edits were made to any `.py` file, `docs/08-decisions.md`, or a shared doc
while producing this hand-off.
