# Retrieval Agent — Hand-off (component 03)

Status: **implemented and verified** on this machine (2026-11-17), including a
manual end-to-end run against the real local embedding model (Ollama
`nomic-embed-text`, 768 dims).

Read with: `docs/agents/03-retrieval-agent.md`, `docs/02-architecture.md`,
`docs/05-data-model.md`, `docs/06-guardrails-pii.md` (Layer 2),
`docs/08-decisions.md` (rulings + integration gaps).

---

## 1. What is implemented

| File | Contents |
|---|---|
| `src/retrieval/search.py` | `RetrieveRequest`, `EntityRef`, `acl_predicate`, `entity_predicate`, `build_candidate_query`, `merge_candidates`, `async def retrieve(...)`, audit events. |
| `src/retrieval/ranker.py` | Pure scoring: RRF, recency decay, date extraction/proximity, entity boost, word-overlap dedupe, per-document cap, score floor. **No DB, no clock, no IO.** |
| `src/retrieval/options.py` | `RetrievalOptions` (frozen dataclass) + `RETRIEVAL_*` env overrides + validated defaults. |
| `src/retrieval/errors.py` | `RetrievalError` / `RetrievalConfigError` / `RetrievalQueryError`. |
| `src/retrieval/__init__.py` | Public surface. |
| `tests/retrieval/` | 52 tests: pure ranker (21), contract + SQL-shape (14), scratch-DB behaviour (17). |

### Contract (as implemented)

```python
class EntityRef(BaseModel):
    name: str
    aliases: list[str] = []
    entity_type: str | None = None
    source_refs: dict[str, str] = {}

class RetrieveRequest(BaseModel):
    query_text: str
    entity_filters: list[EntityRef] = []
    sources: list[SourceName] | None = None      # jira | notion | gmail
    freshness_window: timedelta | None = None
    top_k: int = 12                              # 1..50

async def retrieve(
    req: RetrieveRequest,
    *,
    user_id: str,
    trace_id: str | None = None,                 # trace continuation
    session_factory=None, embedder=None, options=None, now=None,  # injection for tests/evals
) -> EvidencePack
```

`EvidencePack` / `EvidenceChunk` are imported from `src/orchestrator/pipeline.py`
and **not** redefined or modified. `EntityRef.terms` (name + aliases +
`source_refs` values) is what matching consumes, so a memory-agent registry row
maps over directly: `canonical_name → name`, `aliases → aliases`,
`source_refs → source_refs`.

### Query path

Two candidate lists, each a single SQL statement, then fusion in Python:

1. **Vector** — `ORDER BY chunks.embedding <=> :query_vector` (cosine, HNSW),
   `1 - distance` selected as `similarity`.
2. **Full text** — `WHERE chunks.fts @@ websearch_to_tsquery('english'::regconfig, :q)`,
   `ORDER BY ts_rank_cd(chunks.fts, tsquery) DESC`.

Both statements are built by the same `build_candidate_query()`, which appends
every guardrail predicate **inside the WHERE clause**:

```
documents.is_current IS TRUE                     -- integration gap #1
AND ( jsonb_exists(documents.acl->'users', :user_id)
      OR jsonb_exists(documents.acl->'users', '*') )
[AND documents.source IN (...) ]                 -- sources filter
[AND ( lower(title) LIKE … OR lower(source_id) LIKE …
       OR lower(CAST(meta->'labels' AS text)) LIKE …
       OR lower(CAST(meta->'heading_path' AS text)) LIKE … ) ]  -- entity prefilter
```

`websearch_to_tsquery` never raises on quotes/`-`/`or`, which matters because the
query text is model output. `created_at`/`updated_at`/`title`/`url`/`source_id`
come from `core.documents`; `meta` (heading path, labels, token count, fields) is
passed through untouched for the PII/evidence agents.

Fusion and scoring (`retrieval.ranker.rank_candidates`, all pure):

- **RRF** `sum(1/(60+rank))` over the two lists (k configurable). A chunk found by
  both retrievers beats one found by either alone.
- **Recency**: `(1-w) + w·0.5^(age/half_life)` with `w=0.35`, half-life 30 days,
  `age = now - documents.updated_at`. Bounded, so it reorders near-ties and can
  never bury an exact match.
- **Date proximity**: only when the query names a date (ISO `2026-12-04`, `Nov 20`,
  `20 November 2026`; a yearless form uses the request clock's year). Multiplier
  `1 + 0.25·exp(-closest_gap_days/14)`, where candidate dates come from chunk text
  plus the document's `updated_at` date.
- **Entity boost**: `1 + 0.5·matched_terms/total_terms`, matched case-insensitively
  against title, `source_id`, chunk labels and heading path (same needles the SQL
  prefilter uses).
- **Selection**: sort → collapse near-duplicates (same document **and** same
  heading path **and** Jaccard word overlap ≥ 0.82) → per-document cap (default 4)
  → drop everything below the final-score floor (default 0.01) → `top_k`.
- **Staleness**: `age > freshness_window` sets `EvidenceChunk.stale=True`; the
  chunk is still returned, because the planner decides whether a live fetch is
  needed (`docs/06-guardrails-pii.md` Layer 2).

### ACL semantics (the exact rule)

A document is visible only when

```
:user_id  ∈ documents.acl.users      OR      "*" ∈ documents.acl.users
```

- **Fail closed.** Missing `acl`, missing `users` key, empty list, or a non-object
  `acl` (so `acl->'users'` is NULL) ⇒ invisible. Nothing defaults to allow.
- **`acl.groups` / `acl.labels` are deliberately not consulted.** Resolving a
  user's group membership needs an identity source the single-user MVP does not
  have, and treating "member of any mirrored group" as access would grant every
  document to everyone. `acl_predicate()` is the single place to change when the
  identity layer lands.
- **Enforced in SQL, never post-filtered.** There is no Python-side ACL pass to
  bypass; the predicate lives in both candidate statements and the test asserts
  it is inside the `WHERE` clause of the compiled SQL (`tests/retrieval/test_contract.py`),
  not just that a row was absent.

### Events (`audit.events`)

| Event | When | Payload highlights |
|---|---|---|
| `retrieval.ran` | always (even for an empty pack) | `query`, `user_id`, `sources`, `entity_terms`, `freshness_window_s`, `top_k`, `vector_candidates`, `fts_candidates`, `merged_candidates`, `deduped`, `dropped_below_floor`, `returned`, `stale_chunk_ids`, `top_score`, `degraded_reason`, `embedder`, `embed_tokens`, `score_floor`, `rrf_k`, `min_vector_similarity`, plus `latency_ms` |
| `retrieval.fallback` | query embedding failed | `reason` (exception type + message), `mode="fts_only"`, `query`, `embedder`, `top_k` |

Written through `db.tracing.Tracer` + `make_sink(session_factory)` (the database
sink by default, so traces are real `audit.events` rows). A failing trace write
never raises out of retrieval — that policy belongs to the API layer.

### Failure modes

| Failure | Behaviour |
|---|---|
| No `DATABASE_URL` | raises `RetrievalConfigError` (a deployment error, not a degraded answer) |
| Query embedding raises (provider down, auth, timeout, bad dimension) | FTS-only, `degraded_reason` set, `retrieval.fallback` emitted. Never crashes, never returns vector-only |
| Query embedding is non-finite (zero vector ⇒ NaN cosine) | those vector rows are skipped |
| Zero candidates | empty `EvidencePack`, `dropped_below_floor=0`; no padding, no invented rows |
| Everything below the floor | empty pack + `dropped_below_floor` count (audit carries the number) |
| Malformed FTS input (`"`, `-`, `or`, stopwords only) | `websearch_to_tsquery` yields an empty tsquery ⇒ that list is empty; no error |

---

## 2. How to run

```bash
cd /home/varun/projects/agentic_rag_over_live_data

# pure + SQL-shape tests only (no database, no network)
.venv/bin/python -m pytest tests/retrieval -q

# with the scratch database (seeds fixtures/corpus, runs PDF-free offline embeddings)
set -a; source ~/.config/agentic-rag/dev.env; set +a
.venv/bin/python -m pytest tests/retrieval -q

# lint / types (my scope)
.venv/bin/ruff check src/retrieval tests/retrieval
.venv/bin/mypy src/retrieval src/orchestrator
```

### Scratch database (provisioned once, already done here)

```bash
sudo -u postgres createdb -O rag agentic_rag_retrieval_test
sudo -u postgres psql -d agentic_rag_retrieval_test \
  -c "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_trgm;"
```

`tests/retrieval/conftest.py` derives the URL from `DATABASE_URL`
(`…/agentic_rag_retrieval_test`) or `RETRIEVAL_TEST_DATABASE_URL`, **refuses**
any database name not ending in `_retrieval_test`, runs `alembic upgrade head`
(so it picked up agent 01's `0003_documents_partial_unique` automatically), and
truncates `core.documents` (cascades to chunks) + `audit.events` before each test.
The dev database is never written. Without `DATABASE_URL` the 17 DB tests skip
with an explicit reason.

### Manual end-to-end check against the real model (opt-in, not in pytest)

Seeded the scratch DB with **real** `nomic-embed-text` vectors (94 chunks,
11 310 tokens, one batch) and ran three retrievals:

```
$ .venv/bin/python /tmp/retrieval_live_smoke.py     # 2.2 s, warm Ollama
embedder: nomic-embed-text 768
indexed: {'chunks_written': 94, 'chunks_skipped': 0, 'embed_tokens': 11310}
degraded: None
  0.03223  jira   PHX-150                  PHX-150 — Cut over billing webhooks to Ledger v2
  0.03223  jira   PHX-150                  PHX-150 — Cut over billing webhooks to Ledger v2
  0.01529  jira   PHX-142                  PHX-142 — Auth redirect loop after SSO cutover
  0.01517  notion d1a9e6b3-2c58-4f70-9b4d-7e8a3c6f0d21 Alice Migration Cutover Plan
  0.01503  jira   PHX-166                  PHX-166 — Accessibility audit of the billing set
atlas top: [('ATL-88', 0.0244), ('ATL-88', 0.024), ('2b7e4d19-8c63-4f20-a5b8-6d1e9c3a7f04', 0.0236)]
stranger sees: 0 chunks
```

The two 0.03223 rows are the same document at rank 1+2 of *both* lists
(`1/61 + 1/62`); the Atlas query returns only Atlas material; the user outside
`acl.users` sees nothing. That is the real endpoint, real vectors, real SQL.

---

## 3. Tests

`tests/retrieval/test_ranker.py` (no DB): RRF arithmetic and agreement
preference; bounded exponential recency decay + monotonicity; ISO/month-name date
parsing incl. impossible dates and `PHX-150` non-dates; date-proximity bonus;
entity needle expansion and boost; word-overlap Jaccard; newness tie-break;
"newer wins at equal relevance"; duplicate collapse; different-heading
retention; per-document cap; floor dropping; deterministic tie ordering; empty
input.

`tests/retrieval/test_contract.py` (no DB): frozen-pack identity; `RetrievalConfigError`
without a database; request bounds and source literal validation; `EntityRef.terms`;
**ACL predicate compiled into the `WHERE` clause** with both bind values; empty
identity refused; source and `is_current` predicates present; entity prefilter
applied only when asked, with LIKE metacharacters escaped; vector query requires
an embedding and skips NULL vectors; FTS binds the raw query text and pins
`'english'::regconfig`; options env overrides and validation.

`tests/retrieval/test_search_db.py` (scratch DB, corpus seeded + indexed):
known-relevant chunk ranks top-3 (PHX-150 first) · entity filter restricts to
Project Atlas · unknown entity → empty pack · `sources` filter · `top_k` bound ·
ACL denial for a non-listed user (with a raw row count proving exclusion, not
absence) · ACL `*` wildcard grants exactly one document · superseded document
(`is_current=false`, chunks intentionally left behind) excluded · recency orders
equally relevant chunks · freshness window flags but returns · score floor drops
the tail · `min_vector_similarity` gate can empty the pack · near-identical
chunks collapsed (trace `deduped ≥ 1`) · embedder failure → FTS-only pack +
`retrieval.fallback` + `vector_candidates == 0` · `retrieval.ran` payload and
latency · frozen-contract round-trip of the pack · repeatability.

---

## 4. Deviations from the brief (for Lead review)

1. **No `recency_rank` field.** The brief's sketch has
   `recency_rank: list[UUID] | None` on the pack; the frozen
   `orchestrator.pipeline.EvidencePack` (Lead-owned, "import, do not modify") does
   not. Staleness is therefore per-chunk (`stale`) plus `stale_chunk_ids` in the
   `retrieval.ran` payload. If the planner needs the list in the pack, that is a
   frozen-contract change and the Lead's call.
2. **Tunables live in `src/retrieval/options.py`, not `src/config.py`.** `config.py`
   is shared with other agents mid-flight and these are pipeline tunables, not
   deployment settings. Overridable via `RETRIEVAL_RRF_K`,
   `RETRIEVAL_RECENCY_HALF_LIFE_DAYS`, `RETRIEVAL_RECENCY_WEIGHT`,
   `RETRIEVAL_DATE_PROXIMITY_WEIGHT`, `RETRIEVAL_DATE_PROXIMITY_DECAY_DAYS`,
   `RETRIEVAL_ENTITY_WEIGHT`, `RETRIEVAL_SCORE_FLOOR`, `RETRIEVAL_DEDUPE_SIMILARITY`,
   `RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT`, `RETRIEVAL_CANDIDATE_DEPTH`,
   `RETRIEVAL_MIN_VECTOR_SIMILARITY`, `RETRIEVAL_ENTITY_FILTER_MODE`.
   `.env.example` is not mine to edit — the Lead may want them documented there.
3. **`EntityRef` is defined here** (the brief allowed it). Fields: `name`,
   `aliases`, `entity_type`, `source_refs`; extra keys are ignored so the memory
   agent can pass a registry row without trimming it.
4. **Entity SQL filter is a permissive OR prefilter**, not an exact matcher:
   any alias (or any ≥4-char word of a multi-word alias) substring-matching
   title / `source_id` / chunk labels / heading path. The authority on relevance
   is the ranking boost. `RetrievalOptions.entity_filter_mode="boost"` turns the
   SQL restriction off (ranking still boosts) for callers who do not want the
   empty-result cliff.
5. **ACL consults `acl.users` + `"*"` only** — `groups`/`labels` are mirrored but
   ignored until an identity source exists (rationale in §1). This is the
   semantics the brief asked me to document.
6. **`min_vector_similarity` added** (not requested). Dense ANN *always* returns
   its k neighbours, so without a gate an off-topic question still yields a
   full pack; the gate also removes NaN rows from a zero query vector. Default
   `0.0` (off) because the right cosine cutoff is model-specific
   (`nomic-embed-text` scores unrelated text around 0.5). Relevance by default is
   fusion + floor + the evidence stage's sufficiency gate.
7. **Score floor applies to the final score** (fused × multipliers), so a stale
   or entity-unmatched chunk can fall out. `dropped_below_floor` counts only
   floor drops; duplicate/cap removals are reported separately as `deduped` in
   the trace (the pack has one counter, not two).
8. **Per-document cap (default 4) in addition to near-duplicate collapse.**
   Adjacent chunks of a long issue share only the chunker's ~15% overlap, so
   overlap-based dedupe cannot stop one cutover issue from filling a 12-slot
   pack; the cap does. Duplicate collapse alone is the literal brief, the cap is
   the behaviour it asked for.
9. **Two SQL statements + Python fusion**, not one SQL statement with RRF in SQL.
   The brief puts RRF in `ranker.py` as pure unit-testable code, and that is
   worth more than saving one round trip at this scale (~50+50 candidates).
   Both statements share `build_candidate_query()`, so the ACL/filter predicates
   cannot drift apart.
10. **`retrieve()` raises `RetrievalConfigError` with no database** instead of
    returning an empty degraded pack: an unconfigured DB is a misconfiguration,
    and an empty pack would look like "no evidence found".
11. **`retrieve()` takes optional keyword-only collaborators**
    (`session_factory`, `embedder`, `options`, `now`, `trace_id`). The brief's
    `retrieve(req, *, user_id)` call is unchanged; the extras are what let tests
    and the eval harness run offline and pin the clock.

No new dependencies. No schema change/migration. No edits to
`src/orchestrator/`, `src/api/`, `src/connectors/`, `src/indexing/`, `src/pii/`,
`src/evidence/`, `src/evals/`, `fixtures/`, or any shared doc.

---

## 5. Uncertainty / risks a reviewer should know

- **RRF scale and the floor are coupled.** With `k=60` a top result is ~0.033 and
  a single-list rank-40 result is ~0.010 — which is exactly the default floor.
  Changing `RETRIEVAL_RRF_K` changes what `RETRIEVAL_SCORE_FLOOR` means; rescale
  them together.
- **HNSW annotation post-filtering.** The vector query joins `core.documents` and
  filters on ACL/`is_current`/source, and pgvector applies the ANN index *before*
  those filters, so a highly selective filter can return fewer than
  `candidate_depth` vector rows (it never leaks rows, it can just under-fill).
  `hnsw.iterative_scan` (pgvector 0.8) is the fix if that ever bites; not enabled
  because it changes planning for every query. FTS is unaffected and fusion covers
  the gap at corpus scale.
- **Dense retrieval never returns "nothing"** at the default settings — an
  unrelated question still gets k neighbours with a low fused score. The
  protection is the floor, `min_vector_similarity`, and the evidence agent's
  sufficiency gate (documented above), not the retriever.
- **`updated_at` is assumed timezone-aware** (guaranteed by `timestamptz`); a
  naive value would raise in `now - updated_at` rather than silently misorder.
- **ACL test data is corpus-shaped**: all Phoenix documents list the same five
  users and one Nimbus email is single-user, which is why ACL denial is testable
  without mutating rows. Group-based ACLs are untested because they are not
  implemented.
- **Entity matching is substring-based.** "PHX" matches "PHX-150" (good) and would
  also match a hypothetical "PHX-OTHER" (acceptable); "Alice Migration" matches
  the code-name documents even though it is not a person. Precision here is the
  registry's job, not the retriever's.
- **One request = one clock.** Concurrent syncs can move `updated_at` between the
  two statements; retrieval is read-only and this is a ranking nuance, not a
  correctness issue.
- **Unverified**: behaviour at a corpus size where HNSW filtering matters
  (>100k chunks), and any embedding provider other than the local Ollama
  endpoint (the embedder is the indexer's client, only exercised here offline and
  against Ollama).

---

## 6. Verified gates (real output, 2026-11-17)

Scoped to this component (the full-repo gates currently fail only in other
agents' in-flight directories — see the note below):

```
$ .venv/bin/ruff check src/retrieval tests/retrieval
All checks passed!

$ .venv/bin/mypy src/retrieval src/orchestrator
Success: no issues found in 8 source files

$ set -a; source ~/.config/agentic-rag/dev.env; set +a
$ .venv/bin/python -m pytest tests/retrieval -q
52 passed, 1 warning in 12.55s

$ env -u DATABASE_URL .venv/bin/python -m pytest tests/retrieval -q
35 passed, 17 skipped, 1 warning in 0.06s
    # the 17 skips are the scratch-DB tests, skipped with an explicit
    # "DATABASE_URL is not set" reason.

$ .venv/bin/ruff check                       # repo-wide
Found 8 errors.   (src/evidence/{service,sufficiency,validator}.py,
                   src/pii/{audit,recognizers,scrubber}.py — not this component)

$ .venv/bin/mypy src                         # repo-wide
Found 7 errors in 4 files  (src/evidence/service.py 2, src/pii/{recognizers,analyzer,scrubber}.py 5)

$ .venv/bin/python -m pytest -q              # repo-wide, DB set
1 failed, 367 passed, 5 skipped, 1 warning in 17.51s
    # tests/db/test_migrations.py::test_migrations_are_at_head pins
    # "0002_local_embedding"; agent 01's migration 0003 is now head. Not a
    # retrieval test and not a file I own.
```

The single warning everywhere is the third-party
`StarletteDeprecationWarning` from `fastapi.testclient` (pre-existing).

---

## 7. What I did **not** do (scope)

- No PII scrubbing, no evidence/claim verification, no synthesis, no
  orchestration, no connector or indexer changes.
- No LLM calls of any kind: retrieval never touches a completion endpoint, and
  `EvidenceChunk` never leaves this component unscrubbed (the scrubber is the
  only path forward).
- No schema change and no migration.
- No `src/*/HANDOFF.md` or shared-doc edits outside `src/retrieval/`.
- No git operations (git is not installed on this machine).
- No secrets, no `.env` changes; the database password is only ever read from
  the environment.
