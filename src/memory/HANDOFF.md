# Memory — Hand-off (component 08)

Status: **Tier 2 only, implemented**. The Redis answer cache is live in the
request path and its offline tests pass on this machine (2026-11-18). Tier 1
(session scratchpad) and Tier 3 (entity registry) are **not implemented**, and
**the Redis path has not been exercised against a live Redis in this round** —
the five live tests skip without `REDIS_URL`, and no live run is recorded here.

Read with: `docs/agents/08-memory-agent.md` (brief), `docs/07-memory-design.md`
(the three-tier spec, now annotated with what exists), `docs/05-data-model.md`
(`mem.*` tables), `docs/08-decisions.md` (ruling #14).

---

## 1. What is implemented

| File | Contents |
|---|---|
| `src/memory/answer_cache.py` | The whole component: `normalize_question()`, `cache_key_hash()`, `cache_key()`, `AnswerCacheProtocol`, `NullAnswerCache`, `RedisAnswerCache`, the process-wide `get_answer_cache()` / `aclose_answer_cache()` / `reset_answer_cache()`, `build_answer_cache()`, `invalidate_answer_cache()`, `CACHE_PREFIX`, `VERSION_KEY`. |
| `src/memory/__init__.py` | Docstring only; callers import `memory.answer_cache`. |
| `tests/memory/test_answer_cache.py` | 21 test items (16 offline, 5 live-Redis; one parametrised case is four items). |
| `src/orchestrator/run.py` | The hot path: read before retrieve, write after the outcome, `memory.cache_hit` event. |
| `src/connectors/service.py`, `src/api/admin.py` | Invalidation callers (sync / backfill / seed / explicit route). |
| `src/config.py` | `redis_url` (`REDIS_URL`), `answer_cache_ttl_s` (900), `answer_cache_enabled` (true). |

### Key and value

```
key   = "answer_cache:" + sha256( normalized_question "|" user_id "|" version )
value = AskResponse.model_dump_json()      # answer, citations, refusal, trace_id, degraded
```

- `normalize_question()` lowercases, collapses whitespace runs and strips, so
  `"When is  the cutover?"` and `"when is the cutover?"` share an entry.
- The key is **per user** because retrieval enforces `acl.users` in SQL: the same
  question can legitimately have different evidence for two users. It is hashed
  so the question text (which can contain names) never reaches Redis or a trace
  payload; the same hash is what `memory.cache_hit` carries.
- `VERSION_KEY = "answer_cache:version"` is a monotonic counter. It lives inside
  the prefix but is exempt from `clear()`, so it can never move backwards.

### `AnswerCacheProtocol` — failure policy is the interface

| Operation | Contract |
|---|---|
| `get(...) -> AskResponse \| None` | Never raises. Miss, outage, unreadable value (schema drift) → `None`. |
| `put(..., ttl_s)` | Never raises. A lost write costs the next request one full run. |
| `version() -> int` | **May raise** when the backend is unreachable: callers must run without a cache rather than serve an entry whose data version is unknown. |
| `bump_version() -> int` | **Raises** on failure — a silently dropped invalidation would serve stale answers. |
| `clear() -> int` | **Raises** on failure; SCANs `answer_cache:*` (never `FLUSHDB`, the Redis DB is shared) and skips the version key. |
| `aclose()` | Never raises. |

`RedisAnswerCache` sets `decode_responses=True` and 2 s socket connect/read
timeouts, and warns once per process on backend failure (`_warn_once`) so an
outage does not spam the log. `NullAnswerCache` never hits and never stores
(`get` → `None`, `version`/`bump_version` → 0, `clear` → 0); it is what tests and
a deliberately cache-less orchestrator use.

`get_answer_cache(settings=None)` resolves once per process: `answer_cache_enabled
= false` → `None` (info log), no `REDIS_URL` → `None` (warning), otherwise a
`RedisAnswerCache`. `build_answer_cache()` raises `ValueError` without a URL.

`invalidate_answer_cache(settings=None) -> int | None` returns `None` when no
cache is configured. Both a failed bump **and** a cache that cannot even be
constructed (malformed `REDIS_URL`) are logged at **error** and also return
`None`: the ingestion write already committed, so the worst case is stale entries
until TTL — visible, never fatal.

### How the orchestrator uses it

1. **Read** (`_cache_context` → `_cache_get`): resolve the cache and its version;
   a version read failure leaves `version=None`, which disables both the read and
   the write for that request. Any `get` exception is a miss. A hit returns
   `cached.model_copy(update={"trace_id": <live trace>})` before retrieval runs,
   and emits `memory.cache_hit` (`original_trace_id`, `version`,
   `cache_key_hash`) then `orchestrator.completed` with `status="cache_hit"`.
2. **Write** (`_cache_store`): only when the version is known and
   `store_allowed()` approves the response:
   - `degraded=True` → never cached (retrieval fell back; a retry can do better);
   - `refusal.reason == "error"` → never cached (the LLM may recover);
   - answered responses **and** `insufficient_evidence` / `out_of_scope` /
     `guardrail_blocked` refusals → cached (a refusal is a real outcome of this
     evidence).
   TTL is `ANSWER_CACHE_TTL_S` (default 900 s).
3. Every completed/refused trace payload carries `"cached": true|false`.

### Invalidation callers (all verified in code)

| Caller | When |
|---|---|
| `connectors.service.sync()` | after any committed sync; result recorded as `answer_cache_version` in `connector.sync.completed`; a failure is logged, never fatal to the sync |
| `POST /admin/backfill` | when `chunks_written > 0` or `force=true`; version returned as `answer_cache_version` |
| `POST /admin/seed` | after seeding the fixture corpus |
| `POST /admin/cache/invalidate` | `clear()` **and** `bump_version()`; no database required; `{"invalidated": false, "reason": …}` when `REDIS_URL` is unset, 502 on a Redis error |

The dashboard's explicit route is the only one that surfaces a Redis failure to
the caller; the ingestion paths deliberately degrade to "stale until TTL" instead
of failing a committed write.

---

## 2. What is stubbed / missing relative to the brief

The brief asks for `recall()`, `remember()`, `record_turn()`, `update_registry()`
and three tiers. None of those functions exist. Concretely:

- **No Tier 1 session memory.** `mem.session_turns` exists in the schema and is
  never written or read; `session_id` is accepted on `/ask` and only appears in
  the `orchestrator.intake` payload. The orchestrator's plan event lists
  `memory_check` as a skipped stage.
- **No Tier 3 entity registry.** `mem.entity_registry` is never written; the
  registry builder (candidate scan, trgm clustering, LLM merge confirmation,
  corrections outranking inference) does not exist. Consequence: the PII
  scrubber gets no `known_person_names` (ruling #4) and retrieval gets no
  entity filters from memory.
- **No `MemoryCheckRequest`/`MemoryCheckResult`** shapes, therefore no
  `stale_hint`, no `resolved_entities`, no `session_context`.
- **No watermark/`content_hash` staleness gate.** The brief makes "every source
  watermark still ≥ recorded AND every evidence hash still matches" the
  non-negotiable rule; here staleness is a global version counter plus a TTL.
  `CacheWriteRequest.evidence_hashes` / `source_versions` are not stored.
- **No stampede lock.** Two identical concurrent misses both run the full
  pipeline (the brief's failure table wants one computation to serve both).
- **No negative-caching policy beyond the error rule** (e.g. no per-source
  invalidation: any sync invalidates everything).
- **No `mem.answer_cache` writer.** The Postgres table stays in the schema and is
  read by the dashboard's overview counter, but nothing inserts into it, so that
  count is always 0 — it is not a Redis metric.
- **No free-form write path** — which the brief requires, and which is trivially
  satisfied by not having a write API at all.

---

## 3. How to run

```bash
cd /home/varun/projects/agentic_rag_over_live_data

# offline half: normalization, key derivation, null cache, singleton (no Redis)
.venv/bin/python -m pytest tests/memory -q

.venv/bin/ruff check src/memory tests/memory
.venv/bin/mypy src/memory
```

The five live tests run only when `REDIS_URL` is set **and** reachable; otherwise
they skip with the reason. To run them against a scratch Redis:

```bash
REDIS_URL=redis://localhost:6379/0 .venv/bin/python -m pytest tests/memory -q
```

**Warning:** `test_live_clear_removes_entries_and_keeps_the_version_counter` calls
`clear()`, which deletes every `answer_cache:*` key in that database (entries
only — the version counter is preserved). Point it at a Redis you do not share.

The orchestrator half is covered without Redis by
`tests/orchestrator/test_state_machine.py` (`FakeAnswerCache`), no external
service needed.

---

## 4. Tests

Offline (`16 passed`): question normalization (4 parametrised cases) and its
idempotence · the key hash ignores case/whitespace but separates user and version
· the key is exactly `prefix + 64 hex chars` and contains no question text or
user id · `NullAnswerCache` never serves or stores · `get_answer_cache` is `None`
without a URL, `None` when disabled, built once otherwise · `build_answer_cache`
refuses without a URL · `invalidate_answer_cache` returns `None` with no cache,
bumps a configured one, swallows a failing bump (one ERROR log) and survives a
cache that cannot be constructed (also logged at ERROR).

Live (`5 skipped` without `REDIS_URL`): put/get round-trip with a TTL and
normalized-variant read · a version bump makes previous entries unreachable while
the old key survives to expire · `clear()` removes entries and leaves the version
counter alone · keys and values carry no question text · the version is stable
between bumps.

Orchestrator cache tests (4, in `tests/orchestrator/test_state_machine.py`): a
hit skips every stage and emits `memory.cache_hit` without the question text ·
an answered response is cached per user + version with the default TTL, and a
bumped version writes a new entry · degraded and `reason="error"` outcomes are
never cached while `insufficient_evidence` is · a dead cache (version, read and
write all failing) is transparent — the pipeline runs, `cached=false`, and a
read/write-only failure behaves the same.

---

## 5. Deviations from the brief

1. **Redis replaces `mem.answer_cache` at the hot path** — ruling #14. Rationale
   in the ruling: a hit must not contend with retrieval on the same Postgres, TTL
   is native, and the cache must be optional infrastructure. The table stays in
   the schema, unused. New dependency `redis>=5.0` (the first since M0).
2. **Per-user hashed keys instead of "question + entity fingerprint"** — ruling
   #14 makes per-user keying a mandate, not an optimisation: retrieval is
   ACL-scoped, so a question-only key would serve one user's evidence to another.
   The entity fingerprint is impossible anyway: there is no registry (Tier 3).
3. **Staleness is a version counter + TTL, not per-source watermarks and chunk
   hashes.** Any sync/backfill/seed write bumps one global version, making every
   entry unreachable; entries then expire on their TTL. Coarser than the brief
   (one write invalidates all users and all questions) but correct in the failure
   direction: it never serves an answer computed from older data.
4. **The cache stores an `AskResponse`, not a `CachedAnswer` with evidence
   hashes.** The frozen HTTP contract is the only thing worth caching; nothing
   downstream re-validates it.
5. **`insufficient_evidence` refusals are cached; `error` and degraded outcomes
   are not.** The brief only says a blocked answer is never cached. A refusal is
   deterministic for the same question + evidence, so caching it is the same
   decision as caching an answer; a transient failure is not.
6. **`invalidate_answer_cache()` exists instead of `remember()`.** The write path
   is "the orchestrator stores what it just produced"; ingestion code only needs
   to invalidate.
7. **`memory.cache_hit` is emitted by the orchestrator on behalf of the memory
   component** (`agent="memory"`). The name matches what
   `evals.trace_report.aggregate_stats` already counts; keep it.

---

## 6. Uncertainty / risks

- **No live Redis run in this round.** Everything above about Redis behaviour
  comes from reading `answer_cache.py` and from tests that skip. The five live
  tests have not been executed here, so SCAN batching, `clear()` counts, TTLs and
  `INCR` semantics are reasoned-about, not observed.
- **Mid-request outage is only covered by fakes.** `RedisAnswerCache._warn_once`,
  the 2 s socket timeouts and the "fail open" mapping are exercised with a
  `FakeAnswerCache` that raises; no test kills a real connection mid-request. If
  the socket timeout is exceeded, the request pays up to 2 s (read) — not the
  orchestrator budget, but not free.
- **A scrub failure is cached as `guardrail_blocked` for up to 15 minutes.** The
  orchestrator refuses with `degraded=False` on any scrub exception, so
  `store_allowed()` accepts it. A transient analyzer failure therefore pins the
  refusal for the TTL even after the analyzer recovers. `clear`/invalidate fixes
  it; nothing else does.
- **The cache key ignores code and configuration.** Only question | user | data
  version are hashed. Changing `ORCHESTRATOR_TOP_K`, the extraction/synthesis
  prompts, or the model does not invalidate anything — old answers live until the
  TTL or the next data write. There is no cache-version constant tied to the
  prompt.
- **One namespace, one counter.** `CACHE_PREFIX` is a fixed string with no
  deployment or environment component, and `VERSION_KEY` is shared. Two
  deployments pointing at the same Redis database would share entries and
  invalidate each other (and, if their `AskResponse` schemas diverge, silently
  discard each other's writes as "schema drift").
- **`normalize_question` is `lower()` + whitespace only.** No Unicode
  normalisation and no punctuation stripping, so NFC vs NFD spellings or a
  trailing `?` produce different entries (a miss, never a wrong hit).
- **`clear()` is a SCAN loop.** Entries written during the scan can be missed;
  the accompanying `bump_version()` is what actually guarantees they are
  unreachable. The admin route does both, in that order.
- **A Redis outage at startup is silent-ish.** `Redis.from_url` does not connect
  eagerly, so the first request logs one warning and then runs cache-less;
  `REDIS_URL` set-but-wrong looks like a working cache until someone reads the
  log or the `cached` flags. `invalidate_answer_cache` catches that failure too,
  which also means a genuine programming error inside the cache layer (bad
  settings object, protocol violation) is swallowed as "stale until TTL" — the
  blanket `except Exception` is deliberate policy, not oversight.
- **The dashboard's "answer cache" counter is the Postgres table**, not Redis
  (`SELECT count(*) FROM mem.answer_cache`), so it reads 0 even with a populated
  cache.
- **The module docstring is stale on invalidation.** It says "Nothing bumps it
  automatically yet — an ingest/sync caller does", while `connectors.service.sync`,
  `/admin/backfill` and `/admin/seed` all call `invalidate_answer_cache()` today
  (§1). `.env.example` repeats the same stale sentence. The code is right; the
  comments were not updated.
- **Not verified:** live Redis of any version (the module targets `redis>=5.0`),
  concurrent `/ask` requests sharing one cache, eviction under `maxmemory`
  policy, and cache behaviour across a Redis restart (the version counter is not
  persisted by this code — it lives in Redis, so a `FLUSHALL` resets it to 0).

---

## 7. Verified gates (real output, 2026-11-18)

```
$ .venv/bin/python -m pytest tests/memory -q
16 passed, 5 skipped, 1 warning in 0.03s
    # REDIS_URL was unset in this environment; the 5 skips say so:
    #   "REDIS_URL is not set — skipping live answer-cache tests"

$ .venv/bin/ruff check src/memory tests/memory
All checks passed!

$ .venv/bin/mypy src/memory
Success: no issues found in 2 source files
```

Repo-wide at the same moment, `REDIS_URL` unset:
`.venv/bin/python -m pytest -q` → **486 passed, 64 skipped** (the 64 include the
5 live cache tests and the database-backed suites).

The `StarletteDeprecationWarning` from `fastapi.testclient` is the one warning
everywhere in this repo (pre-existing).

---

## 8. Files

Read for this hand-off (no code changed):

```
src/memory/__init__.py              docstring only
src/memory/answer_cache.py          the whole Tier-2 cache
tests/memory/test_answer_cache.py   offline + live Redis tests
src/orchestrator/run.py             read/write wiring, _CacheContext, memory.cache_hit
src/connectors/service.py           invalidation after a committed sync
src/api/admin.py                    /admin/cache/invalidate, backfill + seed invalidation
src/config.py                       redis_url, answer_cache_ttl_s, answer_cache_enabled
src/db/models.py                    mem.answer_cache / mem.entity_registry / mem.session_turns (unused)
```

Nothing outside `src/memory/HANDOFF.md` and the shared docs listed in the task
was edited.
