"""Answer-cache tests: key correctness offline, Redis behaviour when one is available.

The unit half needs nothing but Python: normalization, key derivation (per user,
per data version, no plaintext), the null cache, and the settings-driven
singleton. The live half is skipped — never failed — unless ``REDIS_URL`` is set
*and* reachable, the same contract ``tests/db/test_migrations.py`` uses for
Postgres.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import AsyncIterator

import pytest
from redis.asyncio import Redis

from config import Settings
from memory import answer_cache
from memory.answer_cache import (
    CACHE_PREFIX,
    VERSION_KEY,
    NullAnswerCache,
    RedisAnswerCache,
    build_answer_cache,
    cache_key,
    cache_key_hash,
    get_answer_cache,
    invalidate_answer_cache,
    normalize_question,
    reset_answer_cache,
)
from orchestrator.contracts import AskResponse, Citation

QUESTION = "What did Grace Adeyemi say about the PHX-150 cutover?"
USER = "priya@example.com"


def redis_url() -> str | None:
    """The configured Redis URL, or None when the environment has none."""
    return os.environ.get("REDIS_URL") or None


def make_response(trace_id: str = "trace-original") -> AskResponse:
    return AskResponse(
        answer="The cutover is 2026-11-27 [1].",
        citations=[
            Citation(
                chunk_id="c1",
                document_id="doc-1",
                source="jira",
                title="PHX-150 — Cut over billing webhooks",
                quote="The cutover moves to 2026-11-27.",
            )
        ],
        trace_id=trace_id,
    )


# ---- normalization and keys (no Redis) -----------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("When is the PHX-150 cutover?", "when is the phx-150 cutover?"),
        ("  WHEN\tis  the\ncutover?  ", "when is the cutover?"),
        ("already  normal", "already normal"),
        ("\r\n What? \t", "what?"),
    ],
)
def test_normalize_question_collapses_case_and_whitespace(raw: str, expected: str) -> None:
    assert normalize_question(raw) == expected


def test_normalization_is_idempotent() -> None:
    once = normalize_question("  When IS  the cutover? ")
    assert normalize_question(once) == once


def test_key_hash_ignores_case_and_whitespace_but_not_user_or_version() -> None:
    base = cache_key_hash("When is the cutover?", USER, 0)

    assert cache_key_hash("  when is  the CUTOVER? ", USER, 0) == base
    assert cache_key_hash("When is the cutover?", "sam@example.com", 0) != base
    assert cache_key_hash("When is the cutover?", USER, 1) != base
    assert cache_key_hash("When is the cutover?", USER, 0) == base


def test_key_contains_only_the_prefix_and_a_sha256_digest() -> None:
    key = cache_key(QUESTION, USER, 3)

    assert key.startswith(CACHE_PREFIX)
    digest = key[len(CACHE_PREFIX) :]
    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)
    # No plaintext question, no user id: Redis is readable by the operator.
    assert "grace" not in key.lower()
    assert "cutover" not in key.lower()
    assert "priya" not in key.lower()


async def test_null_cache_never_serves_or_stores() -> None:
    cache = NullAnswerCache()

    await cache.put(QUESTION, USER, 0, make_response(), ttl_s=60)

    assert await cache.get(QUESTION, USER, 0) is None
    assert await cache.version() == 0
    assert await cache.bump_version() == 0
    assert await cache.clear() == 0
    await cache.aclose()


# ---- settings-driven singleton (no Redis connection is made) -------------


def test_cache_is_none_when_redis_url_is_unset() -> None:
    reset_answer_cache()
    try:
        settings = Settings(_env_file=None, redis_url=None)
        assert get_answer_cache(settings) is None
    finally:
        reset_answer_cache()


def test_cache_is_none_when_disabled_even_with_a_url() -> None:
    reset_answer_cache()
    try:
        settings = Settings(
            _env_file=None,
            redis_url="redis://localhost:6379/0",
            answer_cache_enabled=False,
        )
        assert get_answer_cache(settings) is None
    finally:
        reset_answer_cache()


def test_cache_is_built_once_from_settings() -> None:
    reset_answer_cache()
    try:
        settings = Settings(
            _env_file=None,
            redis_url="redis://localhost:6379/0",
            answer_cache_enabled=True,
        )
        first = get_answer_cache(settings)

        assert isinstance(first, RedisAnswerCache)
        assert get_answer_cache() is first
    finally:
        reset_answer_cache()


def test_build_answer_cache_requires_a_url() -> None:
    with pytest.raises(ValueError, match="REDIS_URL"):
        build_answer_cache(Settings(_env_file=None, redis_url=None))


# ---- invalidation after an ingestion write (no Redis) --------------------


class FakeCache:
    """Enough of ``AnswerCacheProtocol`` to observe ``invalidate_answer_cache``."""

    def __init__(self, version: int = 0, *, fail: bool = False) -> None:
        self._version = version
        self.bumps = 0
        self._fail = fail

    async def get(self, question: str, user_id: str, version: int) -> AskResponse | None:
        return None

    async def put(
        self,
        question: str,
        user_id: str,
        version: int,
        response: AskResponse,
        ttl_s: int,
    ) -> None:
        return None

    async def version(self) -> int:
        return self._version

    async def bump_version(self) -> int:
        self.bumps += 1
        if self._fail:
            raise ConnectionError("redis is unreachable")
        self._version += 1
        return self._version

    async def clear(self) -> int:
        return 0

    async def aclose(self) -> None:
        return None


def inject(monkeypatch: pytest.MonkeyPatch, cache: FakeCache) -> None:
    """Install ``cache`` as the resolved module singleton, without touching Redis."""
    monkeypatch.setattr(answer_cache, "_cache", cache)
    monkeypatch.setattr(answer_cache, "_resolved", True)


async def test_invalidate_without_a_configured_cache_returns_none() -> None:
    reset_answer_cache()
    try:
        settings = Settings(_env_file=None, redis_url=None)
        assert await invalidate_answer_cache(settings) is None
    finally:
        reset_answer_cache()


async def test_invalidate_bumps_the_configured_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FakeCache(version=4)
    inject(monkeypatch, cache)

    assert await invalidate_answer_cache() == 5
    assert cache.bumps == 1
    assert await cache.version() == 5


async def test_invalidate_swallows_a_failing_bump_and_logs_it(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A dead Redis must not turn a finished ingestion into a failed one."""
    cache = FakeCache(fail=True)
    inject(monkeypatch, cache)

    with caplog.at_level(logging.ERROR, logger="memory.answer_cache"):
        assert await invalidate_answer_cache() is None

    assert cache.bumps == 1
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "invalidation failed" in errors[0].getMessage()


async def test_invalidate_does_not_raise_on_an_unbuildable_cache(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A malformed ``REDIS_URL`` fails at construction, not at the call site."""

    def explode(settings: Settings | None = None) -> None:
        raise ValueError("Redis URL must specify one of the following schemes")

    monkeypatch.setattr(answer_cache, "get_answer_cache", explode)

    with caplog.at_level(logging.ERROR, logger="memory.answer_cache"):
        assert await invalidate_answer_cache(Settings(_env_file=None)) is None
    assert any(record.levelno == logging.ERROR for record in caplog.records)


# ---- live Redis (skipped unless REDIS_URL is set and reachable) ----------


@pytest.fixture
async def redis_client() -> AsyncIterator[Redis]:
    """A raw client for inspecting keys; skips the test when Redis is absent."""
    url = redis_url()
    if not url:
        pytest.skip("REDIS_URL is not set — skipping live answer-cache tests")
    client = Redis.from_url(
        url,
        decode_responses=True,
        socket_timeout=2,
        socket_connect_timeout=2,
    )
    try:
        await client.ping()
    except Exception as exc:
        await client.aclose()
        pytest.skip(
            f"REDIS_URL is unreachable ({type(exc).__name__}: {exc}) — "
            "skipping live answer-cache tests"
        )
    yield client
    await client.aclose()


@pytest.fixture
async def cache(redis_client: Redis) -> AsyncIterator[RedisAnswerCache]:
    """A cache under test; depends on ``redis_client`` so the skip applies first."""
    url = os.environ["REDIS_URL"]
    built = build_answer_cache(Settings(_env_file=None, redis_url=url))
    yield built
    await built.aclose()


def unique_user() -> str:
    """A per-test identity, so live entries never collide with a real cache."""
    return f"test-{uuid.uuid4()}@example.com"


async def test_live_put_get_roundtrip_sets_a_ttl(
    cache: RedisAnswerCache, redis_client: Redis
) -> None:
    user = unique_user()
    version = await cache.version()
    key = cache_key(QUESTION, user, version)

    try:
        assert await cache.get(QUESTION, user, version) is None
        await cache.put(QUESTION, user, version, make_response(), ttl_s=60)

        stored = await cache.get(QUESTION, user, version)
        assert stored is not None
        assert stored.model_dump() == make_response().model_dump()
        # The question is normalized on both sides of the cache.
        variant = "  what did grace adeyemi SAY about the phx-150 cutover?"
        assert await cache.get(variant, user, version) is not None

        ttl = await redis_client.ttl(key)
        assert 0 < ttl <= 60
    finally:
        await redis_client.delete(key)


async def test_live_version_bump_makes_previous_entries_unreachable(
    cache: RedisAnswerCache, redis_client: Redis
) -> None:
    user = unique_user()
    version = await cache.version()
    key = cache_key(QUESTION, user, version)

    try:
        await cache.put(QUESTION, user, version, make_response(), ttl_s=60)
        assert await cache.get(QUESTION, user, version) is not None

        bumped = await cache.bump_version()

        assert bumped == version + 1
        assert await cache.get(QUESTION, user, bumped) is None
        # The old entry is unreachable, not deleted: it expires on its TTL.
        assert await redis_client.exists(key) == 1
    finally:
        await redis_client.delete(key)


async def test_live_clear_removes_entries_and_keeps_the_version_counter(
    cache: RedisAnswerCache, redis_client: Redis
) -> None:
    """``clear()`` empties the whole namespace, so run this against a test Redis."""
    user = unique_user()
    version = await cache.version()
    first = cache_key(QUESTION, user, version)
    second = cache_key("A different question?", user, version)

    try:
        await cache.put(QUESTION, user, version, make_response(), ttl_s=60)
        await cache.put("A different question?", user, version, make_response(), ttl_s=60)
        assert await redis_client.exists(first, second) == 2

        removed = await cache.clear()

        assert removed >= 2
        assert await redis_client.exists(first, second) == 0
        assert await cache.version() == version, "the version counter must stay monotonic"
    finally:
        await redis_client.delete(first, second)


async def test_live_keys_and_values_carry_no_question_text(
    cache: RedisAnswerCache, redis_client: Redis
) -> None:
    user = unique_user()
    version = await cache.version()
    key = cache_key(QUESTION, user, version)

    try:
        await cache.put(QUESTION, user, version, make_response(), ttl_s=60)

        keys = [scanned async for scanned in redis_client.scan_iter(match=f"{CACHE_PREFIX}*")]
        assert key in keys
        assert len(key) == len(CACHE_PREFIX) + 64, "prefix plus sha256 hex, nothing else"

        value = await redis_client.get(key)
        assert value is not None
        assert "grace" not in value.lower()
        assert "priya" not in value.lower()
        assert QUESTION not in value
        assert "Grace Adeyemi" not in value
    finally:
        await redis_client.delete(key)


async def test_live_version_is_stable_between_bumps(cache: RedisAnswerCache) -> None:
    version = await cache.version()

    assert isinstance(version, int)
    assert await cache.version() == version
    # The counter lives inside the namespace but is exempt from clear().
    assert VERSION_KEY.startswith(CACHE_PREFIX)
