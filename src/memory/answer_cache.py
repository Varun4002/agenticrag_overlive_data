"""Answer cache (memory Tier 2): a repeat question from the same user skips the pipeline.

Four properties make this safe to put in front of the pipeline:

* **Keys are per user.** Retrieval enforces ``acl.users`` in SQL, so two users
  asking the identical question can legitimately get different evidence. The
  key is ``sha256(normalized question | user_id | data version)`` — never the
  question alone, which would hand one user's answer to another.
* **No plaintext question leaves the process.** Questions carry names and other
  personal data, and Redis is readable by whoever operates the machine. Only
  the hash goes in the key; only the scrubbed, cited :class:`AskResponse` goes
  in the value.
* **Per data version.** ``answer_cache:version`` is a monotonic counter:
  bumping it makes every existing entry unreachable and they then expire on
  their TTL. Nothing bumps it automatically yet — an ingest/sync caller does.
* **Optional by construction.** ``REDIS_URL`` unset (or Redis unreachable)
  means "no cache": the pipeline runs exactly as it does without one. Every
  failure here is a cache miss, never a request failure — a cache is a latency
  optimisation and must not become an availability dependency.

Not implemented here (see ``docs/07-memory-design.md``): the per-source
watermark check, the answer-cache stampede lock, session turns, the entity
registry.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol

from redis.asyncio import Redis

from config import Settings, get_settings
from orchestrator.contracts import AskResponse

__all__ = [
    "CACHE_PREFIX",
    "VERSION_KEY",
    "AnswerCacheProtocol",
    "NullAnswerCache",
    "RedisAnswerCache",
    "aclose_answer_cache",
    "build_answer_cache",
    "cache_key",
    "cache_key_hash",
    "get_answer_cache",
    "invalidate_answer_cache",
    "normalize_question",
    "reset_answer_cache",
]

logger = logging.getLogger(__name__)

#: Every key this module owns starts here, so ``clear()`` can delete exactly
#: this namespace instead of flushing a database it shares with other users.
CACHE_PREFIX = "answer_cache:"

#: Monotonic data version. Deliberately inside the prefix but exempt from
#: ``clear()``: it must never move backwards.
VERSION_KEY = f"{CACHE_PREFIX}version"

_VERSION_DEFAULT = 0

#: Socket ceilings for the client. A dead Redis costs one request two seconds,
#: never the orchestrator budget.
_SOCKET_TIMEOUT_S = 2.0


def normalize_question(question: str) -> str:
    """Lowercase, collapse every whitespace run to one space, and strip.

    Deterministic and locale-free, so ``"When is  the cutover?"`` and
    ``"when is the cutover?"`` are the same cache entry. Kept in one place
    because both the key and the tests' fakes must agree on it.
    """
    return " ".join(question.lower().split())


def cache_key_hash(question: str, user_id: str, version: int) -> str:
    """sha256 hex over ``normalized question | user_id | data version``.

    The only form in which a question may reach Redis (or a trace payload).
    """
    material = f"{normalize_question(question)}|{user_id}|{version}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def cache_key(question: str, user_id: str, version: int) -> str:
    """The Redis key holding one user's answer to one question at one version."""
    return f"{CACHE_PREFIX}{cache_key_hash(question, user_id, version)}"


class AnswerCacheProtocol(Protocol):
    """What the orchestrator needs from an answer cache.

    Reads and writes fail open: ``get`` returns ``None`` when nothing can be
    served (miss, outage, unreadable value) and ``put`` drops what it cannot
    store. ``version`` may raise when the backend is unreachable — callers must
    then run without a cache rather than serve an entry whose data version is
    unknown. ``bump_version`` and ``clear`` are invalidation: they raise,
    because a silently dropped invalidation would serve stale answers.
    """

    async def get(self, question: str, user_id: str, version: int) -> AskResponse | None:
        """Return the cached response, or ``None``."""
        ...

    async def put(
        self,
        question: str,
        user_id: str,
        version: int,
        response: AskResponse,
        ttl_s: int,
    ) -> None:
        """Store ``response`` for ``ttl_s`` seconds."""
        ...

    async def version(self) -> int:
        """Current data version."""
        ...

    async def bump_version(self) -> int:
        """Advance the data version and return the new one."""
        ...

    async def clear(self) -> int:
        """Delete every entry; return how many keys were removed."""
        ...

    async def aclose(self) -> None:
        """Release backend resources."""
        ...


class NullAnswerCache:
    """Cache that never hits and never stores.

    Used when caching is deliberately off for an injected orchestrator, and by
    tests that must not touch the environment's Redis.
    """

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
        return _VERSION_DEFAULT

    async def bump_version(self) -> int:
        return _VERSION_DEFAULT

    async def clear(self) -> int:
        return 0

    async def aclose(self) -> None:
        return None


class RedisAnswerCache:
    """One serialized :class:`AskResponse` per key in Redis."""

    def __init__(self, url: str) -> None:
        self._redis = Redis.from_url(
            url,
            decode_responses=True,
            socket_timeout=_SOCKET_TIMEOUT_S,
            socket_connect_timeout=_SOCKET_TIMEOUT_S,
        )
        self._warned = False

    def _warn_once(self, operation: str, exc: Exception) -> None:
        """Report a backend failure once per process: an outage is not news after that."""
        if self._warned:
            return
        self._warned = True
        logger.warning(
            "answer cache unavailable during %s (%s: %s): failing open, requests run "
            "without a cache",
            operation,
            type(exc).__name__,
            exc,
        )

    async def get(self, question: str, user_id: str, version: int) -> AskResponse | None:
        """Return the cached response, or ``None`` for any reason at all."""
        key = cache_key(question, user_id, version)
        try:
            raw = await self._redis.get(key)
        except Exception as exc:
            # Fail open: a cache read must never be why a request fails.
            self._warn_once("get", exc)
            return None
        if raw is None:
            return None
        try:
            return AskResponse.model_validate_json(raw)
        except ValueError:
            # Written by an older schema: treat it as a miss and let the next
            # answer overwrite the key.
            logger.warning("discarding an unreadable answer cache entry (schema drift)")
            return None

    async def put(
        self,
        question: str,
        user_id: str,
        version: int,
        response: AskResponse,
        ttl_s: int,
    ) -> None:
        """Store ``response`` under its key with a TTL; never raises."""
        key = cache_key(question, user_id, version)
        try:
            await self._redis.set(key, response.model_dump_json(), ex=ttl_s)
        except Exception as exc:
            # Fail open: the answer is already produced; a lost write only
            # costs the next request a full pipeline run.
            self._warn_once("put", exc)

    async def version(self) -> int:
        """Current data version; raises when the backend is unreachable."""
        try:
            raw = await self._redis.get(VERSION_KEY)
        except Exception as exc:
            self._warn_once("version", exc)
            raise
        return _VERSION_DEFAULT if raw is None else int(raw)

    async def bump_version(self) -> int:
        """Advance the data version, making every existing entry unreachable.

        Raises on failure: a dropped invalidation would serve answers computed
        from data that has since changed.
        """
        return int(await self._redis.incr(VERSION_KEY))

    async def clear(self) -> int:
        """Delete every cached entry and return the number of keys removed.

        SCANs this module's prefix rather than ``FLUSHDB`` (the Redis database
        is shared) and skips the version counter, which must stay monotonic.
        """
        removed = 0
        batch: list[str] = []
        async for key in self._redis.scan_iter(match=f"{CACHE_PREFIX}*", count=500):
            if key == VERSION_KEY:
                continue
            batch.append(str(key))
            if len(batch) >= 500:
                removed += int(await self._redis.delete(*batch))
                batch.clear()
        if batch:
            removed += int(await self._redis.delete(*batch))
        return removed

    async def aclose(self) -> None:
        """Release the client's connections; never raises."""
        try:
            await self._redis.aclose()
        except Exception as exc:
            logger.debug("closing the answer cache client failed: %s", exc)


_cache: AnswerCacheProtocol | None = None
_resolved = False


def build_answer_cache(settings: Settings) -> RedisAnswerCache:
    """Construct a cache from explicit settings (tests, scripts)."""
    if not settings.redis_url:
        raise ValueError("REDIS_URL is not set")
    return RedisAnswerCache(settings.redis_url)


def get_answer_cache(settings: Settings | None = None) -> AnswerCacheProtocol | None:
    """Return the process-wide answer cache, or ``None`` when caching is off.

    Built once from settings, mirroring ``db.session.get_engine``, so the
    "not configured" warning is logged once per process. Callers that get
    ``None`` must run the pipeline exactly as they would without a cache.
    """
    global _cache, _resolved
    if not _resolved:
        cfg = settings or get_settings()
        if not cfg.answer_cache_enabled:
            logger.info("ANSWER_CACHE_ENABLED is false: answer cache disabled")
        elif not cfg.redis_url:
            logger.warning("REDIS_URL is not set: answer cache disabled")
        else:
            _cache = build_answer_cache(cfg)
        _resolved = True
    return _cache


async def aclose_answer_cache() -> None:
    """Close the process-wide cache (app shutdown); a no-op when unconfigured."""
    global _cache, _resolved
    cache, _cache = _cache, None
    _resolved = False
    if cache is not None:
        await cache.aclose()


def reset_answer_cache() -> None:
    """Drop the singleton without awaiting a close (tests only)."""
    global _cache, _resolved
    _cache = None
    _resolved = False


async def invalidate_answer_cache(settings: Settings | None = None) -> int | None:
    """Bump the data version after any ingestion write (sync/backfill/seed).

    Returns the new version, or ``None`` when no cache is configured. A failed
    bump is logged as an error and also returns ``None``: the ingestion write
    itself already succeeded, so the worst case is that cached answers stay
    stale until their TTL expires — visible in the log, never silently.
    """
    try:
        cache = get_answer_cache(settings)
        if cache is None:
            return None
        return await cache.bump_version()
    except Exception as exc:  # policy: invalidation must never fail a finished sync
        logger.error(
            "answer-cache invalidation failed: %s — cached answers may be stale "
            "until TTL; clear the cache or fix Redis",
            exc,
        )
        return None
