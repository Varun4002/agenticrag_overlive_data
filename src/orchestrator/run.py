"""The orchestrator: one question in, one cited answer or a refusal out.

This is the deterministic state machine from ``docs/agents/00-orchestrator.md``,
scoped to what exists today::

    INTAKE → (answer cache hit) ────────────────────────────→ RESPOND
    INTAKE → PLAN → RETRIEVE → SCRUB → VERIFY → SYNTHESIZE → RESPOND
                                    ↘ REFUSE (scrub / verify / failure)

What it deliberately does *not* do yet, and says so in every trace:

* ``GUARD_INPUT`` / ``GUARD_OUTPUT`` — the guardrail agent (M3) is not written.
  The evidence validator already drops injection-shaped claims, but that is a
  claim filter, not an input/output guardrail.
* ``MEMORY_CHECK`` — only Tier 2 of the memory agent exists: the Redis answer
  cache (``src/memory/answer_cache.py``), which serves a repeat question from
  the same user. Session turns and the entity registry are not written.
* ``FETCH_LIVE`` — a live targeted fetch is useless until fetched documents are
  indexed in the same request (M4). The plan records ``live_fetch=False``.
* The LLM planner — the plan is deterministic (hybrid retrieval, all sources,
  top-k from settings). An LLM planner has no measurable job yet: there are no
  sources to choose between and no live fetch to decide on.

Ordering rule that makes the chain safe: the scrubber is the only path from
retrieval to an LLM, so ``VERIFY`` never sees an ``EvidenceChunk``. Scrubbing is
fail-closed: if it raises, the request is refused, never answered from raw text.

Every stage emits ``audit.events`` with the request's ``trace_id``, including
the stages it skipped, so a trace explains absences as well as actions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from config import Settings, get_settings
from db import session as db_session
from db.tracing import EventSink, Tracer, make_sink, new_trace_id
from llm.errors import LLMError
from memory.answer_cache import AnswerCacheProtocol, cache_key_hash, get_answer_cache
from orchestrator.contracts import AskRequest, AskResponse, Refusal
from orchestrator.pipeline import (
    EvidencePack,
    ScrubbedPack,
    SourceName,
    SynthesisResult,
    VerifiedClaims,
)
from retrieval.errors import RetrievalConfigError
from retrieval.search import RetrieveRequest

__all__ = [
    "AGENT",
    "Orchestrator",
    "Plan",
    "Stages",
    "default_stages",
]

logger = logging.getLogger(__name__)

AGENT = "orchestrator"

#: The memory component owns the answer cache, so its hits are its events.
MEMORY_AGENT = "memory"

#: Retrieval failures that mean "this deployment cannot answer", not "no answer".
CONFIG_ERRORS: tuple[type[Exception], ...] = (RetrievalConfigError,)


#: The literal set from ``orchestrator.contracts.Refusal``.
RefusalReason = Literal["insufficient_evidence", "out_of_scope", "guardrail_blocked", "error"]


#: Stage signatures. Callable[..., ...] because every collaborator takes its own
#: keyword-only injection points (session factory, embedder, client, ...); the
#: real functions are bound in :func:`default_stages`.
RetrieveFn = Callable[..., Awaitable[EvidencePack]]
ScrubFn = Callable[..., Awaitable[ScrubbedPack]]
VerifyFn = Callable[..., Awaitable[VerifiedClaims]]
SynthesizeFn = Callable[..., Awaitable[SynthesisResult]]


@dataclass(frozen=True, slots=True)
class Stages:
    """The four pipeline stages, injectable so tests need no database or LLM."""

    retrieve: RetrieveFn
    scrub: ScrubFn
    verify: VerifyFn
    synthesize: SynthesizeFn


def default_stages() -> Stages:
    """Bind the real components.

    Imported lazily: ``pii`` pulls Presidio + spaCy, which must not slow down
    ``/health`` or app startup, and a broken analyzer should fail one request
    (fail-closed) instead of preventing the process from booting.
    """
    from evidence.service import verify
    from pii.scrubber import scrub
    from retrieval.search import retrieve
    from synthesis.synthesizer import synthesize

    return Stages(retrieve=retrieve, scrub=scrub, verify=verify, synthesize=synthesize)


@dataclass(frozen=True, slots=True)
class Plan:
    """What the request will do. Deterministic until an LLM planner earns its job."""

    sources: tuple[SourceName, ...] | None
    top_k: int
    freshness_window_s: int
    live_fetch: bool
    skipped_stages: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "sources": list(self.sources) if self.sources else "all",
            "top_k": self.top_k,
            "freshness_window_s": self.freshness_window_s,
            "live_fetch": self.live_fetch,
            "skipped_stages": list(self.skipped_stages),
            "planner": "deterministic",
        }


@dataclass(frozen=True, slots=True)
class _CacheContext:
    """What one request needs to read and write the answer cache.

    ``version is None`` means "caching is off for this request" — no Redis, a
    disabled cache, or a version that could not be read. An entry whose data
    version is unknown must not be served or written.
    """

    cache: AnswerCacheProtocol
    version: int | None
    question: str
    user_id: str

    def key_hash(self) -> str:
        """sha256 of question | user | version — the only form a trace may carry."""
        assert self.version is not None, "a cache key needs a known data version"
        return cache_key_hash(self.question, self.user_id, self.version)

    def store_allowed(self, response: AskResponse) -> bool:
        """Cache answered responses and non-error refusals only.

        Never a transient failure (``reason == "error"``: the LLM may recover
        on the next attempt) and never a degraded answer (retrieval fell back,
        so a retry can legitimately do better).
        """
        if self.version is None or response.degraded:
            return False
        return response.refusal is None or response.refusal.reason != "error"


class Orchestrator:
    """Drives one question through retrieve → scrub → verify → synthesize."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        sink: EventSink | None = None,
        stages: Stages | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        cache: AnswerCacheProtocol | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._stages = stages or default_stages()
        self._sink = sink
        self._session_factory = session_factory
        # ``None`` means "resolve from settings", which is ``None`` too when
        # REDIS_URL is unset or ANSWER_CACHE_ENABLED is false.
        self._cache = cache

    async def ask(self, req: AskRequest, *, trace_id: str | None = None) -> AskResponse:
        """Answer ``req`` or refuse; never raises for an answerable failure.

        Raises :class:`~retrieval.errors.RetrievalConfigError` when the
        deployment is misconfigured — that is a 503, not a refusal, because an
        empty refusal would look like "no evidence found".
        """
        factory = self._session_factory or db_session.get_session_factory()
        sink = self._sink or make_sink(factory)
        tracer = Tracer(sink, trace_id or new_trace_id())
        started = time.perf_counter()
        deadline = started + self._settings.orchestrator_budget_s
        question = req.question.strip()

        await tracer.emit(
            AGENT,
            "orchestrator.intake",
            {
                "question": question,
                "question_chars": len(question),
                "session_id": req.session_id,
                "user_id": req.user_id,
            },
        )
        plan = self._plan()
        await tracer.emit(AGENT, "orchestrator.plan", plan.as_payload())

        if not question:
            return await self._refuse(
                tracer,
                question,
                "out_of_scope",
                "the question is empty",
                degraded=False,
                started=started,
            )

        # ---- MEMORY: answer cache ----------------------------------------
        # Per user (retrieval is ACL-scoped, so the same question can legitimately
        # have different answers) and per data version. A hit returns before any
        # retrieval, scrubbing, verification or synthesis runs.
        ctx = await self._cache_context(question, req.user_id)
        if ctx is not None:
            hit = await self._cache_get(ctx)
            if hit is not None:
                return await self._answer_from_cache(tracer, ctx, hit, started)

        # ---- RETRIEVE ----------------------------------------------------
        try:
            pack = await self._stages.retrieve(
                RetrieveRequest(
                    query_text=question,
                    sources=list(plan.sources) if plan.sources else None,
                    freshness_window=timedelta(seconds=plan.freshness_window_s),
                    top_k=plan.top_k,
                ),
                user_id=req.user_id,
                trace_id=tracer.trace_id,
                session_factory=factory,
            )
        except CONFIG_ERRORS:
            await tracer.emit(
                AGENT, "orchestrator.failed", {"stage": "retrieve", "reason": "configuration"}
            )
            raise
        except Exception as exc:
            logger.exception("retrieval failed for trace %s", tracer.trace_id)
            return await self._refuse(
                tracer,
                question,
                "error",
                f"evidence retrieval failed ({type(exc).__name__})",
                degraded=True,
                started=started,
                cache_ctx=ctx,
            )

        # ---- SCRUB (hard gate) -------------------------------------------
        try:
            scrubbed = await self._stages.scrub(
                pack,
                trace_id=tracer.trace_id,
                session_factory=factory,
            )
        except Exception as exc:
            logger.warning(
                "PII scrubbing failed for trace %s: %s", tracer.trace_id, type(exc).__name__
            )
            return await self._refuse(
                tracer,
                question,
                "guardrail_blocked",
                "evidence could not be scrubbed of personal data, so it was not sent to a model",
                degraded=False,
                started=started,
                cache_ctx=ctx,
            )

        # ---- Budget check before the two LLM steps -----------------------
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return await self._refuse(
                tracer,
                question,
                "error",
                "request budget exhausted before claim verification",
                degraded=True,
                started=started,
                cache_ctx=ctx,
            )

        # ---- VERIFY -------------------------------------------------------
        try:
            async with asyncio.timeout(remaining):
                verified = await self._stages.verify(
                    question, scrubbed, trace_id=tracer.trace_id, sink=sink
                )
        except TimeoutError:
            return await self._refuse(
                tracer,
                question,
                "error",
                "request budget exhausted during claim verification",
                degraded=True,
                started=started,
                cache_ctx=ctx,
            )
        except LLMError as exc:
            await tracer.emit(
                AGENT,
                "orchestrator.failed",
                {"stage": "verify", "error_type": type(exc).__name__, "retryable": exc.retryable},
            )
            return await self._refuse(
                tracer,
                question,
                "error",
                f"claim verification failed ({type(exc).__name__})",
                degraded=False,
                started=started,
                cache_ctx=ctx,
            )
        except Exception as exc:
            logger.exception("claim verification crashed for trace %s", tracer.trace_id)
            return await self._refuse(
                tracer,
                question,
                "error",
                f"claim verification failed ({type(exc).__name__})",
                degraded=False,
                started=started,
                cache_ctx=ctx,
            )

        # ---- SYNTHESIZE (refuses deterministically when it cannot answer) --
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return await self._refuse(
                tracer,
                question,
                "error",
                "request budget exhausted before synthesis",
                degraded=True,
                started=started,
                cache_ctx=ctx,
            )
        try:
            async with asyncio.timeout(remaining):
                result = await self._stages.synthesize(
                    question,
                    verified,
                    scrubbed,
                    trace_id=tracer.trace_id,
                    sink=sink,
                )
        except TimeoutError:
            return await self._refuse(
                tracer,
                question,
                "error",
                "request budget exhausted during synthesis",
                degraded=True,
                started=started,
                cache_ctx=ctx,
            )
        except LLMError as exc:
            await tracer.emit(
                AGENT,
                "orchestrator.failed",
                {
                    "stage": "synthesize",
                    "error_type": type(exc).__name__,
                    "retryable": exc.retryable,
                },
            )
            return await self._refuse(
                tracer,
                question,
                "error",
                f"answer synthesis failed ({type(exc).__name__})",
                degraded=False,
                started=started,
                cache_ctx=ctx,
            )
        except Exception as exc:
            logger.exception("synthesis crashed for trace %s", tracer.trace_id)
            return await self._refuse(
                tracer,
                question,
                "error",
                f"answer synthesis failed ({type(exc).__name__})",
                degraded=False,
                started=started,
                cache_ctx=ctx,
            )

        if result.refused:
            detail = (
                verified.insufficiency_reason
                or "the draft answer could not be tied back to the evidence"
            )
            return await self._refuse(
                tracer,
                question,
                "insufficient_evidence",
                detail,
                degraded=bool(pack.degraded_reason),
                started=started,
                answer=result.answer,
                citations_count=0,
                evidence_chunks=len(pack.chunks),
                claims=len(verified.claims),
                cache_ctx=ctx,
            )

        response = AskResponse(
            answer=result.answer,
            citations=result.citations,
            refusal=None,
            trace_id=tracer.trace_id,
            degraded=bool(pack.degraded_reason),
        )
        cached = await self._cache_store(ctx, response) if ctx is not None else False
        await tracer.emit(
            AGENT,
            "orchestrator.completed",
            {
                "status": "answered",
                "chunks": len(pack.chunks),
                "redactions": scrubbed.redaction_count,
                "claims": len(verified.claims),
                "sufficiency": verified.sufficiency,
                "citations": len(result.citations),
                "degraded": response.degraded,
                "degraded_reason": pack.degraded_reason,
                "cached": cached,
            },
            latency_ms=_elapsed_ms(started),
        )
        return response

    # -- internals ---------------------------------------------------------

    def _plan(self) -> Plan:
        return Plan(
            sources=None,
            top_k=self._settings.orchestrator_top_k,
            freshness_window_s=self._settings.sync_stale_after_s,
            live_fetch=False,
            skipped_stages=("guard_input", "memory_check", "guard_output"),
        )

    async def _refuse(
        self,
        tracer: Tracer,
        question: str,
        reason: RefusalReason,
        detail: str,
        *,
        degraded: bool,
        started: float,
        answer: str | None = None,
        citations_count: int = 0,
        evidence_chunks: int = 0,
        claims: int = 0,
        cache_ctx: _CacheContext | None = None,
    ) -> AskResponse:
        from synthesis.synthesizer import refusal_answer

        text = answer or refusal_answer(detail)
        response = AskResponse(
            answer=text,
            citations=[],
            refusal=Refusal(reason=reason, detail=detail),
            trace_id=tracer.trace_id,
            degraded=degraded,
        )
        # A refusal is a real outcome (the same question gets the same answer
        # from the same evidence), so non-error refusals are cached; the
        # policy lives in ``_CacheContext.store_allowed``.
        cached = await self._cache_store(cache_ctx, response) if cache_ctx is not None else False
        await tracer.emit(
            AGENT,
            "orchestrator.refused",
            {
                "reason": reason,
                "detail": detail[:300],
                "degraded": degraded,
                "chunks": evidence_chunks,
                "claims": claims,
                "citations": citations_count,
                "question": question,
                "cached": cached,
            },
            latency_ms=_elapsed_ms(started),
        )
        return response

    # -- answer cache ------------------------------------------------------

    async def _cache_context(self, question: str, user_id: str) -> _CacheContext | None:
        """Resolve the cache and its data version, or ``None`` when caching is off."""
        cache = self._cache if self._cache is not None else get_answer_cache(self._settings)
        if cache is None:
            return None
        try:
            version = await cache.version()
        except Exception as exc:
            # An unknown data version cannot be trusted: run the pipeline and
            # skip the write, exactly as if no cache were configured.
            logger.debug("answer cache version unavailable: %s", type(exc).__name__)
            version = None
        return _CacheContext(cache=cache, version=version, question=question, user_id=user_id)

    async def _cache_get(self, ctx: _CacheContext) -> AskResponse | None:
        """Read the cache; any failure is a miss.

        Bounded by the client's 2 s socket timeouts rather than the request
        budget: a cache must not be able to consume the time a real answer needs.
        """
        if ctx.version is None:
            return None
        try:
            return await ctx.cache.get(ctx.question, ctx.user_id, ctx.version)
        except Exception as exc:
            logger.debug("answer cache read failed: %s", type(exc).__name__)
            return None

    async def _cache_store(self, ctx: _CacheContext | None, response: AskResponse) -> bool:
        """Store a cacheable response; ``False`` when skipped or the write failed."""
        if ctx is None:
            return False
        version = ctx.version
        if version is None or not ctx.store_allowed(response):
            return False
        try:
            await ctx.cache.put(
                ctx.question,
                ctx.user_id,
                version,
                response,
                self._settings.answer_cache_ttl_s,
            )
        except Exception as exc:
            logger.debug("answer cache write failed: %s", type(exc).__name__)
            return False
        return True

    async def _answer_from_cache(
        self,
        tracer: Tracer,
        ctx: _CacheContext,
        cached: AskResponse,
        started: float,
    ) -> AskResponse:
        """Serve a cached response under this request's trace id."""
        await tracer.emit(
            MEMORY_AGENT,
            "memory.cache_hit",
            {
                "original_trace_id": cached.trace_id,
                "version": ctx.version,
                "cache_key_hash": ctx.key_hash(),
            },
        )
        await tracer.emit(
            AGENT,
            "orchestrator.completed",
            {
                "status": "cache_hit",
                "answer_chars": len(cached.answer),
                "citations": len(cached.citations),
                "refusal_reason": cached.refusal.reason if cached.refusal else None,
                "degraded": cached.degraded,
                "cached": True,
            },
            latency_ms=_elapsed_ms(started),
        )
        # The response carries the live trace; the trace that produced it stays
        # in the event payload above. A copy leaves the cached object untouched
        # for the next hit.
        return cached.model_copy(update={"trace_id": tracer.trace_id})


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
