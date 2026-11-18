"""State-machine tests: every downstream agent faked, no database or model.

These assert the things the wiring can get wrong: stage order, what each stage
receives, which failures become refusals vs 503s, and that a skipped stage is
visible in the trace instead of silently implied.
"""

from __future__ import annotations

import pytest
from orch_fakes import (
    FakeAnswerCache,
    FakeStages,
    make_pack,
    make_synthesis,
    make_verified,
)

from config import Settings
from db.tracing import InMemoryEventSink
from llm.errors import LLMAuthError, LLMTimeoutError
from memory.answer_cache import NullAnswerCache, cache_key_hash
from orchestrator.contracts import AskRequest, AskResponse, Citation
from orchestrator.run import Orchestrator
from pii.errors import PiiUnavailableError
from retrieval.errors import RetrievalConfigError


def make_request(
    question: str = "When is the PHX-150 cutover?",
    *,
    user_id: str = "priya@example.com",
) -> AskRequest:
    return AskRequest(question=question, session_id="sess-1", user_id=user_id)


def make_orchestrator(
    stages: FakeStages,
    sink: InMemoryEventSink,
    *,
    budget_s: float = 30.0,
    top_k: int = 12,
    cache: FakeAnswerCache | None = None,
) -> Orchestrator:
    settings = Settings(
        database_url=None,
        orchestrator_budget_s=budget_s,
        orchestrator_top_k=top_k,
    )
    return Orchestrator(
        settings=settings,
        sink=sink,
        stages=stages.as_stages(),
        # A no-op cache unless a test injects one: unit tests must never depend
        # on (or write to) the environment's Redis.
        cache=cache if cache is not None else NullAnswerCache(),
    )


async def test_happy_path_orders_stages_and_returns_citations() -> None:
    fakes = FakeStages()
    sink = InMemoryEventSink()
    orchestrator = make_orchestrator(fakes, sink)

    response = await orchestrator.ask(make_request(), trace_id="trace-1")

    assert fakes.order() == ["retrieve", "scrub", "verify", "synthesize"]
    assert response.answer == "The cutover is 2026-11-27 [1]."
    assert len(response.citations) == 1
    assert response.citations[0].chunk_id == "c1"
    assert response.citations[0].source == "jira"
    assert response.refusal is None
    assert response.degraded is False
    assert response.trace_id == "trace-1"

    # The retriever is scoped to the caller's ACL identity.
    retrieve_args, retrieve_kwargs = fakes.args("retrieve")
    assert retrieve_kwargs["user_id"] == "priya@example.com"
    assert retrieve_kwargs["trace_id"] == "trace-1"
    assert retrieve_args[0].query_text == "When is the PHX-150 cutover?"
    assert retrieve_args[0].top_k == 12

    # The evidence stage sees the scrubbed pack, never the raw one.
    verify_args, verify_kwargs = fakes.args("verify")
    assert verify_args[0] == "When is the PHX-150 cutover?"
    assert verify_args[1] is fakes.scrubbed
    assert verify_kwargs["sink"] is sink

    events = sink.events_for("trace-1")
    assert [event.event for event in events] == [
        "orchestrator.intake",
        "orchestrator.plan",
        "orchestrator.completed",
    ]
    assert all(event.agent == "orchestrator" for event in events)
    plan = events[1].payload
    assert plan["planner"] == "deterministic"
    assert plan["live_fetch"] is False
    assert plan["skipped_stages"] == ["guard_input", "memory_check", "guard_output"]
    assert events[-1].payload["status"] == "answered"
    assert events[-1].payload["citations"] == 1
    assert events[-1].latency_ms is not None


async def test_synthesis_refusal_becomes_insufficient_evidence() -> None:
    fakes = FakeStages(
        verified=make_verified(can_answer=False, reason="no claim covers the question"),
        synthesis=make_synthesis(refused=True, answer="I can't answer that.", citations=0),
    )
    sink = InMemoryEventSink()
    response = await make_orchestrator(fakes, sink).ask(make_request(), trace_id="t-refuse")

    assert response.refusal is not None
    assert response.refusal.reason == "insufficient_evidence"
    assert "no claim covers the question" in (response.refusal.detail or "")
    assert response.citations == []
    assert response.answer == "I can't answer that."
    assert response.degraded is False
    assert [event.event for event in sink.events_for("t-refuse")][-1] == "orchestrator.refused"


async def test_scrub_failure_refuses_and_never_calls_a_model() -> None:
    fakes = FakeStages(scrub_error=PiiUnavailableError("spaCy model missing"))
    sink = InMemoryEventSink()

    response = await make_orchestrator(fakes, sink).ask(make_request(), trace_id="t-pii")

    assert fakes.order() == ["retrieve", "scrub"], "no LLM may run on unscrubbed text"
    assert response.refusal is not None
    assert response.refusal.reason == "guardrail_blocked"
    assert "personal data" in (response.refusal.detail or "")
    assert response.citations == []
    refused = sink.events_for("t-pii")[-1]
    assert refused.event == "orchestrator.refused"
    assert refused.payload["reason"] == "guardrail_blocked"


async def test_retrieval_config_error_is_not_a_refusal() -> None:
    fakes = FakeStages(retrieve_error=RetrievalConfigError("needs DATABASE_URL"))
    sink = InMemoryEventSink()

    with pytest.raises(RetrievalConfigError):
        await make_orchestrator(fakes, sink).ask(make_request(), trace_id="t-config")

    events = sink.events_for("t-config")
    assert events[-1].event == "orchestrator.failed"
    assert events[-1].payload["stage"] == "retrieve"


async def test_unexpected_retrieval_failure_is_a_degraded_refusal() -> None:
    fakes = FakeStages(retrieve_error=RuntimeError("pool exploded"))
    sink = InMemoryEventSink()

    response = await make_orchestrator(fakes, sink).ask(make_request(), trace_id="t-boom")

    assert response.refusal is not None
    assert response.refusal.reason == "error"
    assert "RuntimeError" in (response.refusal.detail or "")
    assert response.degraded is True


async def test_verify_llm_error_is_a_refusal_without_leaking_details() -> None:
    fakes = FakeStages(verify_error=LLMAuthError("401 from provider"))
    sink = InMemoryEventSink()

    response = await make_orchestrator(fakes, sink).ask(make_request(), trace_id="t-auth")

    assert response.refusal is not None
    assert response.refusal.reason == "error"
    assert "LLMAuthError" in (response.refusal.detail or "")
    assert "401" not in response.answer
    assert fakes.order() == ["retrieve", "scrub", "verify"]
    events = sink.events_for("t-auth")
    failed = [event for event in events if event.event == "orchestrator.failed"]
    assert failed, [event.event for event in events]
    assert failed[0].payload["stage"] == "verify"
    assert failed[0].payload["error_type"] == "LLMAuthError"


async def test_synthesis_failure_is_a_refusal() -> None:
    fakes = FakeStages(synthesize_error=LLMTimeoutError("no response"))
    sink = InMemoryEventSink()

    response = await make_orchestrator(fakes, sink).ask(make_request(), trace_id="t-syn")

    assert response.refusal is not None
    assert response.refusal.reason == "error"
    assert response.citations == []


async def test_retrieval_fallback_marks_the_answer_degraded() -> None:
    fakes = FakeStages(pack=make_pack(degraded_reason="embedding_failed:fts_only"))
    sink = InMemoryEventSink()

    response = await make_orchestrator(fakes, sink).ask(make_request(), trace_id="t-degraded")

    assert response.refusal is None
    assert response.degraded is True
    completed = sink.events_for("t-degraded")[-1]
    assert completed.payload["degraded_reason"] == "embedding_failed:fts_only"


async def test_budget_exhausted_before_verify_refuses_without_calling_a_model() -> None:
    fakes = FakeStages(retrieve_delay_s=0.02)
    sink = InMemoryEventSink()

    response = await make_orchestrator(fakes, sink, budget_s=0.01).ask(
        make_request(), trace_id="t-budget"
    )

    assert response.refusal is not None
    assert response.refusal.reason == "error"
    assert "budget" in (response.refusal.detail or "")
    assert response.degraded is True
    assert fakes.order() == ["retrieve", "scrub"]


async def test_slow_verification_hits_the_budget_and_refuses() -> None:
    fakes = FakeStages(verify_delay_s=0.5)
    sink = InMemoryEventSink()

    response = await make_orchestrator(fakes, sink, budget_s=0.05).ask(
        make_request(), trace_id="t-slow"
    )

    assert response.refusal is not None
    assert response.refusal.reason == "error"
    assert "budget" in (response.refusal.detail or "")
    assert response.degraded is True
    assert "synthesize" not in fakes.order()


async def test_blank_question_is_refused_as_out_of_scope() -> None:
    fakes = FakeStages()
    sink = InMemoryEventSink()

    response = await make_orchestrator(fakes, sink).ask(
        AskRequest(question="   ", session_id="s", user_id="u"), trace_id="t-blank"
    )

    assert response.refusal is not None
    assert response.refusal.reason == "out_of_scope"
    assert fakes.order() == []


async def test_trace_id_is_generated_when_absent_and_shared_by_every_event() -> None:
    fakes = FakeStages()
    sink = InMemoryEventSink()

    response = await make_orchestrator(fakes, sink).ask(make_request())

    assert len(response.trace_id) == 36
    events = sink.events_for(response.trace_id)
    assert len(events) == 3
    assert {event.trace_id for event in events} == {response.trace_id}


async def test_no_sink_means_no_database_requirement() -> None:
    """Standalone use (CLI, future MCP) must not need tracing infrastructure."""
    fakes = FakeStages()
    settings = Settings(database_url=None)
    orchestrator = Orchestrator(settings=settings, stages=fakes.as_stages())

    response = await orchestrator.ask(make_request())

    assert response.refusal is None
    assert response.citations


async def test_top_k_and_question_come_from_settings_and_request() -> None:
    fakes = FakeStages()
    sink = InMemoryEventSink()
    orchestrator = make_orchestrator(fakes, sink, top_k=4)

    await orchestrator.ask(AskRequest(question="Q about PHX-142", session_id="s", user_id="u"))

    retrieve_args, _ = fakes.args("retrieve")
    assert retrieve_args[0].top_k == 4
    synthesize_args, _ = fakes.args("synthesize")
    assert synthesize_args[0] == "Q about PHX-142"
    assert synthesize_args[1] is fakes.verified


# ---- answer cache (M4) ---------------------------------------------------

CACHED_ANSWER = "The cutover is 2026-11-27 [1]."


def make_cached_response(trace_id: str = "trace-original") -> AskResponse:
    """A response as an earlier request would have cached it."""
    return AskResponse(
        answer=CACHED_ANSWER,
        citations=[
            Citation(
                chunk_id="c-old",
                document_id="doc-old",
                source="jira",
                title="PHX-150 — Cut over billing webhooks",
                quote="The cutover moves to 2026-11-27.",
            )
        ],
        trace_id=trace_id,
    )


async def test_cache_hit_skips_the_pipeline() -> None:
    fakes = FakeStages()
    sink = InMemoryEventSink()
    cache = FakeAnswerCache()
    cache.seed("When is the PHX-150 cutover?", "priya@example.com", make_cached_response())

    response = await make_orchestrator(fakes, sink, cache=cache).ask(
        make_request(), trace_id="trace-hit"
    )

    assert fakes.order() == [], "a hit must not retrieve, scrub, verify or synthesize"
    assert response.answer == CACHED_ANSWER
    assert [citation.chunk_id for citation in response.citations] == ["c-old"]
    # The response carries the live trace; the producing one stays in the event.
    assert response.trace_id == "trace-hit"
    cached_entry = cache.responses[("when is the phx-150 cutover?", "priya@example.com", 0)]
    assert cached_entry.trace_id == "trace-original"

    events = sink.events_for("trace-hit")
    assert [event.event for event in events] == [
        "orchestrator.intake",
        "orchestrator.plan",
        "memory.cache_hit",
        "orchestrator.completed",
    ]
    hit = events[2]
    assert hit.agent == "memory"
    assert hit.payload["original_trace_id"] == "trace-original"
    assert hit.payload["version"] == 0
    assert hit.payload["cache_key_hash"] == cache_key_hash(
        "When is the PHX-150 cutover?", "priya@example.com", 0
    )
    assert "question" not in hit.payload, "the payload must never carry the question text"

    completed = events[-1]
    assert completed.agent == "orchestrator"
    assert completed.payload["status"] == "cache_hit"
    assert completed.payload["answer_chars"] == len(CACHED_ANSWER)
    assert completed.payload["citations"] == 1
    assert completed.payload["cached"] is True
    assert completed.latency_ms is not None


async def test_answered_response_is_cached_per_user_and_version() -> None:
    fakes = FakeStages()
    sink = InMemoryEventSink()
    cache = FakeAnswerCache()
    orchestrator = make_orchestrator(fakes, sink, cache=cache)

    first = await orchestrator.ask(make_request(), trace_id="t-first")

    assert first.refusal is None
    assert cache.normalized_puts() == [("when is the phx-150 cutover?", "priya@example.com", 0)]
    assert cache.puts[0].ttl_s == 900, "the ANSWER_CACHE_TTL_S default"
    assert sink.events_for("t-first")[-1].payload["cached"] is True

    # Same question up to casing/whitespace, same user: a hit, no pipeline.
    fakes.calls.clear()
    second = await orchestrator.ask(
        make_request("  WHEN is the   phx-150 cutover? "), trace_id="t-second"
    )

    assert fakes.order() == []
    assert second.answer == first.answer
    assert second.trace_id == "t-second"
    assert [event.event for event in sink.events_for("t-second")][-2:] == [
        "memory.cache_hit",
        "orchestrator.completed",
    ]

    # Identical question, different user: a miss. Retrieval enforces acl.users,
    # so that user may not be allowed the evidence this answer was built from.
    fakes.calls.clear()
    third = await orchestrator.ask(make_request(user_id="sam@example.com"), trace_id="t-third")

    assert fakes.order() == ["retrieve", "scrub", "verify", "synthesize"]
    assert third.trace_id == "t-third"
    assert cache.gets[-1] == ("when is the phx-150 cutover?", "sam@example.com", 0)

    # A data version bump (after a sync) makes the stored entry unreachable.
    fakes.calls.clear()
    cache.version_value = 1
    await orchestrator.ask(make_request(), trace_id="t-fourth")

    assert fakes.order() == ["retrieve", "scrub", "verify", "synthesize"]
    assert cache.normalized_puts()[-1] == ("when is the phx-150 cutover?", "priya@example.com", 1)


async def test_degraded_and_error_outcomes_are_not_cached() -> None:
    # Retrieval fell back, so a retry can legitimately produce a better answer.
    degraded = FakeStages(pack=make_pack(degraded_reason="embedding_failed:fts_only"))
    sink = InMemoryEventSink()
    cache = FakeAnswerCache()
    response = await make_orchestrator(degraded, sink, cache=cache).ask(
        make_request(), trace_id="t-degraded-cache"
    )

    assert response.degraded is True
    assert cache.puts == []
    assert sink.events_for("t-degraded-cache")[-1].payload["cached"] is False

    # A transient failure: the model may recover on the next attempt.
    failed = FakeStages(synthesize_error=LLMTimeoutError("no response"))
    sink = InMemoryEventSink()
    cache = FakeAnswerCache()
    response = await make_orchestrator(failed, sink, cache=cache).ask(
        make_request(), trace_id="t-error-cache"
    )

    assert response.refusal is not None
    assert response.refusal.reason == "error"
    assert cache.puts == []

    # "No evidence" is an outcome, not a failure: it is cached like an answer.
    refused = FakeStages(
        verified=make_verified(can_answer=False, reason="no claim covers the question"),
        synthesis=make_synthesis(refused=True, answer="I can't answer that.", citations=0),
    )
    sink = InMemoryEventSink()
    cache = FakeAnswerCache()
    response = await make_orchestrator(refused, sink, cache=cache).ask(
        make_request(), trace_id="t-insufficient-cache"
    )

    assert response.refusal is not None
    assert response.refusal.reason == "insufficient_evidence"
    assert cache.normalized_puts() == [("when is the phx-150 cutover?", "priya@example.com", 0)]
    assert sink.events_for("t-insufficient-cache")[-1].payload["cached"] is True


async def test_cache_unavailable_is_transparent() -> None:
    """A dead cache costs latency, never an answer."""
    fakes = FakeStages()
    sink = InMemoryEventSink()
    broken = ConnectionError("redis is down")
    cache = FakeAnswerCache(get_error=broken, put_error=broken, version_error=broken)

    response = await make_orchestrator(fakes, sink, cache=cache).ask(
        make_request(), trace_id="t-cache-down"
    )

    assert response.refusal is None
    assert response.citations
    assert fakes.order() == ["retrieve", "scrub", "verify", "synthesize"]
    events = sink.events_for("t-cache-down")
    assert [event.event for event in events] == [
        "orchestrator.intake",
        "orchestrator.plan",
        "orchestrator.completed",
    ]
    assert events[-1].payload["status"] == "answered"
    assert events[-1].payload["cached"] is False

    # Same guarantee when only the read/write fail and the version was readable.
    fakes.calls.clear()
    sink = InMemoryEventSink()
    cache = FakeAnswerCache(get_error=broken, put_error=broken)

    response = await make_orchestrator(fakes, sink, cache=cache).ask(
        make_request(), trace_id="t-cache-partial"
    )

    assert response.refusal is None
    assert fakes.order() == ["retrieve", "scrub", "verify", "synthesize"]
    partial_events = [event.event for event in sink.events_for("t-cache-partial")]
    assert partial_events[-1] == "orchestrator.completed"

