"""Evidence Agent service: scrubbed chunks in, verified claims out.

One LLM step (plus at most one repair retry), then deterministic validation and
sufficiency scoring. Failure is always fail-closed: zero claims and
``can_answer=False`` rather than a guess.
"""

from __future__ import annotations

import logging

from config import Settings, get_settings
from db.tracing import EventSink, NullEventSink, Tracer
from evidence.parsing import EvidenceParseError, ExtractionResult, parse_extraction
from evidence.prompts import MAX_OUTPUT_TOKENS, build_messages
from evidence.sufficiency import SufficiencyResult, score_sufficiency
from evidence.validator import ValidationOutcome, validate_claims
from llm.base import CompletionRequest, LLMClient, Message
from llm.openai_compatible import OpenAICompatibleClient
from orchestrator.pipeline import ScrubbedPack, VerifiedClaims

__all__ = ["verify"]

logger = logging.getLogger(__name__)

AGENT = "evidence"


def _usage(client: LLMClient) -> dict[str, object]:
    """Best-effort token usage; fake clients in tests have no ledger."""
    ledger: object = getattr(client, "ledger", None)
    as_payload = getattr(ledger, "as_payload", None)
    if callable(as_payload):
        payload: object = as_payload()
        if isinstance(payload, dict):
            return {str(key): value for key, value in payload.items()}
    return {}


def _default_client(settings: Settings) -> OpenAICompatibleClient:
    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    return OpenAICompatibleClient(
        base_url=settings.llm_base_url,
        api_key=api_key,
        default_model=settings.llm_model,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        backoff_base_s=settings.llm_backoff_base_s,
    )


async def _extract(client: LLMClient, messages: list[Message], model: str) -> str:
    """One completion; returns raw text (parsing/validation happens elsewhere)."""
    response = await client.complete(
        CompletionRequest(
            messages=messages,
            model=model,
            temperature=0.0,
            # Reasoning models spend tokens thinking before answering.
            max_tokens=MAX_OUTPUT_TOKENS,
            # The local 8B JSON model ignores the "JSON only" instruction once a
            # pack gets long and messy, answering in prose instead. Only the
            # provider-level constraint reliably forces a JSON object.
            extra={"response_format": {"type": "json_object"}},
        )
    )
    return response.text


def _provenance(pack: ScrubbedPack) -> dict[str, str]:
    """chunk_id → provenance text for coverage scoring.

    Titles, source ids, labels and sections name the document a claim cites, so
    a claim about a Phoenix task can cover "Phoenix" without repeating it.
    """
    provenance: dict[str, str] = {}
    for chunk in pack.chunks:
        parts = [chunk.title, chunk.source_id]
        labels = chunk.meta.get("labels")
        if isinstance(labels, list):
            parts.extend(str(label) for label in labels if label)
        section = chunk.meta.get("section")
        if section:
            parts.append(str(section))
        provenance[chunk.chunk_id] = " ".join(part for part in parts if part)
    return provenance


def _insufficiency_reason(outcome: ValidationOutcome, suff: SufficiencyResult) -> str:
    if not outcome.claims:
        return (
            "no supported claims survived validation "
            f"(dangling citations: {outcome.dangling_ids}, "
            f"unsupported: {outcome.unsupported}, "
            f"injection-like: {len(outcome.injection_dropped)})"
        )
    missing = ", ".join(suff.missing[:5]) or "none"
    return (
        f"question coverage {suff.score:.2f} is below the {suff.threshold:.2f} threshold; "
        f"not covered by evidence: {missing}"
    )


async def verify(
    question: str,
    pack: ScrubbedPack,
    *,
    trace_id: str | None = None,
    client: LLMClient | None = None,
    sink: EventSink | None = None,
    settings: Settings | None = None,
) -> VerifiedClaims:
    """Extract, validate, and score claims for ``question`` given ``pack``.

    Args:
        question: the user's question (falls back to ``pack.question``).
        pack: scrubbed evidence — the only data the model is allowed to see.
        trace_id: reused for audit events; generated when omitted.
        client: injected LLM client (tests); a real one is built and closed when
            omitted.
        sink: audit sink; defaults to a no-op sink so this needs no database.
        settings: config override (thresholds, model, endpoint, chunk cap).
    """
    config = settings or get_settings()
    model = config.llm_json_model or config.llm_model
    tracer = Tracer(sink or NullEventSink(), trace_id)
    asked = question.strip() or pack.question
    threshold = config.evidence_sufficiency_threshold
    overlap = config.evidence_min_claim_overlap

    if not pack.chunks:
        await tracer.emit(
            AGENT,
            "evidence.extracted",
            {
                "claims": 0,
                "chunks": 0,
                "chunks_sent": 0,
                "reason": "empty_pack",
                "question": asked,
            },
        )
        await tracer.emit(
            AGENT,
            "evidence.insufficient",
            {"sufficiency": 0.0, "threshold": threshold, "reason": "no_chunks"},
        )
        return VerifiedClaims(
            question=asked,
            sufficiency=0.0,
            can_answer=False,
            insufficiency_reason="retrieval returned no evidence chunks",
        )

    owns_client = client is None
    llm: LLMClient = client or _default_client(config)
    calls = 0
    extraction: ExtractionResult | None = None
    # The model, not the retriever, sets the pack size: a long pack makes the
    # local 8B JSON model echo the input instead of answering. Packs are
    # relevance-ordered, so the slice keeps the best evidence for both attempts.
    sent_chunks = pack.chunks[: config.evidence_extraction_max_chunks]
    try:
        raw = await _extract(llm, build_messages(asked, sent_chunks), model)
        calls += 1
        try:
            extraction = parse_extraction(raw)
        except EvidenceParseError as exc:
            # The retry re-sends the same capped, ranked slice with the error text.
            await tracer.emit(
                AGENT,
                "evidence.repair_attempted",
                {
                    "error": str(exc)[:500],
                    "question": asked,
                    "chunks": len(sent_chunks),
                    "chunks_before": len(pack.chunks),
                },
            )
            repair_messages = build_messages(asked, sent_chunks, repair=True, error=str(exc))
            raw = await _extract(llm, repair_messages, model)
            calls += 1
            extraction = parse_extraction(raw)
    except EvidenceParseError as exc:
        await tracer.emit(
            AGENT,
            "evidence.parse_failed",
            {"error": str(exc)[:500], "calls": calls, "usage": _usage(llm)},
        )
        return VerifiedClaims(
            question=asked,
            sufficiency=0.0,
            can_answer=False,
            insufficiency_reason=f"claim extraction failed: {exc}",
        )
    finally:
        if owns_client:
            await llm.aclose()

    assert extraction is not None  # guaranteed by the try/except above
    usage = _usage(llm)
    outcome = validate_claims(extraction.claims, pack.chunks, min_overlap=overlap)
    suff = score_sufficiency(
        asked,
        outcome.claims,
        threshold=threshold,
        context=_provenance(pack),
    )

    await tracer.emit(
        AGENT,
        "evidence.extracted",
        {
            "claims": len(outcome.claims),
            "dropped_dangling": outcome.dangling_ids,
            "dropped_unsupported": outcome.unsupported,
            "dropped_injection": len(outcome.injection_dropped),
            "duplicates_merged": outcome.duplicates_merged,
            "contradictions": [
                {"subject": c.subject, "attribute": c.attribute, "values": list(c.values)}
                for c in outcome.contradictions
            ],
            "chunks": len(pack.chunks),
            "chunks_sent": len(sent_chunks),
            "calls": calls,
            "sufficiency": suff.score,
            "can_answer": suff.can_answer,
            "usage": usage,
        },
    )
    if outcome.injection_dropped:
        await tracer.emit(
            AGENT,
            "evidence.injection_dropped",
            {
                "count": len(outcome.injection_dropped),
                "samples": [claim.text[:120] for claim in outcome.injection_dropped[:3]],
            },
        )

    event = "evidence.sufficient" if suff.can_answer else "evidence.insufficient"
    await tracer.emit(
        AGENT,
        event,
        {
            "sufficiency": suff.score,
            "threshold": suff.threshold,
            "covered": list(suff.covered),
            "missing": list(suff.missing),
        },
    )

    can_answer = suff.can_answer
    return VerifiedClaims(
        question=asked,
        claims=outcome.claims,
        dropped_claims=[*outcome.dropped, *outcome.injection_dropped],
        sufficiency=suff.score,
        can_answer=can_answer,
        insufficiency_reason=None if can_answer else _insufficiency_reason(outcome, suff),
    )
