"""Pure post-validation of extracted claims — no LLM calls.

The model proposes; this module decides what survives. Every rejection has a
reason and is reported, because "the answer got worse" must be traceable to a
specific rule.

Order matters: citation validity -> injection filter -> lexical support ->
dedupe -> contradiction detection -> renumbering.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from evidence.parsing import ExtractedClaim
from orchestrator.pipeline import Claim, ScrubbedChunk

__all__ = [
    "DEFAULT_MIN_OVERLAP",
    "Contradiction",
    "ValidationOutcome",
    "is_injection",
    "lexical_support",
    "tokenise",
    "validate_claims",
]

logger = logging.getLogger(__name__)

#: Default share of a claim's content tokens that must appear in its cited
#: chunks. Tuned via evals; configurable through ``EVIDENCE_MIN_CLAIM_OVERLAP``.
DEFAULT_MIN_OVERLAP = 0.4

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NORMALISE_RE = re.compile(r"[^a-z0-9]+")

_STOPWORD_TEXT = (
    "a an and are as at be been but by can could did do does for from had has have he her his "
    "if in into is it its may might must not of on or our shall she should than that the their "
    "them then there these they this those to was we were what when where which while who will "
    "with would you your"
)
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())

#: Instruction-shaped text must never become a "claim". Matching here is
#: deliberate: documents in the corpus plant these exact shapes.
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bignore\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+"
        r"(?:instructions?|prompts?|rules?)",
        re.IGNORECASE,
    ),
    re.compile(r"\bdisregard\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|above|earlier)",
               re.IGNORECASE),
    re.compile(r"\b(?:reveal|print|show|output|repeat|leak)\s+(?:your\s+|the\s+)?"
               r"(?:system\s+prompt|api\s+key|credentials?|secrets?)", re.IGNORECASE),
    re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"\bnew\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE),
    re.compile(r"\bdo\s+not\s+(?:follow|obey)\b", re.IGNORECASE),
    re.compile(r"\b(?:reply|respond)\s+(?:only\s+)?with\b", re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class Contradiction:
    """Two or more different values asserted for the same subject+attribute."""

    subject: str
    attribute: str
    values: tuple[str, ...]
    claim_ids: tuple[int, ...]


@dataclass(slots=True)
class ValidationOutcome:
    """Kept claims plus everything rejected and why."""

    claims: list[Claim] = field(default_factory=list)
    dropped: list[Claim] = field(default_factory=list)
    injection_dropped: list[Claim] = field(default_factory=list)
    dangling_ids: int = 0
    unsupported: int = 0
    duplicates_merged: int = 0
    contradictions: list[Contradiction] = field(default_factory=list)


def tokenise(text: str) -> list[str]:
    """Lowercase content tokens: words of 3+ chars, digits, placeholders."""
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(text.lower()):
        if (len(match) >= 3 or match.isdigit()) and match not in _STOPWORDS:
            tokens.append(match)
    return tokens


def is_injection(text: str) -> bool:
    """True when the text is shaped like an instruction, not a work-state fact."""
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


def lexical_support(claim_text: str, cited_texts: Sequence[str]) -> float:
    """Share of the claim's content tokens present in the cited chunks."""
    claim_tokens = tokenise(claim_text)
    if not claim_tokens:
        return 0.0
    support = set(tokenise(" ".join(cited_texts)))
    matched = sum(1 for token in claim_tokens if token in support)
    return matched / len(claim_tokens)


def _to_claim(claim: ExtractedClaim, claim_id: int) -> Claim:
    return Claim(
        claim_id=claim_id,
        text=claim.text,
        chunk_ids=list(claim.chunk_ids),
        confidence=claim.confidence,
    )


def _dedupe_key(text: str) -> str:
    return _NORMALISE_RE.sub(" ", text.lower()).strip()


def _detect_contradictions(claims: Sequence[Claim], extracted: Sequence[ExtractedClaim]) -> list[
    Contradiction
]:
    """Group kept claims by subject+attribute; report conflicting values."""
    groups: dict[tuple[str, str], list[tuple[Claim, str]]] = {}
    for claim, source in zip(claims, extracted, strict=True):
        subject = (source.subject or "").strip().lower()
        attribute = (source.attribute or "").strip().lower()
        value = (source.value or "").strip()
        if not subject or not attribute or not value:
            continue
        groups.setdefault((subject, attribute), []).append((claim, value.lower()))

    contradictions: list[Contradiction] = []
    for (subject, attribute), members in groups.items():
        distinct = {value for _, value in members}
        if len(distinct) < 2:
            continue
        contradictions.append(
            Contradiction(
                subject=subject,
                attribute=attribute,
                values=tuple(sorted(distinct)),
                claim_ids=tuple(claim.claim_id for claim, _ in members),
            )
        )
        logger.warning(
            "contradictory evidence kept for %s / %s: %s",
            subject,
            attribute,
            sorted(distinct),
        )
    return contradictions


def validate_claims(
    extracted: Sequence[ExtractedClaim],
    chunks: Sequence[ScrubbedChunk],
    *,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> ValidationOutcome:
    """Validate extracted claims against the chunks that were actually provided.

    Dropped claims keep the extraction index as ``claim_id`` so a trace can point
    at the exact model proposal.
    """
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    outcome = ValidationOutcome()

    kept_pairs: list[tuple[ExtractedClaim, list[str]]] = []

    for index, claim in enumerate(extracted):
        valid_ids = [chunk_id for chunk_id in claim.chunk_ids if chunk_id in by_id]
        if not valid_ids:
            outcome.dangling_ids += 1
            outcome.dropped.append(_to_claim(claim, index))
            continue

        if is_injection(claim.text):
            outcome.injection_dropped.append(_to_claim(claim, index))
            continue

        support = lexical_support(claim.text, [by_id[chunk_id].text for chunk_id in valid_ids])
        if support < min_overlap:
            outcome.unsupported += 1
            outcome.dropped.append(_to_claim(claim, index))
            continue

        kept_pairs.append((claim, valid_ids))

    # Dedupe: identical (normalised) text keeps one claim with the union of ids.
    merged: dict[str, tuple[ExtractedClaim, list[str]]] = {}
    order: list[str] = []
    for claim, ids in kept_pairs:
        key = _dedupe_key(claim.text)
        existing = merged.get(key)
        if existing is None:
            merged[key] = (claim, list(ids))
            order.append(key)
            continue
        outcome.duplicates_merged += 1
        existing_claim, existing_ids = existing
        for chunk_id in ids:
            if chunk_id not in existing_ids:
                existing_ids.append(chunk_id)
        if claim.confidence > existing_claim.confidence:
            merged[key] = (claim, existing_ids)

    kept_extracted = [merged[key][0] for key in order]
    final: list[Claim] = []
    for position, key in enumerate(order):
        claim, ids = merged[key]
        final.append(
            Claim(claim_id=position, text=claim.text, chunk_ids=ids, confidence=claim.confidence)
        )

    outcome.claims = final
    outcome.contradictions = _detect_contradictions(final, kept_extracted)
    return outcome
