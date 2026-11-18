"""Pure ranking for hybrid retrieval.

No database, no clock, no network: every function here takes explicit inputs
(including ``now``) and returns a value. That makes the whole scoring model
unit-testable without Postgres, which matters because these are the numbers the
planner, the score floor, and the evidence stage all reason about.

The model, in order:

1. **Reciprocal rank fusion** (:func:`reciprocal_rank_fusion`) combines the
   pgvector ANN list and the Postgres FTS list. RRF ignores the raw scores
   (cosine distance and ``ts_rank_cd`` are not comparable) and uses only ranks,
   so there is nothing to calibrate.
2. **Multipliers**, each in ``[1 - weight, 1 + weight]`` and multiplied into the
   fused score:
   - recency — exponential decay on ``documents.updated_at``;
   - date proximity — only when the query names a date;
   - entity match — resolved planner entities against labels/title/source id.
   All three are bounded, so they reorder near-ties without letting a fresh
   irrelevant chunk outrank an exact match.
3. **Selection**: near-duplicate collapse (same document *and* same heading path
   *and* high word overlap), a per-document cap, then the score floor. The
   per-document cap is what stops one long issue from filling the pack: adjacent
   chunks of a long description share only their 15% overlap, so they are not
   near-duplicates, but the pack only needs a few of them.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from retrieval.options import (
    DEFAULT_DATE_PROXIMITY_DECAY_DAYS,
    DEFAULT_DATE_PROXIMITY_WEIGHT,
    DEFAULT_RECENCY_HALF_LIFE,
    DEFAULT_RECENCY_WEIGHT,
    RetrievalOptions,
)

# ---------------------------------------------------------------------------
# Candidate shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    """One chunk joined with its document, before scoring.

    Ranks are 1-based positions in the retriever that produced this row; a
    candidate found by only one retriever leaves the other as ``None``.
    """

    chunk_id: str
    document_id: str
    source: str
    source_id: str
    title: str
    url: str | None
    updated_at: datetime
    text: str
    meta: Mapping[str, Any] = field(default_factory=dict)
    vector_rank: int | None = None
    vector_similarity: float | None = None
    fts_rank: int | None = None
    fts_score: float | None = None

    @property
    def heading_path(self) -> tuple[str, ...]:
        """Chunker heading path (``()`` when absent or malformed)."""
        raw = self.meta.get("heading_path")
        if not isinstance(raw, list | tuple):
            return ()
        return tuple(str(part) for part in raw)

    @property
    def labels(self) -> tuple[str, ...]:
        """Chunker labels (source labels, not ACL labels)."""
        raw = self.meta.get("labels")
        if not isinstance(raw, list | tuple):
            return ()
        return tuple(str(part) for part in raw)


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """A candidate plus the numbers that decided its position."""

    candidate: Candidate
    score: float
    fused_score: float
    stale: bool


@dataclass(frozen=True, slots=True)
class RankOutcome:
    """Result of ranking and selecting candidates."""

    chunks: list[ScoredCandidate]
    dropped_below_floor: int = 0
    deduped: int = 0
    stale_chunk_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

#: A multi-word entity ("Project Phoenix") is also matched through its longer
#: words, so a document titled "Phoenix — Goals" still matches. Short words are
#: excluded because "team"/"page"/"of" match everything.
MIN_ENTITY_WORD_LEN = 4
_ENTITY_SPLIT_RE = re.compile(r"[^\w]+")


def _normalize_term(term: str) -> str:
    return " ".join(term.casefold().split())


def entity_needles(terms: Iterable[str]) -> list[str]:
    """Casefolded substrings that count as a match for the given entity terms.

    Used by both the SQL prefilter and the ranking boost, so the two never
    disagree about what "matches the entity" means. Order-preserving and
    de-duplicated.
    """
    needles: list[str] = []
    for term in terms:
        cleaned = _normalize_term(term)
        if not cleaned:
            continue
        if cleaned not in needles:
            needles.append(cleaned)
        words = [w for w in _ENTITY_SPLIT_RE.split(cleaned) if len(w) >= MIN_ENTITY_WORD_LEN]
        if len(words) > 1:  # leave single-word terms alone: they are already exact
            for word in words:
                if word not in needles:
                    needles.append(word)
    return needles


def entity_fields(candidate: Candidate) -> list[str]:
    """Fields the entity boost inspects: labels, source id, title, headings."""
    return [candidate.title, candidate.source_id, *candidate.labels, *candidate.heading_path]


def entity_multiplier(terms: Sequence[str], *, fields: Sequence[str], weight: float) -> float:
    """``1 + weight * matched_terms / total_terms``; 1.0 when nothing matches.

    Substring matching (casefolded) keeps it predictable: no tokenization
    surprises, no stemming, and behaviour identical to the SQL ``LIKE`` filter.
    """
    unique_terms = [t for t in dict.fromkeys(_normalize_term(t) for t in terms) if t]
    if not unique_terms or weight <= 0.0:
        return 1.0
    haystack = " | ".join(field.casefold() for field in fields if field)
    if not haystack:
        return 1.0
    matched = sum(1 for term in unique_terms if any(n in haystack for n in entity_needles([term])))
    return 1.0 + weight * matched / len(unique_terms)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    vector_ranks: Mapping[str, int],
    fts_ranks: Mapping[str, int],
    *,
    k: int = 60,
) -> dict[str, float]:
    """Sum ``1 / (k + rank)`` across retrievers for every chunk id.

    A chunk found by both retrievers scores strictly higher than one found by
    either alone at the same rank, which is the whole point of hybrid search.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    fused: dict[str, float] = {}
    for ranks in (vector_ranks, fts_ranks):
        for chunk_id, rank in ranks.items():
            if rank < 1:
                raise ValueError(f"rank for {chunk_id!r} must be >= 1")
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused


# ---------------------------------------------------------------------------
# Recency
# ---------------------------------------------------------------------------


def recency_multiplier(
    age: timedelta,
    *,
    half_life: timedelta = DEFAULT_RECENCY_HALF_LIFE,
    weight: float = DEFAULT_RECENCY_WEIGHT,
) -> float:
    """Bounded exponential decay: ``(1 - weight) + weight * 0.5 ** (age/half_life)``.

    A brand-new document scores 1.0; one half-life old scores ``1 - weight/2``;
    arbitrarily old asymptotes to ``1 - weight`` (0.65 by default), so recency
    can break ties but cannot erase relevance.
    """
    if half_life <= timedelta(0):
        raise ValueError("half_life must be positive")
    seconds = max(age.total_seconds(), 0.0)
    decay = math.pow(0.5, seconds / half_life.total_seconds())
    return (1.0 - weight) + weight * decay


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
_MONTH_DAY_RE = re.compile(
    r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b"
)
_DAY_MONTH_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?(?:,?\s+(\d{4}))?\b"
)


def find_dates(text: str, *, reference_year: int) -> tuple[date, ...]:
    """Dates mentioned in ``text``, oldest first.

    Supports ISO (``2026-11-20``) and English month forms (``Nov 20``,
    ``20 November 2026``); a month form without a year uses ``reference_year``
    (the request clock). Invalid calendar dates are skipped rather than raised —
    chunk text is untrusted input.
    """
    found: list[date] = []

    def add(year: int, month: int, day: int) -> None:
        try:
            found.append(date(year, month, day))
        except ValueError:
            return

    for match in _ISO_DATE_RE.finditer(text):
        add(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    for match in _MONTH_DAY_RE.finditer(text):
        month = _MONTHS.get(match.group(1).casefold())
        if month is not None:
            year = int(match.group(3)) if match.group(3) else reference_year
            add(year, month, int(match.group(2)))

    for match in _DAY_MONTH_RE.finditer(text):
        month = _MONTHS.get(match.group(2).casefold())
        if month is not None:
            year = int(match.group(3)) if match.group(3) else reference_year
            add(year, month, int(match.group(1)))

    return tuple(sorted(dict.fromkeys(found)))


def date_proximity_multiplier(
    query_dates: Sequence[date],
    candidate_dates: Sequence[date],
    *,
    weight: float = DEFAULT_DATE_PROXIMITY_WEIGHT,
    decay_days: float = DEFAULT_DATE_PROXIMITY_DECAY_DAYS,
) -> float:
    """``1 + weight * exp(-closest_gap_days / decay_days)``.

    Returns 1.0 when either side has no dates: a query without a date gets no
    bonus, which is what keeps plain topical questions from being reordered by
    document dates.
    """
    if not query_dates or not candidate_dates or weight <= 0.0:
        return 1.0
    if decay_days <= 0:
        raise ValueError("decay_days must be positive")
    closest = min(
        abs((query - candidate).days) for query in query_dates for candidate in candidate_dates
    )
    return 1.0 + weight * math.exp(-closest / decay_days)


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[^\w]+")


def _word_set(text: str) -> frozenset[str]:
    return frozenset(word for word in _WORD_RE.split(text.casefold()) if word)


def text_overlap(left: str, right: str) -> float:
    """Jaccard similarity of the two texts' word sets, in ``[0, 1]``."""
    left_words, right_words = _word_set(left), _word_set(right)
    if not left_words or not right_words:
        return 0.0
    intersection = len(left_words & right_words)
    return intersection / len(left_words | right_words)


def _is_near_duplicate(left: ScoredCandidate, right: ScoredCandidate, threshold: float) -> bool:
    """Same document *and* same heading path *and* high word overlap."""
    a, b = left.candidate, right.candidate
    if a.document_id != b.document_id or a.heading_path != b.heading_path:
        return False
    return text_overlap(a.text, b.text) >= threshold


# ---------------------------------------------------------------------------
# Ranking entry point
# ---------------------------------------------------------------------------


def rank_candidates(
    candidates: Sequence[Candidate],
    *,
    options: RetrievalOptions | None = None,
    now: datetime,
    query_text: str = "",
    entity_terms: Sequence[str] = (),
    freshness_window: timedelta | None = None,
) -> RankOutcome:
    """Fuse, score, de-duplicate, and floor one candidate set.

    ``now`` is supplied by the caller (never read from the clock here) so tests
    are reproducible and one request uses one consistent instant. ``updated_at``
    values must be timezone-aware, which the ``timestamptz`` columns guarantee.
    """
    opts = options or RetrievalOptions()
    if not candidates:
        return RankOutcome(chunks=[])

    fused = reciprocal_rank_fusion(
        {c.chunk_id: c.vector_rank for c in candidates if c.vector_rank is not None},
        {c.chunk_id: c.fts_rank for c in candidates if c.fts_rank is not None},
        k=opts.rrf_k,
    )

    query_dates = find_dates(query_text, reference_year=now.year)
    scored: list[ScoredCandidate] = []
    for candidate in candidates:
        fused_score = fused.get(candidate.chunk_id, 0.0)
        age = now - candidate.updated_at
        score = (
            fused_score
            * recency_multiplier(age, half_life=opts.recency_half_life, weight=opts.recency_weight)
            * entity_multiplier(
                entity_terms, fields=entity_fields(candidate), weight=opts.entity_weight
            )
            * date_proximity_multiplier(
                query_dates,
                (*find_dates(candidate.text, reference_year=now.year), candidate.updated_at.date()),
                weight=opts.date_proximity_weight,
                decay_days=opts.date_proximity_decay_days,
            )
        )
        stale = freshness_window is not None and age > freshness_window
        scored.append(
            ScoredCandidate(
                candidate=candidate, score=score, fused_score=fused_score, stale=stale
            )
        )

    # Deterministic order: score, then fused score, then id. Two identical
    # scores can only come from identical inputs, so the id tie-break is enough.
    scored.sort(key=lambda item: (-item.score, -item.fused_score, item.candidate.chunk_id))

    kept: list[ScoredCandidate] = []
    deduped = 0
    per_document: dict[str, int] = {}
    for item in scored:
        document_id = item.candidate.document_id
        if per_document.get(document_id, 0) >= opts.max_chunks_per_document:
            deduped += 1
            continue
        if any(_is_near_duplicate(item, existing, opts.dedupe_similarity) for existing in kept):
            deduped += 1
            continue
        per_document[document_id] = per_document.get(document_id, 0) + 1
        kept.append(item)

    above_floor = [item for item in kept if item.score >= opts.score_floor]
    dropped_below_floor = len(kept) - len(above_floor)
    return RankOutcome(
        chunks=above_floor,
        dropped_below_floor=dropped_below_floor,
        deduped=deduped,
        stale_chunk_ids=[item.candidate.chunk_id for item in above_floor if item.stale],
    )


__all__ = [
    "Candidate",
    "RankOutcome",
    "ScoredCandidate",
    "date_proximity_multiplier",
    "entity_fields",
    "entity_multiplier",
    "entity_needles",
    "find_dates",
    "rank_candidates",
    "recency_multiplier",
    "reciprocal_rank_fusion",
    "text_overlap",
]
