"""Retrieval tuning knobs.

Everything the hybrid ranker needs to decide *which* candidates surface. Kept in
one frozen dataclass so tests can pin every value and so a caller (orchestrator,
eval harness) can build a variant without touching module globals.

Values come from ``RETRIEVAL_*`` environment variables when the caller does not
pass an explicit :class:`RetrievalOptions` (see :meth:`RetrievalOptions.from_env`).
They are **not** in ``config.py``: that module is shared with the other agents
mid-flight, and these are pipeline tunables rather than deployment settings.

Defaults, and why:

``rrf_k = 60``
    The value from the retrieval brief (and the original RRF paper). Fusing two
    ranked lists, the best possible score is ``2 / (60 + 1) ≈ 0.0328``; the gap
    between ranks is small, which is the point of RRF: it needs no score
    calibration between cosine similarity and ``ts_rank_cd``.

``score_floor = 0.01``
    Applied to the *final* score. For a fresh (multiplier 1.0) chunk that is
    roughly "appears in the top ~40 of at least one retriever". Anything weaker
    is padding, and padding is what the evidence/synthesis stages must not see.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta

#: Reciprocal rank fusion constant.
DEFAULT_RRF_K = 60

#: Recency half-life: a 30-day-old document keeps half of its recency weight.
DEFAULT_RECENCY_HALF_LIFE = timedelta(days=30)

#: How much recency can move a score. 0.35 => a maximally old chunk keeps 65%
#: of its fused score. Multiplicative and bounded so it can reorder near-ties
#: but never bury a clearly-superior match.
DEFAULT_RECENCY_WEIGHT = 0.35

#: Bonus for chunks whose content dates sit near a date named in the query.
DEFAULT_DATE_PROXIMITY_WEIGHT = 0.25

#: Days over which the date-proximity bonus decays by 1/e.
DEFAULT_DATE_PROXIMITY_DECAY_DAYS = 14.0

#: Bonus when resolved entities match chunk metadata.
DEFAULT_ENTITY_WEIGHT = 0.5

#: Final-score floor; see the module docstring.
DEFAULT_SCORE_FLOOR = 0.01

#: Jaccard word overlap above which two chunks of the same document *and* the
#: same heading path count as one.
DEFAULT_DEDUPE_SIMILARITY = 0.82

#: Hard cap on chunks per document, so one long thread cannot take the pack.
DEFAULT_MAX_CHUNKS_PER_DOCUMENT = 4

#: Candidates pulled from each retriever before fusion.
DEFAULT_CANDIDATE_DEPTH = 50

#: Optional cosine-similarity gate for the vector list. 0.0 disables it: the
#: right threshold is model-specific (``nomic-embed-text`` has a high baseline,
#: so unrelated text still scores ~0.5), and dense ANN *always* returns its k
#: nearest neighbours. Operators who know their model's scale can set it; by
#: default relevance is decided by fusion, the score floor, and the evidence
#: stage instead of a guessed cutoff.
DEFAULT_MIN_VECTOR_SIMILARITY = 0.0


@dataclass(frozen=True, slots=True)
class RetrievalOptions:
    """All ranking tunables in one place (immutable)."""

    rrf_k: int = DEFAULT_RRF_K
    recency_half_life: timedelta = DEFAULT_RECENCY_HALF_LIFE
    recency_weight: float = DEFAULT_RECENCY_WEIGHT
    date_proximity_weight: float = DEFAULT_DATE_PROXIMITY_WEIGHT
    date_proximity_decay_days: float = DEFAULT_DATE_PROXIMITY_DECAY_DAYS
    entity_weight: float = DEFAULT_ENTITY_WEIGHT
    score_floor: float = DEFAULT_SCORE_FLOOR
    dedupe_similarity: float = DEFAULT_DEDUPE_SIMILARITY
    max_chunks_per_document: int = DEFAULT_MAX_CHUNKS_PER_DOCUMENT
    candidate_depth: int = DEFAULT_CANDIDATE_DEPTH
    min_vector_similarity: float = DEFAULT_MIN_VECTOR_SIMILARITY
    #: ``"filter"`` restricts SQL candidates to entity matches (planner intent);
    #: ``"boost"`` only re-ranks them. See ``search.entity_predicate``.
    entity_filter_mode: str = "filter"

    def __post_init__(self) -> None:
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be >= 1")
        if self.recency_half_life <= timedelta(0):
            raise ValueError("recency_half_life must be positive")
        for name in (
            "recency_weight",
            "date_proximity_weight",
            "entity_weight",
            "dedupe_similarity",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.date_proximity_decay_days <= 0:
            raise ValueError("date_proximity_decay_days must be positive")
        if self.score_floor < 0:
            raise ValueError("score_floor must be >= 0")
        if self.max_chunks_per_document < 1:
            raise ValueError("max_chunks_per_document must be >= 1")
        if self.candidate_depth < 1:
            raise ValueError("candidate_depth must be >= 1")
        if not -1.0 <= self.min_vector_similarity <= 1.0:
            raise ValueError("min_vector_similarity must be in [-1, 1]")
        if self.entity_filter_mode not in {"filter", "boost"}:
            raise ValueError("entity_filter_mode must be 'filter' or 'boost'")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> RetrievalOptions:
        """Build options from ``RETRIEVAL_*`` variables, falling back to defaults.

        Unparseable values raise ``ValueError`` instead of being silently
        ignored: a typo in a tunable should be a loud startup failure, not a
        mystery ranking difference.
        """
        env = os.environ if environ is None else environ

        def read_float(name: str, default: float) -> float:
            raw = env.get(name)
            return default if raw is None or raw == "" else float(raw)

        def read_int(name: str, default: int) -> int:
            raw = env.get(name)
            return default if raw is None or raw == "" else int(raw)

        half_life_days = read_float(
            "RETRIEVAL_RECENCY_HALF_LIFE_DAYS", DEFAULT_RECENCY_HALF_LIFE.total_seconds() / 86400
        )
        return cls(
            rrf_k=read_int("RETRIEVAL_RRF_K", DEFAULT_RRF_K),
            recency_half_life=timedelta(days=half_life_days),
            recency_weight=read_float("RETRIEVAL_RECENCY_WEIGHT", DEFAULT_RECENCY_WEIGHT),
            date_proximity_weight=read_float(
                "RETRIEVAL_DATE_PROXIMITY_WEIGHT", DEFAULT_DATE_PROXIMITY_WEIGHT
            ),
            date_proximity_decay_days=read_float(
                "RETRIEVAL_DATE_PROXIMITY_DECAY_DAYS", DEFAULT_DATE_PROXIMITY_DECAY_DAYS
            ),
            entity_weight=read_float("RETRIEVAL_ENTITY_WEIGHT", DEFAULT_ENTITY_WEIGHT),
            score_floor=read_float("RETRIEVAL_SCORE_FLOOR", DEFAULT_SCORE_FLOOR),
            dedupe_similarity=read_float(
                "RETRIEVAL_DEDUPE_SIMILARITY", DEFAULT_DEDUPE_SIMILARITY
            ),
            max_chunks_per_document=read_int(
                "RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT", DEFAULT_MAX_CHUNKS_PER_DOCUMENT
            ),
            candidate_depth=read_int("RETRIEVAL_CANDIDATE_DEPTH", DEFAULT_CANDIDATE_DEPTH),
            min_vector_similarity=read_float(
                "RETRIEVAL_MIN_VECTOR_SIMILARITY", DEFAULT_MIN_VECTOR_SIMILARITY
            ),
            entity_filter_mode=env.get("RETRIEVAL_ENTITY_FILTER_MODE", "filter"),
        )


__all__ = [
    "DEFAULT_CANDIDATE_DEPTH",
    "DEFAULT_DATE_PROXIMITY_DECAY_DAYS",
    "DEFAULT_DATE_PROXIMITY_WEIGHT",
    "DEFAULT_DEDUPE_SIMILARITY",
    "DEFAULT_ENTITY_WEIGHT",
    "DEFAULT_MAX_CHUNKS_PER_DOCUMENT",
    "DEFAULT_MIN_VECTOR_SIMILARITY",
    "DEFAULT_RECENCY_HALF_LIFE",
    "DEFAULT_RECENCY_WEIGHT",
    "DEFAULT_RRF_K",
    "DEFAULT_SCORE_FLOOR",
    "RetrievalOptions",
]
