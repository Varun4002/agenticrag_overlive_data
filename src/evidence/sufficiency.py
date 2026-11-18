"""Deterministic question-coverage scoring for verified claims.

No LLM here on purpose: sufficiency decides answer-vs-refuse, so it must be
reproducible and cheap. The score is the share of the question's *aspects*
(ticket keys, placeholders, quoted phrases, proper nouns, content words) that
the surviving claims actually mention, either in their own text or — via the
citable chunks' provenance — in the documents they cite.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from orchestrator.pipeline import Claim

__all__ = [
    "DEFAULT_THRESHOLD",
    "Aspect",
    "SufficiencyResult",
    "question_aspects",
    "score_sufficiency",
]

DEFAULT_THRESHOLD = 0.5

_TICKET_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}-\d{1,6}\b")
_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_]*_\d+\]")
_QUOTED_RE = re.compile(r'"([^"{}\n]{3,80})"')
_CAPITALISED_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")
_WORD_RE = re.compile(r"[a-z0-9]+")

_MAX_KEYWORDS = 8

#: Capitalised words that carry no entity signal in a question.
_IGNORED_CAPITALISED = frozenset(
    {
        "Project", "The", "What", "Which", "When", "Where", "Who", "How", "Why",
        "Give", "List", "Show", "Summarize", "Summarise", "Tell", "Any", "Does",
        "Did", "Has", "Have", "Will", "Was", "Were", "Are", "Is", "And", "But",
        "For", "Our", "Their", "This", "That", "These", "Those", "Please",
    }
)

_STOPWORD_TEXT = (
    "a an and are as at be been but by can could did do does for from had has have he her his "
    "if in into is it its may might must not of on or our shall she should than that the their "
    "them then there these they this those to was we were what when where which while who will "
    "with would you your "
    # Answer-form scaffolding: these say how to present the answer, not what it
    # is about, so they must never become keyword aspects.
    "give list show tell summarize summarise each please"
)
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())

AspectKind = Literal["ticket", "placeholder", "phrase", "name", "keyword"]


@dataclass(frozen=True, slots=True)
class Aspect:
    """One thing the question asks about."""

    text: str
    kind: AspectKind
    weight: float


@dataclass(frozen=True, slots=True)
class SufficiencyResult:
    """Outcome of coverage scoring."""

    score: float
    threshold: float
    can_answer: bool
    aspects: tuple[Aspect, ...] = ()
    covered: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()


def _add(aspects: list[Aspect], seen: dict[str, int], aspect: Aspect) -> None:
    key = aspect.text.lower()
    if key in seen:
        index = seen[key]
        if aspect.weight > aspects[index].weight:
            aspects[index] = aspect
        return
    seen[key] = len(aspects)
    aspects.append(aspect)


def question_aspects(question: str) -> tuple[Aspect, ...]:
    """Extract weighted aspects from the question, strongest first."""
    aspects: list[Aspect] = []
    seen: dict[str, int] = {}

    for match in _TICKET_RE.finditer(question):
        _add(aspects, seen, Aspect(text=match.group(0), kind="ticket", weight=3.0))

    for match in _PLACEHOLDER_RE.finditer(question):
        _add(aspects, seen, Aspect(text=match.group(0), kind="placeholder", weight=2.0))

    for match in _QUOTED_RE.finditer(question):
        phrase = match.group(1).strip()
        if phrase:
            _add(aspects, seen, Aspect(text=phrase, kind="phrase", weight=2.0))

    for match in _CAPITALISED_RE.finditer(question):
        word = match.group(1)
        if word in _IGNORED_CAPITALISED:
            continue
        _add(aspects, seen, Aspect(text=word, kind="name", weight=1.5))

    keyword_count = 0
    for word in _WORD_RE.findall(question.lower()):
        if keyword_count >= _MAX_KEYWORDS:
            break
        if len(word) < 4 or word in _STOPWORDS:
            continue
        if word in seen:
            continue
        # Generic keywords describe the shape of the sought answer, not its
        # identity: they may not (and need not) appear in a claim's phrasing,
        # so they must not sink the score below the threshold on their own.
        _add(aspects, seen, Aspect(text=word, kind="keyword", weight=0.25))
        keyword_count += 1

    return tuple(aspects)


def score_sufficiency(
    question: str,
    claims: Sequence[Claim],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    context: Mapping[str, str] | None = None,
) -> SufficiencyResult:
    """Score how much of the question the surviving claims cover.

    An aspect counts as covered when it appears in a claim's text or in the
    provenance text (title, source, labels, section) of a chunk that claim
    cites: "Phoenix" is answered by a claim over a Phoenix document even when
    the claim says only "PHX-142 is due 2026-11-10". Uncited chunks never count.

    Fails closed: no claims, or a question with no aspects, scores 0.0.
    """
    aspects = question_aspects(question)
    if not claims or not aspects:
        return SufficiencyResult(
            score=0.0,
            threshold=threshold,
            can_answer=False,
            aspects=aspects,
            missing=tuple(aspect.text for aspect in aspects),
        )

    pieces = [claim.text for claim in claims]
    if context:
        cited = sorted({chunk_id for claim in claims for chunk_id in claim.chunk_ids})
        pieces.extend(text for chunk_id in cited if (text := context.get(chunk_id)))
    haystack = " ".join(pieces).lower()
    covered: list[str] = []
    missing: list[str] = []
    covered_weight = 0.0
    total_weight = 0.0
    for aspect in aspects:
        total_weight += aspect.weight
        if aspect.text.lower() in haystack:
            covered.append(aspect.text)
            covered_weight += aspect.weight
        else:
            missing.append(aspect.text)

    score = round(covered_weight / total_weight, 4) if total_weight else 0.0
    return SufficiencyResult(
        score=score,
        threshold=threshold,
        can_answer=score >= threshold,
        aspects=aspects,
        covered=tuple(covered),
        missing=tuple(missing),
    )
