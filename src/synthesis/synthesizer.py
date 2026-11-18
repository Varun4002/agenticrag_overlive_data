"""Answer synthesis: verified claims in, cited answer out.

Invariants:
- Nothing but ``VerifiedClaims`` may be restated. Uncited factual sentences are
  either matched to a verified claim deterministically (with a numeric guard) or
  stripped. Matching exists because the model can ignore the marker instruction
  while still writing a faithful answer: a live 8B draft restated seven verified
  claims and cited none, so stripping turned a good answer into a refusal.
- Citation markers are renumbered so every ``[n]`` in the delivered answer maps
  to exactly one entry in the delivered citations (duplicate claims on the same
  chunk collapse to one citation).
- Refusal is deterministic (no LLM call) whenever the evidence layer says the
  question cannot be answered, or the model's draft hit the output-token limit.
- Every refusal also carries its deterministic reason in
  ``SynthesisResult.refusal_reason``, so the orchestrator can report it verbatim
  instead of a generic fallback.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from config import Settings, get_settings
from db.tracing import EventSink, NullEventSink, Tracer
from llm.base import CompletionRequest, LLMClient
from llm.openai_compatible import OpenAICompatibleClient
from orchestrator.contracts import Citation
from orchestrator.pipeline import (
    Claim,
    ScrubbedChunk,
    ScrubbedPack,
    SynthesisResult,
    VerifiedClaims,
)
from synthesis.prompts import MAX_OUTPUT_TOKENS, build_messages

__all__ = ["refusal_answer", "synthesize"]

logger = logging.getLogger(__name__)

AGENT = "synthesis"

_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_LABEL_RE = re.compile(r"\b(?:citations?|references?)\b:?", re.IGNORECASE)

#: Narration the model wraps around its own markers ("as indicated in claim [1]").
#: The system prompt forbids it and the deployed model writes it anyway, so the
#: cleaner strips the phrase and keeps the marker it pointed at. A list of plain
#: phrases, not one opaque regex: each shape is readable and testable on its own.
META_CITATION_PHRASES: tuple[str, ...] = (
    "as indicated in claim",
    "as stated in claim",
    "as mentioned in claim",
    "as shown in claim",
    "as noted in claim",
    "as described in claim",
    "as per claim",
    "according to claim",
)


def _meta_pattern(phrase: str) -> str:
    """Regex for one listed phrase: any spacing, optional ``the`` before ``claim``."""
    head, _, tail = phrase.rpartition(" ")
    pattern = r"\s+".join(re.escape(part) for part in head.split())
    if tail == "claim":
        return rf"{pattern}\s+(?:the\s+)?claims?"
    return rf"{pattern}\s+{re.escape(tail)}"


_META_CITATION_RE = re.compile(
    "|".join(_meta_pattern(phrase) for phrase in META_CITATION_PHRASES), re.IGNORECASE
)

#: ``(claim [1])`` — the same narration as a parenthetical. The marker survives.
_CLAIM_PAREN_RE = re.compile(r"\(\s*(?:the\s+)?claims?\s*(?:\[(\d+)\])?\s*\)", re.IGNORECASE)

#: Characters of the source chunk kept as the citation quote.
QUOTE_CHARS = 200

#: Sentences shorter than this are treated as headings/fragments, not claims.
MIN_FACTUAL_WORDS = 4

#: Share of a marker-less sentence's words that must appear in one verified claim
#: before that claim is treated as the sentence's support. 0.7 is measured, not
#: guessed: the live marker-less sentence scores 0.733 against its claim (so the
#: real incident is still recovered), while "The cutover is 2026-11-20 and the
#: vendor agreed." against "The cutover is 2026-11-20." scores 0.667 — an
#: unsupported tail must be rejected instead of riding along on a citation.
#: High on purpose: a wrong citation is worse than a dropped sentence, which
#: merely refuses.
MIN_IMPLIED_CITE_OVERLAP = 0.7

#: Explicit negation words. Matching the sentence against a claim of the opposite
#: polarity would cite the inverse of what the claim says, so the polarity has to
#: agree. Semantic negation ("delayed" vs "on time", "up" vs "down") is out of
#: reach for a lexical check.
_NEGATION_CUES = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "without",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "doesn't",
        "don't",
        "didn't",
        "hasn't",
        "haven't",
        "hadn't",
        "won't",
        "wouldn't",
        "can't",
        "cannot",
        "couldn't",
        "shouldn't",
        "nothing",
        "nobody",
    }
)

#: Word pattern that keeps apostrophes, so ``isn't`` stays one token:
#: ``_TOKEN_RE`` splits it into ``"isn"`` + ``"t"`` and would hide the negation.
_WORD_RE = re.compile(r"[a-z0-9']+")

#: Provider finish reasons that mean "output budget exhausted", i.e. the draft
#: was cut off mid-sentence. A stop reason we do not recognise is treated as a
#: normal end (the cleaning pass still has to cite every delivered sentence).
TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens", "token_limit"})


def refusal_answer(reason: str | None = None) -> str:
    """Deterministic refusal text — same shape every time, no model involved."""
    base = "I can't answer that from the evidence I could verify."
    detail = f" {reason}" if reason else ""
    return f"{base}{detail}\n\nTry naming the project or ticket, or widening the time window."


def _is_factual(sentence: str) -> bool:
    return len(sentence.split()) >= MIN_FACTUAL_WORDS and not sentence.rstrip().endswith(":")


def _strip_invalid_markers(sentence: str, claim_count: int) -> str:
    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        return match.group(0) if 1 <= number <= claim_count else ""

    return _CITATION_RE.sub(replace, sentence)


def _merge_marker_only(sentences: list[str]) -> list[str]:
    """Attach a citation-only fragment to the sentence before it.

    Models write ``"... because the vendor slipped. [1]"``. The marker belongs
    to the previous claim, not to a claim of its own: keeping it separate drops
    the real sentence (it looked uncited) and can leave an answer that is just
    ``"[1]"``.
    """
    merged: list[str] = []
    for sentence in sentences:
        stripped = sentence.strip()
        if not stripped:
            continue
        if merged and not _CITATION_RE.sub("", stripped).strip():
            merged[-1] = f"{merged[-1]} {stripped}"
        else:
            merged.append(stripped)
    return merged


def _strip_meta_citation(line: str) -> str:
    """Remove the model's narration about a marker; keep the marker.

    ``"... admin access issues, as indicated in claim [1]."`` ships as
    ``"... admin access issues [1]."``: the phrase is scaffolding the prompt
    already bans, and it reads as if the answer were quoting an assertion rather
    than citing a source. Text with no such phrase is returned untouched, so the
    punctuation tidy-up cannot rewrite ordinary prose.
    """
    stripped = _META_CITATION_RE.sub(" ", line)
    stripped = _CLAIM_PAREN_RE.sub(
        lambda match: f"[{match.group(1)}]" if match.group(1) else "", stripped
    )
    if stripped == line:
        return line
    # The phrase took its own comma/space with it: tidy what is left dangling.
    stripped = re.sub(r"\s*,\s*(?=\[)", " ", stripped)
    stripped = re.sub(r"\s*,\s*(?=[.;:]|$)", "", stripped)
    stripped = re.sub(r"\s+([.,;:])", r"\1", stripped)
    return " ".join(stripped.split())


def _is_decorative_line(line: str) -> bool:
    """True when a line is nothing but markers and a ``Citations:``-style label.

    Models bolt ``Citations: [1] [2] [3]`` onto a draft that says nothing.
    Marker-bearing lines are normally kept, so the dump must be recognised as
    decoration or it ships as the whole answer.
    """
    stripped = _LABEL_RE.sub("", _CITATION_RE.sub("", line))
    return not _TOKEN_RE.findall(stripped.lower())


def _line_key(line: str) -> str:
    """Comparison key for a line: markers, punctuation and spacing gone, lowercased.

    Punctuation is dropped rather than collapsed because removing a trailing
    marker leaves a stray space before the full stop (``"2026-12-04 [1]."``
    becomes ``"2026-12-04 ."``), which would defeat a whitespace-only compare.
    """
    return " ".join(_TOKEN_RE.findall(_CITATION_RE.sub("", line).lower()))


def _dedupe_lines(lines: list[str]) -> tuple[list[str], list[str]]:
    """Keep the first of each repeated line; return (kept, dropped duplicates)."""
    kept: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    for line in lines:
        key = _line_key(line)
        if key in seen:
            duplicates.append(line)
        else:
            seen.add(key)
            kept.append(line)
    return kept, duplicates


def _clean_answer(
    raw: str, claims: Sequence[Claim]
) -> tuple[str, list[int], list[str], int]:
    """Clean the draft; return (answer, used markers, dropped, implied markers).

    A factual sentence the model left unmarked is cited to the claim it restates
    when ``_implied_marker`` finds one; ``implied`` counts how often that
    happened, so an operator can see the model ignored the marker instruction.
    """
    used: list[int] = []
    implied = 0
    kept_lines: list[str] = []
    dropped: list[str] = []

    for raw_line in raw.splitlines():
        line = _strip_meta_citation(_strip_invalid_markers(raw_line.strip(), len(claims)))
        if not line:
            continue
        if _is_decorative_line(line):
            dropped.append(raw_line.strip())
            continue
        kept: list[str] = []
        for sentence in _merge_marker_only(_SENTENCE_RE.split(line)):
            markers = [int(match) for match in _CITATION_RE.findall(sentence)]
            if markers:
                kept.append(sentence)
                for marker in markers:
                    if marker not in used:
                        used.append(marker)
            elif _is_factual(sentence):
                # A model-written marker always wins; this only fills a gap.
                implied_marker = _implied_marker(sentence, claims)
                if implied_marker is None:
                    dropped.append(sentence)
                else:
                    kept.append(f"{sentence} [{implied_marker}]")
                    implied += 1
                    if implied_marker not in used:
                        used.append(implied_marker)
            else:
                kept.append(sentence)
        if kept:
            kept_lines.append(" ".join(kept))

    kept_lines, duplicates = _dedupe_lines(kept_lines)
    dropped.extend(duplicates)

    answer = "\n".join(kept_lines)
    if not _CITATION_RE.sub("", answer).strip():
        # Nothing but citation markers: there is no answer to show.
        if answer:
            dropped.append(answer)
        return "", [], dropped, implied
    # A dropped duplicate must not leave its claim behind in the citation list.
    cited = {int(match) for match in _CITATION_RE.findall(answer)}
    used = [marker for marker in used if marker in cited]
    return answer, used, dropped, implied


def _renumber(answer: str, numbers: dict[int, int]) -> str:
    """Map markers onto the delivered citation numbers; drop markers without one."""
    def replace(match: re.Match[str]) -> str:
        new = numbers.get(int(match.group(1)))
        return f"[{new}]" if new is not None else ""

    return _CITATION_RE.sub(replace, answer)


def _token_set(text: str) -> set[str]:
    """Lowercase alphanumeric words — punctuation-free, dates keep their parts."""
    return set(_TOKEN_RE.findall(text.lower()))


def _best_supporting_chunk(claim: Claim, by_id: dict[str, ScrubbedChunk]) -> ScrubbedChunk | None:
    """Chunk in ``claim.chunk_ids`` whose text best supports the claim text.

    Picking the first listed id is wrong: a claim about the *moved* cutover date
    (2026-12-04, from the email) can list the Jira comment saying 2026-11-20
    first, so the answer cited the value it contradicted.

    Score is containment of the claim's word set in the chunk (overlap / claim
    words, in 0..1), so a chunk that actually states the claim's values outranks
    a longer chunk that merely mentions the same ticket. Equal scores keep
    ``chunk_ids`` order (strict ``>``); a claim with no overlap still gets its
    first available candidate rather than no citation.
    """
    claim_tokens = _token_set(claim.text)
    best: ScrubbedChunk | None = None
    best_score = -1.0
    for chunk_id in claim.chunk_ids:
        chunk = by_id.get(chunk_id)
        if chunk is None:
            continue
        overlap = len(claim_tokens & _token_set(chunk.text))
        score = overlap / len(claim_tokens) if claim_tokens else 0.0
        if score > best_score:
            best = chunk
            best_score = score
    return best


def _claim_markers(claims: Sequence[Claim]) -> dict[int, Claim]:
    """Citation marker -> claim, for the numbering the model was shown.

    ``prompts.build_user_prompt`` numbers the claims ``[1]..[n]`` in list order,
    so a marker is a *position* and not a ``claim_id``. Resolving through this
    map (instead of ``claims[marker - 1]``) keeps a non-contiguous ``claim_id``
    from being read as an index; the delivered ``used_claim_ids`` come from
    ``Claim.claim_id``, never from arithmetic on the marker.
    """
    return {index: claim for index, claim in enumerate(claims, start=1)}


def _is_negated(text: str) -> bool:
    """True when ``text`` states an explicit negation ("not", "isn't", "never").

    Catches explicit cues only: a claim saying "delayed" against a sentence
    saying "on time" is semantically negated and reads as positive here.
    """
    words = _WORD_RE.findall(text.lower().replace("\u2019", "'"))
    return any(word in _NEGATION_CUES or word.endswith("n't") for word in words)


def _implied_marker(sentence: str, claims: Sequence[Claim]) -> int | None:
    """Marker position of the verified claim a marker-less sentence restates.

    The best claim is the one holding the largest share of the sentence's words
    (``overlap / sentence words``), and only counts when that share is at least
    ``MIN_IMPLIED_CITE_OVERLAP``. Two guards then check what overlap cannot see.
    Numeric: the sentence carries a number the claim does not state, so
    ``2026-12-04`` against ``2026-11-20`` is a contradiction, not a paraphrase.
    Polarity: an explicit negation on one side but not the other makes the
    sentence say the inverse of its claim. Ties keep claim order (strict ``>``).
    """
    sentence_tokens = _token_set(sentence)
    if not sentence_tokens:
        return None
    best_marker: int | None = None
    best_claim: Claim | None = None
    best_overlap = 0.0
    for marker, claim in _claim_markers(claims).items():
        overlap = len(sentence_tokens & _token_set(claim.text)) / len(sentence_tokens)
        if overlap > best_overlap:
            best_marker, best_claim, best_overlap = marker, claim, overlap
    if best_marker is None or best_claim is None or best_overlap < MIN_IMPLIED_CITE_OVERLAP:
        return None
    numbers = {token for token in sentence_tokens if any(char.isdigit() for char in token)}
    if not numbers <= _token_set(best_claim.text):
        return None
    if _is_negated(sentence) != _is_negated(best_claim.text):
        return None
    return best_marker


def _fixed_windows(text: str) -> list[str]:
    """Split text into ``QUOTE_CHARS``-wide windows cut at word boundaries."""
    windows: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + QUOTE_CHARS, len(text))
        if end < len(text):
            # Never ship half a word: back off to the last space inside the window.
            space = text.rfind(" ", start + 1, end)
            if space > start:
                end = space
        window = text[start:end].strip()
        if window:
            windows.append(window)
        start = end
    return windows


def _quote_units(text: str) -> list[str]:
    """Units the quote window slides over: sentences, else fixed character windows.

    A sentence is what a reader (and the lexical judge) treats as support, so a
    window starting on the sentence that states the claim's values can be quoted
    whole. An unusually long sentence, or a chunk with no sentence break at all,
    falls back to fixed windows so every unit fits in ``QUOTE_CHARS``.
    """
    sentences = [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]
    if len(sentences) <= 1:
        return _fixed_windows(text)
    units: list[str] = []
    for sentence in sentences:
        if len(sentence) <= QUOTE_CHARS:
            units.append(sentence)
        else:
            units.extend(_fixed_windows(sentence))
    return units


def _quote_for(chunk_text: str, claim_text: str) -> str:
    """Quote for a citation: the window of the chunk that best supports the claim.

    A blind prefix fails when the claim's support sits deeper than
    ``QUOTE_CHARS``: the delivered quote then omits it, which misleads the reader
    and makes a lexical grounding judge score the claim zero (the live eval did
    exactly that). Windows are scored by how many of the claim's words they
    contain; ties keep the earliest window (strict ``>``), and a chunk sharing
    nothing with the claim keeps the old prefix behaviour.
    """
    units = _quote_units(chunk_text)
    claim_tokens = _token_set(claim_text)
    best = ""
    best_score = 0
    for start in range(len(units)):
        window = ""
        for unit in units[start:]:
            candidate = f"{window} {unit}" if window else unit
            if len(candidate) > QUOTE_CHARS:
                break
            window = candidate
            score = len(claim_tokens & _token_set(window))
            if score > best_score:
                best = window
                best_score = score
    if not best:
        return chunk_text[:QUOTE_CHARS]
    return best


def _citations_for(
    used: Sequence[int], markers: dict[int, Claim], by_id: dict[str, ScrubbedChunk]
) -> tuple[list[Citation], dict[int, int]]:
    """Delivered citations, plus marker -> 1-based citation number.

    Citations are deduplicated by chunk id, preserving first-use order: two
    claims resting on the same chunk are one delivered citation, and every
    marker pointing at that chunk is renumbered onto that single entry. A marker
    whose claim has no chunk in the pack gets no entry at all, so ``_renumber``
    removes it rather than leaving a dangling reference.
    """
    citations: list[Citation] = []
    numbers: dict[int, int] = {}
    by_chunk: dict[str, int] = {}
    for marker in used:
        claim = markers.get(marker)
        if claim is None:
            continue
        chunk = _best_supporting_chunk(claim, by_id)
        if chunk is None:
            continue
        number = by_chunk.get(chunk.chunk_id)
        if number is None:
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source=chunk.source,
                    title=chunk.title,
                    quote=_quote_for(chunk.text, claim.text),
                    url=chunk.url,
                )
            )
            number = len(citations)
            by_chunk[chunk.chunk_id] = number
        numbers[marker] = number
    return citations, numbers


def _truncated(finish_reason: str | None) -> bool:
    """True when the provider stopped at the output-token limit, not at an end."""
    return (finish_reason or "").strip().lower() in TRUNCATED_FINISH_REASONS


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


async def synthesize(
    question: str,
    verified: VerifiedClaims,
    pack: ScrubbedPack,
    *,
    trace_id: str | None = None,
    client: LLMClient | None = None,
    sink: EventSink | None = None,
    settings: Settings | None = None,
) -> SynthesisResult:
    """Write the user-facing answer from ``verified`` claims only.

    ``pack`` supplies the citation metadata (title, url, quote); claims were
    already validated against those chunk ids by the Evidence Agent.
    """
    config = settings or get_settings()
    tracer = Tracer(sink or NullEventSink(), trace_id)
    asked = question.strip() or verified.question

    def refuse(detail: str | None, *, refusal_reason: str | None = None) -> SynthesisResult:
        """Refusal with user-visible text ``detail`` and the typed ``refusal_reason``.

        ``refusal_reason`` defaults to ``detail``; the truncation and cleanup
        paths override it where the user-facing sentence is not the reason a
        trace reports (``"The draft was cut off..."`` vs ``"answer truncated by
        the token limit"``).
        """
        return SynthesisResult(
            answer=refusal_answer(detail),
            citations=[],
            used_claim_ids=[],
            refused=True,
            refusal_reason=refusal_reason if refusal_reason is not None else detail,
        )

    if not verified.can_answer or not verified.claims:
        reason = verified.insufficiency_reason or "no verified claims were available"
        await tracer.emit(AGENT, "synthesis.refused", {"reason": reason[:500], "question": asked})
        return refuse(reason, refusal_reason=f"insufficient evidence: {reason}")

    by_id = {chunk.chunk_id: chunk for chunk in pack.chunks}
    # Synthesis is prose, but a reasoning model still wastes the budget thinking.
    model = config.llm_json_model or config.llm_model

    owns_client = client is None
    llm: LLMClient = client or _default_client(config)
    try:
        response = await llm.complete(
            CompletionRequest(
                messages=build_messages(asked, verified.claims),
                model=model,
                temperature=0.0,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
        )
    finally:
        if owns_client:
            await llm.aclose()

    if _truncated(response.finish_reason):
        # A cut-off answer can drop its citation or negate the claim it was
        # making, and groundedness cannot detect truncation (ruling #17).
        await tracer.emit(
            AGENT,
            "synthesis.refused",
            {
                "reason": "answer truncated by the token limit",
                "finish_reason": response.finish_reason,
                "question": asked,
            },
        )
        return refuse(
            "The draft was cut off by the output-token limit.",
            refusal_reason="answer truncated by the token limit",
        )

    markers = _claim_markers(verified.claims)
    cleaned, used, dropped, implied = _clean_answer(response.text, verified.claims)
    citations, numbers = _citations_for(used, markers, by_id)
    if not cleaned or not used or not citations:
        await tracer.emit(
            AGENT,
            "synthesis.refused",
            {
                "reason": "answer had no supported, cited sentences",
                "dropped_sentences": len(dropped),
                "question": asked,
            },
        )
        return refuse(
            "The model's draft could not be tied back to the evidence.",
            refusal_reason="the draft answer could not be tied back to the evidence",
        )

    answer = _renumber(cleaned, numbers)
    await tracer.emit(
        AGENT,
        "synthesis.completed",
        {
            "answer_chars": len(answer),
            "citations": len(citations),
            "dropped_sentences": len(dropped),
            # Markers attached by matching, not written by the model: non-zero
            # means the draft ignored the citation instruction.
            "implied_citations": implied,
            "claims_available": len(verified.claims),
        },
    )
    return SynthesisResult(
        answer=answer,
        citations=citations,
        # A claim that shares its chunk with an earlier one still appears here:
        # it supported the answer even though its citation was deduplicated.
        used_claim_ids=[markers[marker].claim_id for marker in used if marker in numbers],
        refused=False,
    )
