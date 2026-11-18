"""Pure parsing of Evidence-Agent model output.

The local model (``qwen3.5:4b``) is a reasoning model: its responses routinely
carry `` thinking`` blocks, markdown fences, and prose around the JSON payload.
This module turns that raw text into validated :class:`ExtractedClaim` objects
and nothing else — no I/O, no logging, no retries.

Failure policy is strict because the caller fails closed: text that does not
yield a schema-valid ``{"claims": [...]}`` payload raises
:class:`EvidenceParseError`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

__all__ = [
    "EvidenceParseError",
    "ExtractedClaim",
    "ExtractionResult",
    "extract_json_object",
    "parse_extraction",
]

_CLOSERS = frozenset({"}","]"})
_BRACKETS = {"{": "}", "[": "]"}
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_FENCE_RE = re.compile(r"```[ \t]*(?:json)?[ \t]*\r?\n?(.*?)```", re.DOTALL)
_ID_SPLIT_RE = re.compile(r"[,\s]+")


class EvidenceParseError(ValueError):
    """Model output could not be parsed into the extraction schema."""


class ExtractedClaim(BaseModel):
    """One model-proposed claim, before citation/support validation."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    text: str = Field(min_length=1)
    chunk_ids: list[str] = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Only used for contradiction detection; the model may omit them.
    subject: str | None = None
    attribute: str | None = None
    value: str | None = None

    @field_validator("chunk_ids", mode="before")
    @classmethod
    def _split_chunk_ids(cls, value: object) -> object:
        """Accept ``"a, b"`` and ``["a", "b"]``; drop blanks."""
        if isinstance(value, str):
            return [part for part in _ID_SPLIT_RE.split(value) if part]
        if isinstance(value, list):
            cleaned: list[str] = []
            for item in value:
                if item is None:
                    continue
                text = item if isinstance(item, str) else str(item)
                if text.strip():
                    cleaned.append(text.strip())
            return cleaned
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: object) -> float:
        """Clamp hallucinated confidence values instead of failing the claim."""
        if isinstance(value, bool):
            return 0.5
        if isinstance(value, int | float):
            return min(1.0, max(0.0, float(value)))
        if isinstance(value, str):
            raw = value.strip()
            percent = raw.endswith("%")
            try:
                number = float(raw.rstrip("%").strip())
            except ValueError:
                return 0.5
            if percent:
                number /= 100.0
            return min(1.0, max(0.0, number))
        return 0.5


class ExtractionResult(BaseModel):
    """A schema-valid extraction payload (possibly with zero claims)."""

    claims: list[ExtractedClaim] = Field(default_factory=list)


def _candidates(text: str) -> Iterator[str]:
    """Yield plausible payload regions: fenced blocks first, then everything."""
    cleaned = _THINK_BLOCK_RE.sub(" ", text)
    cleaned = _THINK_OPEN_RE.sub(" ", cleaned)
    for match in _FENCE_RE.finditer(cleaned):
        yield match.group(1)
    yield cleaned


def _json_spans(text: str) -> Iterator[str]:
    """Yield balanced top-level ``{...}`` / ``[...]`` spans, string-aware."""
    stack: list[str] = []
    start = -1
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            if stack:
                in_string = True
            continue

        closer = _BRACKETS.get(char)
        if closer is not None:
            if not stack:
                start = index
            stack.append(closer)
            continue

        if char in _CLOSERS and stack:
            if char != stack[-1]:
                # Mismatched nesting: abandon this span and keep scanning.
                stack.clear()
                start = -1
                continue
            stack.pop()
            if not stack and start >= 0:
                yield text[start : index + 1]
                start = -1


def _load_payload(text: str) -> object:
    """Return the first JSON payload that looks like an extraction result."""
    fallback: object | None = None
    for candidate in _candidates(text):
        for span in _json_spans(candidate):
            try:
                value: Any = json.loads(span)
            except ValueError:
                continue
            if isinstance(value, dict):
                if "claims" in value:
                    return value
                if fallback is None:
                    fallback = value
            elif isinstance(value, list):
                if value and all(isinstance(item, dict) and "text" in item for item in value):
                    return {"claims": value}
                if fallback is None:
                    fallback = value
    if fallback is not None:
        return fallback
    raise EvidenceParseError("model output contained no JSON object or array")


def _normalise_claim(raw: object, index: int) -> dict[str, Any]:
    """Coerce one claim object into something the schema can validate."""
    if not isinstance(raw, dict):
        raise EvidenceParseError(f"claim {index} is not a JSON object")
    data: dict[str, Any] = {str(key): value for key, value in raw.items()}
    if "chunk_ids" not in data and data.get("chunk_id") is not None:
        data["chunk_ids"] = data["chunk_id"]
    for key in ("text", "subject", "attribute", "value"):
        value = data.get(key)
        if value is not None and not isinstance(value, str):
            data[key] = str(value)
    if data.get("confidence") is None:
        data.pop("confidence", None)
    return data


def _brief_error(exc: ValidationError) -> str:
    first = exc.errors()[0]
    location = ".".join(str(part) for part in first.get("loc", ()))
    return f"{location or 'claim'}: {first.get('msg', 'invalid value')}"


def extract_json_object(text: str) -> dict[str, Any]:
    """Return the payload as a dict; a bare claim array is wrapped."""
    payload = _load_payload(text)
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return {"claims": payload}
    # Unreachable by construction: _load_payload only returns dicts/lists.
    raise EvidenceParseError("model output was not a JSON object or array")


def parse_extraction(text: str) -> ExtractionResult:
    """Parse and validate a model response.

    Raises:
        EvidenceParseError: no payload, missing ``claims`` key, or a claim that
            fails schema validation. Callers treat this as "extraction failed"
            and fail closed.
    """
    payload = extract_json_object(text)
    if "claims" not in payload:
        raise EvidenceParseError("payload is missing the 'claims' key")
    raw_claims = payload["claims"]
    if not isinstance(raw_claims, list):
        raise EvidenceParseError("'claims' must be a JSON array")

    claims: list[ExtractedClaim] = []
    for index, raw in enumerate(raw_claims):
        try:
            claims.append(ExtractedClaim.model_validate(_normalise_claim(raw, index)))
        except ValidationError as exc:
            raise EvidenceParseError(f"claim {index} invalid — {_brief_error(exc)}") from exc
    return ExtractionResult(claims=claims)
