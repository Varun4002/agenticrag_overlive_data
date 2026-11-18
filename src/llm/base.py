"""Provider-agnostic LLM interface and its data models.

Every provider adapter implements :class:`LLMClient`. Callers (planner,
evidence agent, synthesizer) depend only on this module, so swapping providers
never touches agent code.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Role(enum.StrEnum):
    """Chat message roles."""

    system = "system"
    user = "user"
    assistant = "assistant"


class Message(BaseModel):
    """One chat message."""

    role: Role
    content: str


class CompletionRequest(BaseModel):
    """A single completion request."""

    messages: list[Message] = Field(min_length=1)
    model: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    # Free-form passthrough for provider-specific knobs (e.g. response_format).
    extra: dict[str, object] = Field(default_factory=dict)


class Usage(BaseModel):
    """Token counts reported by the provider."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResponse(BaseModel):
    """Normalized completion result."""

    text: str
    model: str
    usage: Usage
    latency_ms: int
    finish_reason: str | None = None
    provider_request_id: str | None = None


@dataclass(slots=True)
class TokenLedger:
    """Accumulates token usage across calls for one trace/request.

    Providers report usage per call; budgets and traces need the request total,
    which is why the client records into a ledger the caller owns.
    """

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    per_model: dict[str, int] = field(default_factory=dict)

    def record(self, model: str, usage: Usage) -> None:
        self.calls += 1
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.per_model[model] = self.per_model.get(model, 0) + usage.total_tokens

    def as_payload(self) -> dict[str, object]:
        """JSON-serializable snapshot for an ``audit.events`` payload."""
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "per_model": dict(self.per_model),
        }


@runtime_checkable
class LLMClient(Protocol):
    """The only interface agents may depend on."""

    provider: str

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Run one completion, raising an :class:`LLMError` subclass on failure."""
        ...

    async def aclose(self) -> None:
        """Release transport resources."""
        ...
