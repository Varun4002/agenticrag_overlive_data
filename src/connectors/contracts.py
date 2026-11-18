"""Cross-source sync contracts (the Connector Agent's typed interface).

These are the models other agents and the CLI depend on; they are additive to
the frozen contracts in ``orchestrator/contracts.py`` (which this component does
not touch). Shape follows ``docs/agents/01-connector-agent.md``:

    SyncResult: source, documents_upserted, watermark, errors

Extra fields below (``documents_unchanged``, ``documents_retired``, ``full``,
timing) are observability additions — they never change the four required ones.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from connectors.normalizer import CanonicalDoc

SourceName = Literal["jira", "notion", "gmail"]


class SyncRequest(BaseModel):
    """What to sync. Incremental unless ``full``."""

    model_config = ConfigDict(extra="forbid")

    source: SourceName
    full: bool = False


class ErrorReport(BaseModel):
    """One failed item or fetch, collected instead of raised.

    Never contains response bodies wholesale — just a short message — so a sync
    cannot dump source data (or a token echoed by a broken proxy) into logs.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    # None when the failure was not attributable to one item (e.g. a page fetch).
    source_id: str | None = None
    # Coarse phase: "fetch" / "normalize" / "persist" / "comments".
    stage: str
    error_type: str
    message: str
    retryable: bool = False


class SyncResult(BaseModel):
    """Outcome of one sync run. Partial failures are reported, not raised."""

    model_config = ConfigDict(extra="forbid")

    source: str
    documents_upserted: int = 0
    watermark: AwareDatetime
    errors: list[ErrorReport] = Field(default_factory=list)

    # --- observability additions ---------------------------------------
    documents_unchanged: int = 0
    documents_retired: int = 0
    # Items returned by the source but outside the configured allowlist.
    documents_skipped: int = 0
    full: bool = False
    started_at: AwareDatetime | None = None
    finished_at: AwareDatetime | None = None
    requests: int = 0
    retries: int = 0
    # Audit id of this run in ``audit.events`` (set by connectors.service.sync).
    trace_id: str | None = None

    @property
    def ok(self) -> bool:
        """True when the run had no collected errors."""
        return not self.errors


class Connector(Protocol):
    """The interface the service (and later the orchestrator) depends on."""

    source: str

    async def sync(self, req: SyncRequest) -> SyncResult:
        """Pull the allowlisted scope and persist canonical documents."""
        ...

    async def fetch_targeted(
        self,
        *,
        source_id: str | None = None,
        query: str | None = None,
        limit: int = 10,
    ) -> list[CanonicalDoc]:
        """Ad-hoc fetch for freshness ("due today") — returns docs, writes nothing."""
        ...

    async def aclose(self) -> None:
        """Release transport resources."""
        ...


def result_timestamp(value: datetime) -> AwareDatetime:
    """Narrow a datetime for the ``AwareDatetime`` fields (mypy-friendly)."""
    if value.tzinfo is None:
        raise ValueError("sync timestamps must be timezone-aware")
    return value


__all__ = [
    "Connector",
    "ErrorReport",
    "SourceName",
    "SyncRequest",
    "SyncResult",
    "result_timestamp",
]
