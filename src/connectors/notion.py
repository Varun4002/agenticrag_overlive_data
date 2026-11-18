"""Notion connector stub (M2).

M2 will implement: page + database-entry fetch by ``last_edited_time``, section
documents for long pages, workspace + explicitly-shared-user ACL mirroring.
Nothing here may touch the network yet — the class exists so the service can
dispatch on ``source`` and fail with one clear message instead of an
``AttributeError``.
"""

from __future__ import annotations

from connectors.contracts import SyncRequest, SyncResult
from connectors.normalizer import CanonicalDoc


class NotionConnector:
    """Placeholder that refuses to fetch until M2."""

    source = "notion"

    def __init__(self, *, database_ids: tuple[str, ...] = ()) -> None:
        self.database_ids = database_ids

    async def sync(self, req: SyncRequest) -> SyncResult:
        raise NotImplementedError(
            "Notion connector lands in M2 (docs/03-mvp-plan.md): page/database sync by "
            "last_edited_time, section documents, workspace ACL mirror. "
            "No request was made and no documents were written."
        )

    async def fetch_targeted(
        self,
        *,
        source_id: str | None = None,
        query: str | None = None,
        limit: int = 10,
    ) -> list[CanonicalDoc]:
        raise NotImplementedError("Notion targeted fetch lands in M2. No request was made.")

    async def aclose(self) -> None:
        """Nothing to close: this stub owns no transport."""
        return None


__all__ = ["NotionConnector"]
