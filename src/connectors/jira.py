"""Jira Cloud REST API v3 adapter (read-only).

What it does, and nothing else:

- full sync of the allowlisted projects (JQL), incremental sync by the ``updated``
  cursor kept in ``core.sync_state``;
- one canonical document per issue plus one per comment;
- ACL mirror (project group + assignee/reporter/comment author identifiers);
- ``content_hash`` change detection (unchanged documents are skipped);
- supersession/deletion detection: vanished items become ``is_current=false``,
  nothing is hard-deleted;
- ``fetch_targeted`` for the planner's on-demand freshness fetches.

Every method here is a ``GET``. Credentials come from the environment
(``connectors.settings``) and are never logged.
"""

from __future__ import annotations

import base64
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from connectors.base import (
    BaseConnector,
    ConnectorAuthError,
    ConnectorConfigError,
    ConnectorError,
    ConnectorRateBudgetExhausted,
    ConnectorResponseError,
    Pagination,
    RateBudget,
)
from connectors.contracts import ErrorReport, SyncRequest, SyncResult, result_timestamp
from connectors.normalizer import (
    CanonicalDoc,
    DocumentStore,
    SyncStateRecord,
    UpsertOutcome,
    apply_document,
    jira_issue_to_docs,
    parse_jira_timestamp,
    retire_missing,
)
from connectors.settings import ConnectorSettings, JiraCredentials, get_connector_settings
from db.models import Source

logger = logging.getLogger(__name__)

# Fields we ask Jira for. Kept explicit: the default payload is much larger and the
# sync only needs these (docs/agents/01-connector-agent.md, normalization table).
ISSUE_FIELDS = (
    "summary",
    "description",
    "status",
    "assignee",
    "reporter",
    "creator",
    "duedate",
    "priority",
    "labels",
    "issuetype",
    "project",
    "resolution",
    "created",
    "updated",
)

# Jira project keys: uppercase letter first, then uppercase alphanumerics/underscore.
_PROJECT_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,49}$")
# Issue keys are "<PROJECT>-<number>"; enforced before interpolating into a path.
_ISSUE_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,49}-\d+$")

COMMENT_PAGINATION = Pagination(items_key="comments")


class JiraConnector(BaseConnector):
    """Jira Cloud issues + comments -> ``core.documents``."""

    source = "jira"

    def __init__(
        self,
        *,
        store: DocumentStore | None = None,
        settings: ConnectorSettings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        config = settings or get_connector_settings()
        credentials = config.require_jira()
        super().__init__(
            base_url=credentials.base_url,
            headers=_auth_headers(credentials),
            timeout_s=config.jira_request_timeout_s,
            max_retries=config.jira_max_retries,
            backoff_base_s=config.jira_backoff_base_s,
            budget=RateBudget(
                max_calls=config.jira_rate_budget_calls,
                window_s=config.jira_rate_budget_window_s,
            ),
            transport=transport,
            sleep=sleep,
        )
        self._config = config
        self._credentials = credentials
        # Validate once, at construction: project keys are interpolated into JQL.
        self.projects = validate_project_keys(credentials.projects)
        self._store = store
        self._page_size = config.jira_page_size
        # Injectable for deterministic tests.
        self._now = now or (lambda: datetime.now(UTC))

    # --- sync -----------------------------------------------------------
    async def sync(self, req: SyncRequest) -> SyncResult:
        """Pull the allowlisted projects; collect per-item failures, never abort on them."""
        store = self._store
        if store is None:
            raise ConnectorConfigError(
                "JiraConnector.sync needs a DocumentStore; construct it with store=..."
            )

        started_at = self._now()
        state = await store.load_sync_state(self.source)
        full = req.full or state.watermark is None
        jql = self._build_jql(None if full else state.watermark)

        seen: set[str] = set()
        failed: set[str] = set()
        errors: list[ErrorReport] = []
        counts = {outcome: 0 for outcome in UpsertOutcome}
        retired = 0
        skipped = 0
        interrupted = False
        watermark = state.watermark
        earliest_failure: datetime | None = None

        logger.info(
            "jira sync starting (full=%s, projects=%s, since=%s)",
            full,
            ",".join(self.projects),
            None if full else state.watermark,
        )

        try:
            async for issue in self._iter_issues(jql):
                key = issue.get("key")
                if not isinstance(key, str) or not key:
                    errors.append(
                        _error(
                            stage="normalize",
                            error_type="MalformedItem",
                            message="search result had no 'key'",
                        )
                    )
                    continue

                project_key = _issue_project_key(issue)
                if project_key is not None and project_key not in self.projects:
                    # JQL already scopes the search; this is the belt-and-braces
                    # check that no out-of-scope issue is ever ingested.
                    logger.info(
                        "jira: skipping %s (project %s is not allowlisted)", key, project_key
                    )
                    skipped += 1
                    continue

                try:
                    item_counts, item_retired = await self._sync_issue(issue, key)
                except ConnectorRateBudgetExhausted:
                    # Stop fetching; the partial result is reported below.
                    raise
                except ConnectorAuthError:
                    # Credentials are broken: retrying items cannot help.
                    raise
                except (ConnectorError, ValidationError, ValueError, KeyError) as exc:
                    errors.append(
                        _error(
                            source_id=key,
                            stage=_stage_for(exc),
                            error_type=type(exc).__name__,
                            message=_short(exc),
                            retryable=getattr(exc, "retryable", False),
                        )
                    )
                    failed.add(key)
                    earliest_failure = _earliest(earliest_failure, _issue_updated(issue))
                    continue
                except SQLAlchemyError as exc:
                    errors.append(
                        _error(
                            source_id=key,
                            stage="persist",
                            error_type=type(exc).__name__,
                            message=_short(exc),
                        )
                    )
                    failed.add(key)
                    earliest_failure = _earliest(earliest_failure, _issue_updated(issue))
                    continue

                seen.add(key)
                for outcome, count in item_counts.items():
                    counts[outcome] += count
                retired += item_retired
                watermark = _latest(watermark, _issue_updated(issue))

        except ConnectorRateBudgetExhausted as exc:
            interrupted = True
            errors.append(
                _error(
                    stage="fetch",
                    error_type=type(exc).__name__,
                    message=_short(exc),
                    retryable=False,
                )
            )
            logger.warning("jira sync stopped early: %s", exc)
        except ConnectorAuthError as exc:
            await self._record_failure(store, state, started_at, exc)
            raise
        except ConnectorError as exc:
            interrupted = True
            errors.append(
                _error(
                    stage="fetch",
                    error_type=type(exc).__name__,
                    message=_short(exc),
                    retryable=exc.retryable,
                )
            )
            logger.warning("jira sync interrupted: %s", exc)

        if full and not interrupted:
            # A completion-guarantee: only a full run that read every page can prove
            # an issue was deleted. Items that errored are excluded, or a transient
            # failure would look like a deletion.
            try:
                retired += len(
                    await retire_missing(store, source=Source.jira, seen=seen | failed)
                )
            except SQLAlchemyError as exc:
                errors.append(
                    _error(
                        stage="persist",
                        error_type=type(exc).__name__,
                        message=_short(exc),
                    )
                )

        finished_at = self._now()
        new_watermark = self._resolve_watermark(
            watermark=watermark, failed_updated=earliest_failure, fallback=started_at
        )
        status = "ok" if not errors else "partial"
        await store.save_sync_state(
            _state_after_run(self.source, new_watermark, started_at, status, errors)
        )

        result = SyncResult(
            source=self.source,
            documents_upserted=counts[UpsertOutcome.created] + counts[UpsertOutcome.updated],
            watermark=result_timestamp(new_watermark),
            errors=errors,
            documents_unchanged=counts[UpsertOutcome.unchanged],
            documents_retired=retired,
            documents_skipped=skipped,
            full=full,
            started_at=result_timestamp(started_at),
            finished_at=result_timestamp(finished_at),
            requests=self.stats.requests,
            retries=self.stats.retries,
        )
        logger.info(
            "jira sync finished (full=%s, upserted=%d, unchanged=%d, retired=%d, errors=%d)",
            full,
            result.documents_upserted,
            result.documents_unchanged,
            result.documents_retired,
            len(result.errors),
        )
        return result

    async def _sync_issue(
        self, issue: Mapping[str, Any], key: str
    ) -> tuple[dict[UpsertOutcome, int], int]:
        """Normalize + persist one issue and its comments inside a savepoint."""
        store = self._store
        assert store is not None  # guarded by sync()

        counts = {outcome: 0 for outcome in UpsertOutcome}
        async with store.unit_of_work():
            comments = await self._fetch_comments(key)
            docs = jira_issue_to_docs(
                issue,
                base_url=self._credentials.base_url,
                fetched_at=self._now(),
                comments=comments,
            )
            for doc in docs:
                record = await apply_document(store, doc)
                counts[record.outcome] += 1

            # A comment deleted from the issue must stop being current.
            comment_ids = {doc.source_id for doc in docs if doc.parent_source_id == key}
            retired = await retire_missing(
                store, source=Source.jira, seen=comment_ids, parent_source_id=key
            )
        return counts, len(retired)

    # --- fetching -------------------------------------------------------
    async def _iter_issues(self, jql: str) -> AsyncIterator[Mapping[str, Any]]:
        """Stream issues for ``jql``.

        Jira Cloud is mid-migration between the classic offset search
        (``startAt``/``total``) and the token search (``nextPageToken``/``isLast``),
        and which one a tenant serves depends on its rollout state. Rather than
        betting on one, this follows whatever pagination the response advertises.
        """
        params: dict[str, object] = {
            "jql": jql,
            "fields": ",".join(ISSUE_FIELDS),
            "maxResults": self._page_size,
        }
        path = self._config.jira_search_path
        token: str | None = None
        start = 0
        while True:
            query = dict(params)
            if token is not None:
                query["nextPageToken"] = token
            elif start:
                query["startAt"] = start

            payload = await self.get_json(path, query)
            issues = payload.get("issues")
            if issues is None:
                raise ConnectorResponseError(
                    "search response had no 'issues' list", source=self.source
                )
            if not isinstance(issues, list):
                raise ConnectorResponseError(
                    "search response had no 'issues' list", source=self.source
                )

            for issue in issues:
                if not isinstance(issue, Mapping):
                    raise ConnectorResponseError(
                        "search result item was not an object", source=self.source
                    )
                yield issue

            next_token = payload.get("nextPageToken")
            if isinstance(next_token, str) and next_token:
                token = next_token
                continue
            total = payload.get("total")
            if isinstance(total, int):
                start += len(issues)
                if not issues or start >= total:
                    return
                continue
            if payload.get("isLast") is False and issues:
                raise ConnectorResponseError(
                    "search response indicated more pages but supplied no cursor",
                    source=self.source,
                )
            return

    async def _fetch_comments(self, key: str) -> list[dict[str, Any]]:
        """All comments of one issue (offset paginated)."""
        comments: list[dict[str, Any]] = []
        async for comment in self.paginate(
            f"/rest/api/3/issue/{key}/comment",
            pagination=COMMENT_PAGINATION,
            page_size=self._page_size,
        ):
            comments.append(comment)
        return comments

    async def _fetch_issue(self, key: str) -> dict[str, Any]:
        return await self.get_json(f"/rest/api/3/issue/{key}", {"fields": ",".join(ISSUE_FIELDS)})

    # --- targeted fetches -----------------------------------------------
    async def fetch_targeted(
        self,
        *,
        source_id: str | None = None,
        query: str | None = None,
        limit: int = 10,
    ) -> list[CanonicalDoc]:
        """Fetch fresh documents on demand. Writes nothing.

        ``source_id`` fetches one issue (and its comments); ``query`` runs a JQL
        fragment scoped to the allowlisted projects (e.g. ``duedate <= endOfWeek()``).
        """
        fetched_at = self._now()
        if source_id is not None:
            key = _validate_issue_key(source_id)
            issue = await self._fetch_issue(key)
            comments = await self._fetch_comments(key)
            return jira_issue_to_docs(
                issue,
                base_url=self._credentials.base_url,
                fetched_at=fetched_at,
                comments=comments,
            )

        if query is None:
            raise ConnectorConfigError("fetch_targeted needs either source_id or query")

        docs: list[CanonicalDoc] = []
        async for search_hit in self._iter_issues(self._build_jql(None, extra=query)):
            hit_key = search_hit.get("key")
            if not isinstance(hit_key, str) or not hit_key:
                continue
            comments = await self._fetch_comments(hit_key)
            docs.extend(
                jira_issue_to_docs(
                    search_hit,
                    base_url=self._credentials.base_url,
                    fetched_at=fetched_at,
                    comments=comments,
                )
            )
            if len(docs) >= limit:
                break
        return docs[:limit]

    # --- query building -------------------------------------------------
    def _build_jql(self, since: datetime | None, *, extra: str | None = None) -> str:
        """Scope + cursor + optional extra constraint, always ordered by ``updated``."""
        quoted = ", ".join(f'"{key}"' for key in sorted(self.projects))
        clauses = [f"project IN ({quoted})"]
        if since is not None:
            # JQL takes minute precision; ``>=`` is inclusive, so the boundary issue
            # is re-fetched and then skipped by content_hash.
            clauses.append(f'updated >= "{since.astimezone(UTC).strftime("%Y-%m-%d %H:%M")}"')
        extra_clause = extra if extra is not None else self._config.jira_jql_extra
        if extra_clause:
            clauses.append(f"({extra_clause})")
        return " AND ".join(clauses) + " ORDER BY updated ASC"

    # --- sync-state helpers ---------------------------------------------
    def _resolve_watermark(
        self, *, watermark: datetime | None, failed_updated: datetime | None, fallback: datetime
    ) -> datetime:
        """New cursor.

        When an item failed, the cursor must not move past it or its next change
        would be skipped forever: clamp to just before the earliest failure.
        """
        candidate = watermark or fallback
        if failed_updated is not None:
            clamped = failed_updated - timedelta(seconds=1)
            if clamped < candidate:
                candidate = clamped
        return candidate

    async def _record_failure(
        self,
        store: DocumentStore,
        state: SyncStateRecord,
        started_at: datetime,
        exc: ConnectorError,
    ) -> None:
        """Persist a terminal failure so the next run's status is explainable."""
        await store.save_sync_state(
            _state_after_run(
                self.source,
                state.watermark,
                started_at,
                "error",
                [_error(stage="fetch", error_type=type(exc).__name__, message=_short(exc))],
            )
        )


def _auth_headers(credentials: JiraCredentials) -> dict[str, str]:
    """HTTP Basic auth: ``email:api_token``. Never logged, never stored."""
    raw = f"{credentials.email}:{credentials.api_token.get_secret_value()}".encode()
    token = base64.b64encode(raw).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def _validate_issue_key(key: str) -> str:
    normalised = key.strip().upper()
    if not _ISSUE_KEY_RE.match(normalised):
        raise ConnectorConfigError(f"invalid Jira issue key: {key!r}")
    return normalised


def validate_project_keys(keys: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Reject anything that is not a plain project key (JQL injection guard)."""
    normalised = []
    for key in keys:
        candidate = key.strip().upper()
        if not _PROJECT_KEY_RE.match(candidate):
            raise ConnectorConfigError(f"invalid Jira project key: {key!r}")
        normalised.append(candidate)
    return tuple(normalised)


def _state_after_run(
    source: str,
    watermark: datetime | None,
    started_at: datetime,
    status: str,
    errors: list[ErrorReport],
) -> SyncStateRecord:
    return SyncStateRecord(
        source=source,
        watermark=watermark,
        last_run_at=started_at,
        last_status=status,
        # First error only: the full list lives in SyncResult. Keeps the column short.
        error=errors[0].message if errors else None,
    )


def _error(
    *,
    stage: str,
    error_type: str,
    message: str,
    source_id: str | None = None,
    retryable: bool = False,
) -> ErrorReport:
    return ErrorReport(
        source="jira",
        source_id=source_id,
        stage=stage,
        error_type=error_type,
        message=message,
        retryable=retryable,
    )


def _stage_for(exc: BaseException) -> str:
    if isinstance(exc, ConnectorError):
        return "fetch"
    return "normalize"


def _short(exc: BaseException) -> str:
    """One-line, length-capped message: source bodies must not flood the report."""
    text = str(exc).replace("\n", " ").strip()
    return text[:300] or type(exc).__name__


def _issue_project_key(issue: Mapping[str, Any]) -> str | None:
    """Project key of a search result, or None when Jira did not return one."""
    fields = issue.get("fields")
    if not isinstance(fields, Mapping):
        return None
    project = fields.get("project")
    if not isinstance(project, Mapping):
        return None
    key = project.get("key")
    return str(key) if key else None


def _issue_updated(issue: Mapping[str, Any]) -> datetime | None:
    fields = issue.get("fields")
    if not isinstance(fields, Mapping):
        return None
    raw = fields.get("updated")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return parse_jira_timestamp(raw)
    except ValueError:
        return None


def _latest(current: datetime | None, candidate: datetime | None) -> datetime | None:
    if candidate is None:
        return current
    if current is None or candidate > current:
        return candidate
    return current


def _earliest(current: datetime | None, candidate: datetime | None) -> datetime | None:
    if candidate is None:
        return current
    if current is None or candidate < current:
        return candidate
    return current


__all__ = ["ISSUE_FIELDS", "JiraConnector", "validate_project_keys"]
