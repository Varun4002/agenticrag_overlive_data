"""Test helpers for the connector suite: fixture loading and a fake Jira server.

Kept out of ``conftest.py`` so test modules can import them directly
(``from jira_fixtures import JiraStub, load_json``) — pytest puts this directory
on ``sys.path`` because the suite has no ``__init__.py``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from connectors.settings import ConnectorSettings

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "jira"
JIRA_BASE_URL = "https://acme.atlassian.net"
SEARCH_PATH = "/rest/api/3/search/jql"

# Fixed clock so fetched_at/watermarks are deterministic.
FIXED_NOW = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)

_ISSUE_PATH_RE = re.compile(r"^/rest/api/3/issue/([A-Z][A-Z0-9_]*-\d+)$")
_COMMENTS_PATH_RE = re.compile(r"^/rest/api/3/issue/([A-Z][A-Z0-9_]*-\d+)/comment$")


def load_json(name: str) -> dict[str, Any]:
    """Load a Jira fixture payload by file name."""
    with (FIXTURE_DIR / name).open(encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
        return payload


def issue_from(name: str, index: int = 0) -> dict[str, Any]:
    """Pull one issue object out of a search fixture."""
    payload = load_json(name)
    issues = payload["issues"]
    assert isinstance(issues, list)
    issue: dict[str, Any] = issues[index]
    return issue


def comment_payloads(key: str) -> list[dict[str, Any]]:
    """Comment objects for an issue key, from the matching fixture file."""
    name = f"comments_{key.lower().replace('-', '_')}.json"
    comments = load_json(name)["comments"]
    assert isinstance(comments, list)
    return list(comments)


def make_settings(**overrides: Any) -> ConnectorSettings:
    """Connector settings for tests — fake credentials, never real ones."""
    values: dict[str, Any] = {
        "jira_base_url": JIRA_BASE_URL,
        "jira_email": "sync-bot@example.com",
        "jira_api_token": "test-token-not-real",
        "jira_projects": "PHX",
        "jira_page_size": 2,
        "jira_max_retries": 1,
        "jira_backoff_base_s": 0.001,
        "jira_rate_budget_calls": 100,
    }
    values.update(overrides)
    return ConnectorSettings(**values)


class JiraStub:
    """A fake Jira REST v3 server over ``httpx.MockTransport``.

    Serves issue search (offset *and* token pages), per-issue comments and single
    issue fetches, and records every request so tests can assert what the
    connector actually asked for. Failures are injectable per path substring.
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.search_pages: list[dict[str, Any]] = []
        self.comments: dict[str, list[dict[str, Any]]] = {}
        self.issues: dict[str, dict[str, Any]] = {}
        self.failures: dict[str, httpx.Response] = {}

    # --- configuration -------------------------------------------------
    def with_search_payloads(self, *pages: Mapping[str, Any]) -> JiraStub:
        self.search_pages = [dict(page) for page in pages]
        return self

    def with_comments(self, key: str, *comments: Mapping[str, Any]) -> JiraStub:
        self.comments[key] = [dict(comment) for comment in comments]
        return self

    def with_fixture_comments(self, key: str) -> JiraStub:
        return self.with_comments(key, *comment_payloads(key))

    def with_issue(self, issue: Mapping[str, Any]) -> JiraStub:
        self.issues[str(issue.get("key"))] = dict(issue)
        return self

    def fail(self, path_contains: str, response: httpx.Response) -> JiraStub:
        self.failures[path_contains] = response
        return self

    # --- transport -----------------------------------------------------
    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def calls_to(self, path_prefix: str) -> list[httpx.Request]:
        return [request for request in self.requests if request.url.path.startswith(path_prefix)]

    def search_jqls(self) -> list[str]:
        return [
            request.url.params.get("jql", "")
            for request in self.requests
            if request.url.path == SEARCH_PATH
        ]

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        for needle, response in self.failures.items():
            if needle in path:
                return response

        if path == SEARCH_PATH:
            return self._search(request)

        comments_match = _COMMENTS_PATH_RE.match(path)
        if comments_match:
            return self._comments(comments_match.group(1))

        issue_match = _ISSUE_PATH_RE.match(path)
        if issue_match:
            issue = self.issues.get(issue_match.group(1))
            if issue is None:
                return httpx.Response(
                    404,
                    json={"errorMessages": ["Issue does not exist or you do not have permission"]},
                )
            return httpx.Response(200, json=issue)

        return httpx.Response(404, json={"errorMessages": [f"unexpected path {path}"]})

    def _search(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        token = params.get("nextPageToken")
        start = int(params.get("startAt", "0"))
        page: Mapping[str, Any] | None = None
        for candidate in self.search_pages:
            if token is not None:
                if candidate.get("issuedToken") == token:
                    page = candidate
                    break
            elif candidate.get("issuedToken") is None and int(candidate.get("startAt", 0)) == start:
                page = candidate
                break
        if page is None:
            return httpx.Response(
                200, json={"startAt": start, "maxResults": 0, "total": 0, "issues": []}
            )
        body = {key: value for key, value in page.items() if key != "issuedToken"}
        return httpx.Response(200, json=body)

    def _comments(self, key: str) -> httpx.Response:
        items = self.comments.get(key, [])
        return httpx.Response(
            200,
            json={"startAt": 0, "maxResults": 50, "total": len(items), "comments": items},
        )
