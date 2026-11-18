"""Fixtures shared by the connector tests.

No live API calls anywhere: every adapter is driven through
``httpx.MockTransport`` against the hand-built payloads in ``fixtures/jira``, and
the sync logic runs against ``InMemoryDocumentStore``. ``test_db_store.py``
exercises the SQLAlchemy store against a scratch database instead.
"""

from __future__ import annotations

import pytest
from jira_fixtures import JiraStub, load_json

from connectors.normalizer import InMemoryDocumentStore


@pytest.fixture
def store() -> InMemoryDocumentStore:
    return InMemoryDocumentStore()


@pytest.fixture
def stub() -> JiraStub:
    """Full-sync stub: 3 issues over 2 pages plus their comments."""
    return (
        JiraStub()
        .with_search_payloads(load_json("search_page_1.json"), load_json("search_page_2.json"))
        .with_fixture_comments("PHX-101")
        .with_fixture_comments("PHX-102")
        .with_fixture_comments("PHX-103")
        .with_issue(load_json("issue_phx_102.json"))
    )
