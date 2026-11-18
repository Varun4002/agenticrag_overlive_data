"""Shared fixtures for evidence tests."""

from __future__ import annotations

import pytest

from evals.fixtures import FixtureCorpus, load_fixture_corpus


@pytest.fixture(scope="session")
def corpus() -> FixtureCorpus:
    """The synthetic corpus (fictional data, no network)."""
    return load_fixture_corpus()
