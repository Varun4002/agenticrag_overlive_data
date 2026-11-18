"""Import-surface tests: the orchestrator package must import from any entry point.

The regression these cover: ``orchestrator/__init__.py`` imported
``orchestrator.run`` eagerly, and ``orchestrator.run`` imports
``memory.answer_cache``, which imports the leaf ``orchestrator.contracts``.
Anything that pulled ``memory`` (or ``connectors``, which imports ``memory``)
before ``orchestrator`` therefore died with ``ImportError: cannot import name
'AnswerCacheProtocol' from partially initialized module``.

Each test runs a fresh interpreter: the pytest session has already imported
both packages, so an in-process check would pass on the broken code too.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"


def run_python(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a fresh interpreter from the project root with ``src`` importable."""
    return subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": str(SRC)},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def test_connectors_imported_before_orchestrator() -> None:
    """The exact failing order: ``connectors.service`` first, then ``orchestrator``."""
    result = run_python("-c", "import connectors.service; import orchestrator; print('ok')")

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_contracts_import_does_not_pull_the_state_machine() -> None:
    """``orchestrator.contracts`` stays a leaf: no ``orchestrator.run``, no ``memory``."""
    result = run_python(
        "-c",
        "import sys, orchestrator.contracts; "
        "assert 'orchestrator.run' not in sys.modules, 'orchestrator.run was imported'; "
        "assert 'memory.answer_cache' not in sys.modules, 'memory.answer_cache was imported'; "
        "print('ok')",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_package_surface_is_still_importable_from_the_package() -> None:
    """``from orchestrator import ...`` keeps working through the lazy attribute."""
    import orchestrator
    from orchestrator import (
        AskRequest,
        AskResponse,
        Citation,
        Orchestrator,
        Plan,
        Refusal,
        Stages,
        default_stages,
    )
    from orchestrator.contracts import AskRequest as ContractAskRequest
    from orchestrator.run import Orchestrator as RunOrchestrator
    from orchestrator.run import Stages as RunStages
    from orchestrator.run import default_stages as run_default_stages

    assert AskRequest is ContractAskRequest
    assert Orchestrator is RunOrchestrator
    assert Stages is RunStages
    assert default_stages is run_default_stages
    assert orchestrator.Orchestrator is RunOrchestrator
    assert Plan is not None
    assert AskResponse is not None
    assert Citation is not None
    assert Refusal is not None


def test_package_unknown_attribute_raises() -> None:
    """A lazy ``__init__`` must still fail honestly on a typo."""
    import orchestrator

    try:
        orchestrator.NoSuchThing  # noqa: B018 - the AttributeError is the assertion
    except AttributeError as exc:
        assert "NoSuchThing" in str(exc)
    else:  # pragma: no cover - would only run on a broken lazy surface
        raise AssertionError("unknown attribute did not raise AttributeError")


def test_sync_jira_help_runs() -> None:
    """The CLI entry point imports ``connectors`` first and must still start."""
    result = run_python("scripts/sync_jira.py", "--help")

    assert result.returncode == 0, result.stderr
    assert "--source" in result.stdout
