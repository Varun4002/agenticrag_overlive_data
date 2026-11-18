"""Orchestrator: the request state machine and the frozen cross-agent contracts.

Public surface:

- :class:`~orchestrator.contracts.AskRequest` / :class:`~orchestrator.contracts.AskResponse`
  — frozen HTTP contract.
- :class:`~orchestrator.pipeline.EvidencePack` / ``ScrubbedPack`` /
  ``VerifiedClaims`` / ``SynthesisResult`` — frozen internal hand-off shapes.
- :class:`~orchestrator.run.Orchestrator` — retrieve → scrub → verify →
  synthesize, with refusals and audit events.

Only the contracts are imported eagerly; the ``run``-owned names are resolved on
first attribute access (PEP 562). Importing them eagerly would close an import
cycle: ``orchestrator.run`` imports ``memory.answer_cache``, which imports the
leaf ``orchestrator.contracts``. With an eager ``__init__``, importing ``memory``
— or ``connectors``, which imports ``memory`` — before ``orchestrator`` crashed
on a partially initialised module. Keeping this module down to the contracts
keeps ``orchestrator.contracts`` a leaf that anything may import in any order.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from orchestrator.contracts import AskRequest, AskResponse, Citation, Refusal

if TYPE_CHECKING:
    from orchestrator.run import Orchestrator, Plan, Stages, default_stages

__all__ = [
    "AskRequest",
    "AskResponse",
    "Citation",
    "Orchestrator",
    "Plan",
    "Refusal",
    "Stages",
    "default_stages",
]

#: Submodule owning the eagerly-heavy names; imported on first access only.
_LAZY_MODULE = "orchestrator.run"
_LAZY_NAMES = frozenset({"Orchestrator", "Plan", "Stages", "default_stages"})


def __getattr__(name: str) -> Any:
    """Resolve the ``run``-owned names lazily, so importing this package stays cheap."""
    if name not in _LAZY_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(_LAZY_MODULE), name)
