"""Orchestrator: the request state machine and the frozen cross-agent contracts.

Public surface:

- :class:`~orchestrator.contracts.AskRequest` / :class:`~orchestrator.contracts.AskResponse`
  — frozen HTTP contract.
- :class:`~orchestrator.pipeline.EvidencePack` / ``ScrubbedPack`` /
  ``VerifiedClaims`` / ``SynthesisResult`` — frozen internal hand-off shapes.
- :class:`~orchestrator.run.Orchestrator` — retrieve → scrub → verify →
  synthesize, with refusals and audit events.
"""

from __future__ import annotations

from orchestrator.contracts import AskRequest, AskResponse, Citation, Refusal
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
