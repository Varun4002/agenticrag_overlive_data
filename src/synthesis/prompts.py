"""Prompts for answer synthesis.

The model may only restate verified claims. Claims are numbered so the answer
can cite them with ``[n]`` markers that the validator resolves to real chunks.
"""

from __future__ import annotations

from collections.abc import Sequence

from llm.base import Message, Role
from orchestrator.pipeline import Claim

__all__ = ["SYSTEM_PROMPT", "build_messages", "build_user_prompt"]

#: Answer budget. 1024 cut off real six-claim answers from the local 8B model
#: (``finish_reason="length"``, refused after ~80 s), so the cap is 2048. Extra
#: room cannot turn into uncited content: ``_clean_answer`` still drops every
#: sentence that carries no citation.
MAX_OUTPUT_TOKENS = 2048

SYSTEM_PROMPT = """You write a short answer using ONLY the verified claims provided.

Rules:
- Every factual sentence must carry citation markers like [1] or [2][3] that
  point at the numbered claims it is based on.
- Put the markers as bare brackets directly after the sentence they support.
  Never write "claim [1]" or "as indicated in claim".
- Use only the claims. Do not add facts, dates, names, or numbers that are not
  in a claim. Never use outside knowledge.
- Placeholders such as [PERSON_1] are opaque tokens. Repeat them exactly; never
  guess or expand them.
- If two claims conflict, present both values with their citations. Never pick
  one side silently.
- An answer like "I could not find evidence" is handled elsewhere; never invent
  a refusal, and never ask a question back.
- Be brief: at most 4 sentences in total. If you use a list, enumerate at most
  5 items and summarise the rest instead of listing every claim.
- Cite with the marker immediately after the sentence it supports, and never
  append a "Citations:" or "References:" list.
- Plain prose or a short markdown list. No headings, no preamble."""


def build_user_prompt(question: str, claims: Sequence[Claim]) -> str:
    """Question plus the numbered claims the model is allowed to use."""
    lines = [f"Question: {question}", "", "Verified claims:"]
    for index, claim in enumerate(claims, start=1):
        lines.append(f"[{index}] {claim.text}")
    lines.append("")
    lines.append("Write the answer with citation markers.")
    return "\n".join(lines)


def build_messages(question: str, claims: Sequence[Claim]) -> list[Message]:
    """Message list for one synthesis call."""
    return [
        Message(role=Role.system, content=SYSTEM_PROMPT),
        Message(role=Role.user, content=build_user_prompt(question, claims)),
    ]
