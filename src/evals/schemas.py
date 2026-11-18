"""Golden-question and eval-report schemas (component 09).

``GoldenQuestion`` is the frozen golden-set item from
``docs/agents/09-eval-observability-agent.md``. The rest of this module is the
harness's report shape: ``QuestionReport`` per question and ``EvalResult`` as
the suite roll-up.

``fixtures/golden.json`` carries an explicit ``schema_version``; the loader
fails loudly when a suite and the harness disagree, so eval-set drift after a
schema change is a hard failure rather than a silently empty run
(``docs/agents/09-eval-observability-agent.md`` failure table).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# Bump when the golden file's shape changes. The loader refuses older files.
GOLDEN_SCHEMA_VERSION = 1

#: Source types a citation can point at (mirrors ``Citation.source``).
SOURCE_TYPES = frozenset({"jira", "notion", "gmail"})

Behavior = Literal["answer", "refuse"]


class GoldenSetError(RuntimeError):
    """The golden set is missing, malformed, or built for another schema version."""


class GoldenQuestion(BaseModel):
    """One hand-labelled question and what a correct pipeline must do with it.

    Field-for-field the contract in the component brief. ``notes`` carries the
    hand-labelling rationale and starts with a category tag such as
    ``[overdue]`` so coverage can be checked without adding fields the brief
    does not define.
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    expected_entities: list[str] = Field(default_factory=list)
    expected_behavior: Behavior
    must_cite_source_types: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("must_cite_source_types")
    @classmethod
    def _known_sources(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - SOURCE_TYPES)
        if unknown:
            raise ValueError(
                f"unknown source types {unknown}; expected subset of {sorted(SOURCE_TYPES)}"
            )
        return value

    @property
    def category(self) -> str:
        """Category tag from ``notes`` (``[overdue] ...``), or ``uncategorised``."""
        if self.notes.startswith("[") and "]" in self.notes:
            return self.notes[1 : self.notes.index("]")]
        return "uncategorised"


class GoldenSet(BaseModel):
    """The golden file: schema version + corpus pointer + questions."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    corpus: str = "fixtures/corpus"
    description: str = ""
    questions: list[GoldenQuestion] = Field(min_length=1)


def load_golden_set(path: Path | str) -> GoldenSet:
    """Load and validate a golden file, failing hard on schema drift."""
    golden_path = Path(path)
    try:
        raw = json.loads(golden_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GoldenSetError(f"golden set not found: {golden_path}") from exc
    except json.JSONDecodeError as exc:
        raise GoldenSetError(f"golden set is not valid JSON: {golden_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise GoldenSetError(
            "golden set must be a JSON object with 'schema_version' and "
            f"'questions' ({golden_path})"
        )

    version = raw.get("schema_version")
    if version != GOLDEN_SCHEMA_VERSION:
        raise GoldenSetError(
            f"golden set schema version mismatch in {golden_path}: file says {version!r}, "
            f"harness expects {GOLDEN_SCHEMA_VERSION}. Update the golden file or bump "
            "GOLDEN_SCHEMA_VERSION; do not silently reload a stale suite."
        )

    try:
        golden = GoldenSet.model_validate(raw)
    except ValidationError as exc:
        raise GoldenSetError(f"golden set failed validation ({golden_path}):\n{exc}") from exc

    seen: set[str] = set()
    duplicates: set[str] = set()
    for question in golden.questions:
        if question.question in seen:
            duplicates.add(question.question)
        seen.add(question.question)
    if duplicates:
        raise GoldenSetError(f"duplicate golden questions in {golden_path}: {sorted(duplicates)}")
    return golden


class GateCheck(BaseModel):
    """One pass/fail gate from the project success criteria."""

    name: str
    passed: bool
    actual: float | int | None = None
    threshold: str = ""
    detail: str = ""


class QuestionReport(BaseModel):
    """Scoring outcome for one golden question.

    ``answer_excerpt`` and ``uncited_claims`` are PII-masked before they land
    here, so a failing suite's markdown report never re-prints a leak.
    """

    question: str
    category: str = "uncategorised"
    expected_behavior: Behavior
    actual_behavior: Literal["answered", "refused", "error"] = "error"
    refusal_correct: bool = False
    groundedness: float | None = None
    structural_groundedness: float = 0.0
    pii_leaks: int = 0
    pii_findings: list[str] = Field(default_factory=list)
    citation_count: int = 0
    cited_source_types: list[str] = Field(default_factory=list)
    missing_source_types: list[str] = Field(default_factory=list)
    entity_coverage: float = 0.0
    latency_ms: int = 0
    trace_id: str | None = None
    answer_excerpt: str = ""
    uncited_claims: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    error: str | None = None


class EvalResult(BaseModel):
    """Suite-level roll-up plus per-question detail.

    ``groundedness`` is ``None`` when no judge was available — never a silent
    ``1.0``. ``groundedness_structural`` is the deterministic citation-
    resolution score and is always populated, so a report without a judge is
    still informative without pretending to be an entailment check.
    """

    groundedness: float | None = None
    pii_leaks: int = 0
    refusal_correct: float = 0.0
    latency_p50_ms: int = 0
    latency_p95_ms: int = 0
    per_question: list[QuestionReport] = Field(default_factory=list)

    suite: str = "golden"
    judge: str = "none"
    #: The ACL identity the questions were asked as. Retrieval enforces
    #: ``acl.users`` in SQL, so this value decides what the suite can see.
    user_id: str = "eval-user"
    pii_roster_names: int = 0
    question_count: int = 0
    errors: int = 0
    groundedness_structural: float = 0.0
    entity_coverage: float = 0.0
    gates: list[GateCheck] = Field(default_factory=list)
    gate_passed: bool = False
    flags: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
