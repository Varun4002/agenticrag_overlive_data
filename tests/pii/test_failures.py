"""Fail-closed behaviour the ``pii.errors`` contract promises.

Two paths matter and both are easy to get wrong in the quiet direction:

* an entity class Presidio returns that has no placeholder label must abort the
  pack (ruling #15) — skipping a detected span is a silent leak;
* an anonymizer failure must surface as a typed error, never as partially
  replaced text.

Neither message may contain the value that was being redacted.
"""

from __future__ import annotations

import pytest
from pii_fixtures import make_pack

from pii import analyzer as pii_analyzer
from pii import scrub
from pii.errors import PiiScrubError, PiiUnavailableError

TEXT = "Priya Raghunathan owns the weekly status report."
NAME = "Priya Raghunathan"


async def test_unmapped_entity_class_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A class we cannot map is an error, not a counted skip."""

    async def fake_analyze(
        text: str, *, entities: object, person_names: object = ()
    ) -> list[pii_analyzer.DetectedSpan]:
        return [
            pii_analyzer.DetectedSpan(
                start=0, end=len(NAME), entity_type="MEDICAL_LICENSE", score=0.9
            )
        ]

    monkeypatch.setattr(pii_analyzer, "analyze", fake_analyze)

    with pytest.raises(PiiScrubError) as excinfo:
        await scrub(make_pack(TEXT), trace_id="trace-unmapped")

    message = str(excinfo.value)
    assert "MEDICAL_LICENSE" in message
    assert NAME not in message


@pytest.mark.usefixtures("engines")
async def test_anonymizer_failure_surfaces_as_pii_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The anonymizer is part of the machinery: its failure is ``PiiUnavailableError``."""

    class ExplodingAnonymizer:
        def anonymize(self, **kwargs: object) -> object:
            raise RuntimeError("anonymizer down")

    async def fake_analyze(
        text: str, *, entities: object, person_names: object = ()
    ) -> list[pii_analyzer.DetectedSpan]:
        return [
            pii_analyzer.DetectedSpan(
                start=0, end=len(NAME), entity_type="PERSON", score=0.9
            )
        ]

    monkeypatch.setattr(pii_analyzer, "analyze", fake_analyze)
    monkeypatch.setattr(
        pii_analyzer,
        "get_engines",
        lambda: pii_analyzer.Engines(analyzer=None, anonymizer=ExplodingAnonymizer()),
    )

    with pytest.raises(PiiUnavailableError) as excinfo:
        await scrub(make_pack(TEXT), trace_id="trace-anonymizer")

    assert NAME not in str(excinfo.value)
