"""Eval harness and observability tooling (component 09).

``schemas`` — golden-question and report models; ``scoring`` — pure
groundedness / PII / refusal checks; ``run`` — the ``/ask``-driven suite CLI;
``trace_report`` — per-trace timeline and aggregate stats from ``audit.events``.

The fixture-corpus loader stays in ``fixtures.py`` (owned by the indexer agent).
"""
