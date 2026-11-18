"""Parsing: tolerant of model noise, strict about schema."""

from __future__ import annotations

import pytest

from evidence.parsing import EvidenceParseError, extract_json_object, parse_extraction


def test_parses_fenced_json() -> None:
    raw = '```json\n{"claims":[{"text":"PHX-150 cutover is 2026-11-20","chunk_ids":["c1"]}]}\n```'
    result = parse_extraction(raw)
    assert len(result.claims) == 1
    assert result.claims[0].chunk_ids == ["c1"]


def test_tolerates_prose_around_payload() -> None:
    raw = 'Sure! Here is the extraction:\n{"claims":[]}\nHope that helps.'
    assert parse_extraction(raw).claims == []


def test_strips_reasoning_block_with_decoy_json() -> None:
    raw = (
        '<think >decoy {"claims":[{"text":"DECOY","chunk_ids":["ghost"]}]}</think >'
        '\n{"claims":[]}'
    )
    assert parse_extraction(raw).claims == []


def test_braces_inside_strings_do_not_break_scanning() -> None:
    raw = 'prefix {"claims":[{"text":"uses {curly} braces","chunk_ids":["c1"]}]} suffix'
    assert parse_extraction(raw).claims[0].text == "uses {curly} braces"


def test_accepts_bare_claim_array() -> None:
    raw = '[{"text":"PHX-131 is blocked","chunk_ids":["c1"]}]'
    assert len(parse_extraction(raw).claims) == 1


def test_coerces_chunk_id_string_and_percentage_confidence() -> None:
    raw = '{"claims":[{"text":"x","chunk_ids":"c1, c2","confidence":"80%"}]}'
    claim = parse_extraction(raw).claims[0]
    assert claim.chunk_ids == ["c1", "c2"]
    assert claim.confidence == pytest.approx(0.8)


def test_rejects_missing_claims_key() -> None:
    with pytest.raises(EvidenceParseError):
        parse_extraction('{"results":[]}')


def test_rejects_non_json_output() -> None:
    with pytest.raises(EvidenceParseError):
        parse_extraction("I could not find anything useful.")


def test_rejects_claim_without_text() -> None:
    with pytest.raises(EvidenceParseError):
        parse_extraction('{"claims":[{"chunk_ids":["c1"]}]}')


def test_extract_json_object_wraps_arrays() -> None:
    assert extract_json_object('[{"text":"x","chunk_ids":["c1"]}]')["claims"]
