"""Terms that must never be redacted.

Redaction policy keeps org-level information intact (``docs/06-guardrails-pii.md``:
"project names, task titles, deadlines, statuses, org-level info ... are the
point of the system"). A statistical NER model does not know that: on the
fixture corpus ``en_core_web_sm`` labels ``Phoenix``, ``Jira``, ``Notion`` and
``EU`` as ``LOCATION`` and ``Alice Migration`` as ``PERSON``.

Two kinds of signal suppress those false positives:

* **Corpus / registry entities** — the codename ``Alice Migration`` and the
  project names/aliases from the shared fixture corpus manifest
  (``manifest.false_positive_names`` + ``manifest.projects``). In production the
  equivalent signal comes from ``mem.entity_registry`` (non-person entities);
  ``PII_ALLOWLIST_PATH`` points the scrubber at such a JSON document with the
  same shape.
* **Coarse geo terms** — countries and regions. spaCy tags them as GPE and
  Presidio maps that to ``LOCATION``; redacting them would damage the text
  without protecting anyone.

Suppression is only applied to name-like entity classes (``PERSON``,
``LOCATION``, ``ADDRESS``). A hard-signal class (email, phone, card, key) is
always redacted even if its text happens to contain an allowlisted term.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from config import get_settings

logger = logging.getLogger(__name__)

# Countries and supra-national regions: org-level geography, never personal.
COARSE_GEO: frozenset[str] = frozenset(
    {
        "us",
        "u.s.",
        "usa",
        "u.s.a.",
        "uk",
        "u.k.",
        "eu",
        "e.u.",
        "emea",
        "apac",
        "latam",
    }
)

#: Repo-local corpus manifest — the dev stand-in for a populated entity registry.
_REPO_MANIFEST = Path(__file__).resolve().parents[2] / "fixtures" / "corpus" / "manifest.json"


class AllowListError(ValueError):
    """An allowlist file exists but cannot be interpreted."""


@lru_cache(maxsize=32)
def _term_pattern(term: str) -> re.Pattern[str]:
    """Match one term as whole words, with any whitespace run between tokens."""
    parts = [re.escape(piece) for piece in term.split()]
    return re.compile(r"(?<![\w-])" + r"\s+".join(parts) + r"(?![\w-])", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class AllowList:
    """Non-person terms that must survive scrubbing untouched."""

    terms: frozenset[str]

    def is_empty(self) -> bool:
        return not self.terms

    def ranges(self, text: str) -> list[tuple[int, int]]:
        """Character ranges covered by allowlisted terms in ``text``."""
        spans: list[tuple[int, int]] = []
        for term in self.terms:
            for match in _term_pattern(term).finditer(text):
                spans.append((match.start(), match.end()))
        return spans


def _terms_from_manifest(path: Path) -> set[str]:
    """Read non-person entity names from a corpus/registry-shaped JSON file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AllowListError(f"cannot read PII allowlist {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AllowListError(f"PII allowlist {path} must contain a JSON object")

    terms: set[str] = set()

    # Shared fixture corpus shape: false_positive_names.items[].value
    false_positives = payload.get("false_positive_names", {})
    items = false_positives.get("items", []) if isinstance(false_positives, dict) else []
    for item in items if isinstance(items, list) else []:
        value = item.get("value") if isinstance(item, dict) else None
        if isinstance(value, str) and value.strip():
            terms.add(value.strip())

    # Shared fixture corpus shape: projects[].name / projects[].aliases
    for project in payload.get("projects", []) if isinstance(payload.get("projects"), list) else []:
        if not isinstance(project, dict):
            continue
        name = project.get("name")
        if isinstance(name, str) and name.strip():
            terms.add(name.strip())
        aliases = project.get("aliases", [])
        for alias in aliases if isinstance(aliases, list) else []:
            if isinstance(alias, str) and alias.strip():
                terms.add(alias.strip())

    # Registry-shaped shape: entities[].canonical_name / entities[].aliases,
    # restricted to non-person entities (person entries must stay redactable).
    for entity in payload.get("entities", []) if isinstance(payload.get("entities"), list) else []:
        if not isinstance(entity, dict) or entity.get("entity_type") == "person":
            continue
        canonical = entity.get("canonical_name")
        if isinstance(canonical, str) and canonical.strip():
            terms.add(canonical.strip())
        for alias in entity.get("aliases", []) if isinstance(entity.get("aliases"), list) else []:
            if isinstance(alias, str) and alias.strip():
                terms.add(alias.strip())

    return terms


@lru_cache(maxsize=8)
def load_allowlist_terms(path: str | None) -> frozenset[str]:
    """Allowlist terms from ``path`` (or the repo corpus manifest), plus coarse geo."""
    terms: set[str] = set(COARSE_GEO)
    if path:
        manifest = Path(path)
        if not manifest.is_file():
            raise AllowListError(f"PII allowlist not found: {manifest}")
        terms |= _terms_from_manifest(manifest)
    elif _REPO_MANIFEST.is_file():
        # Dev/test convenience: the fixture corpus is the non-person signal.
        try:
            terms |= _terms_from_manifest(_REPO_MANIFEST)
        except AllowListError as exc:  # pragma: no cover - corrupt fixture corpus
            logger.warning("ignoring unreadable fixture corpus manifest: %s", exc)
    return frozenset(terms)


def default_allowlist() -> AllowList:
    """The configured allowlist (``PII_ALLOWLIST_PATH``, else corpus manifest)."""
    path = get_settings().pii_allowlist_path
    return AllowList(terms=load_allowlist_terms(path))


def build_allowlist(
    extra_terms: Iterable[str] | None = None,
) -> AllowList:
    """Configured allowlist plus caller-supplied non-person terms."""
    terms = set(default_allowlist().terms)
    for term in extra_terms or ():
        cleaned = " ".join(term.split())
        if cleaned:
            terms.add(cleaned)
    return AllowList(terms=frozenset(terms))
