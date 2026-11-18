"""Deterministic, dependency-free token estimation.

Chunk sizing needs a token count, but pulling in a model-specific tokenizer
(``tiktoken``) would pin the chunker to one embedding provider and add a heavy
dependency for a rough budget. Instead this module uses a fixed word/punctuation
split: one token per word plus one per punctuation mark.

Accuracy: close to BPE behaviour on prose; it under-counts long identifiers and
over-counts markdown punctuation. Absolute values are approximate, but the
function is *deterministic and stable across processes and Python versions*,
which is what chunk boundaries and the golden tests depend on.
"""

from __future__ import annotations

import hashlib
import re

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Threshold above which a document is logged as oversized (indexer side).
OVERSIZED_DOCUMENT_TOKENS = 100_000


def estimate_tokens(text: str) -> int:
    """Approximate the number of model tokens in ``text``.

    Deterministic: same input, same count, on every machine.
    """
    if not text:
        return 0
    return len(_TOKEN_RE.findall(text))


def sha256_text(text: str) -> str:
    """Stable content hash used for chunk dedupe and embedding-cache lookups."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
