#!/usr/bin/env python
"""Manual, opt-in smoke check for the real embedding endpoint.

Not part of ``pytest``: the suite must never touch the network. Run it by hand
after changing the embedding provider, the model, or the schema dimension::

    set -a; source ~/.config/agentic-rag/dev.env; set +a
    .venv/bin/python tools/smoke_embeddings.py

What it proves:

1. ``POST {LLM_BASE_URL}/embeddings`` answers for ``EMBEDDING_MODEL``.
2. Vectors have exactly ``EMBEDDING_DIMENSIONS`` floats, i.e. they fit
   ``core.chunks.embedding`` (migration 0002 → 768).
3. Vectors are L2-normalized (so cosine distance is well-behaved).
4. The same text embeds (near-)identically twice, and different texts do not.

The API key is read from settings and never printed.
"""

from __future__ import annotations

import asyncio
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import get_settings  # noqa: E402
from indexing.embedder import OpenAICompatibleEmbedder  # noqa: E402
from indexing.errors import EmbeddingError  # noqa: E402

TEXTS = [
    "Project Phoenix cutover is targeted at 2026-12-04.",
    "Project Phoenix cutover is targeted at 2026-12-04.",
    "Project Atlas retires the legacy warehouse ETL on 2026-11-26.",
]


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


async def main() -> int:
    settings = get_settings()
    embedder = OpenAICompatibleEmbedder(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )

    print(f"endpoint : {settings.llm_base_url}")
    print(f"model    : {settings.embedding_model}")
    print(f"expected : {settings.embedding_dimensions} dimensions")

    started = time.perf_counter()
    try:
        async with embedder:
            result = await embedder.embed(TEXTS)
    except EmbeddingError as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    dimensions = {len(vector) for vector in result.vectors}
    norms = [round(math.sqrt(sum(v * v for v in vector)), 6) for vector in result.vectors]
    same_text = cosine(result.vectors[0], result.vectors[1])
    different_text = cosine(result.vectors[0], result.vectors[2])

    print(f"latency  : {elapsed_ms} ms for {len(TEXTS)} inputs ({result.batches} request)")
    print(f"tokens   : {result.tokens} (provider usage, else estimate)")
    print(f"dims     : {sorted(dimensions)}")
    print(f"first 5  : {[round(value, 6) for value in result.vectors[0][:5]]}")
    print(f"l2 norms : {norms}")
    print(f"cosine(same text)      : {same_text:.6f}")
    print(f"cosine(different text) : {different_text:.6f}")

    failures: list[str] = []
    if dimensions != {settings.embedding_dimensions}:
        failures.append(
            f"dimension mismatch: got {sorted(dimensions)}, "
            f"expected {settings.embedding_dimensions} (schema is vector(768); a "
            "different model needs a new migration)"
        )
    if any(abs(norm - 1.0) > 1e-3 for norm in norms):
        failures.append(f"vectors are not L2-normalized: {norms}")
    if same_text < 0.999:
        failures.append(f"same text produced different vectors (cosine {same_text:.6f})")
    if different_text > 0.99:
        failures.append(f"different texts are nearly identical (cosine {different_text:.6f})")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("OK: live embedding endpoint matches the configured model and schema dimension")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
