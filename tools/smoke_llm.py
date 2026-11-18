"""Live provider smoke test — one real completion, nothing mocked.

Run once a key is configured (never commit the key):

    set -a; source ~/.config/agentic-rag/dev.env; set +a   # or export LLM_API_KEY
    .venv/bin/python tools/smoke_llm.py

Exit codes: 0 = provider answered, 1 = provider call failed, 2 = no key set.
This is intentionally NOT a pytest test: it needs the network and a paid key.
"""

from __future__ import annotations

import asyncio
import sys

from config import get_settings
from llm.base import CompletionRequest, Message, Role
from llm.openai_compatible import OpenAICompatibleClient


async def main() -> int:
    settings = get_settings()
    base_url = settings.llm_base_url
    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    is_local = any(
        host in base_url for host in ("localhost", "127.0.0.1", "[::1]", "0.0.0.0")
    )
    if not api_key and not is_local:
        print(
            "NO_KEY: LLM_API_KEY is not set and the base URL is not local — "
            "nothing was called.\n"
            "Set LLM_BASE_URL + LLM_API_KEY + LLM_MODEL and rerun."
        )
        return 2
    if not api_key:
        print(f"no key needed for local provider at {base_url}")

    client = OpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key,
        default_model=settings.llm_model,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        backoff_base_s=settings.llm_backoff_base_s,
    )
    try:
        response = await client.complete(
            CompletionRequest(
                messages=[Message(role=Role.user, content="Reply with exactly: OK")],
                # Generous budget on purpose: reasoning models (qwen3.5, deepseek-r1,
                # ...) emit a thinking block first and return empty content if the
                # budget is exhausted before the answer. Callers must budget likewise.
                max_tokens=256,
            )
        )
    except Exception as exc:  # a smoke test reports every failure mode
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1
    finally:
        await client.aclose()

    print(f"model:         {response.model}")
    print(f"text:          {response.text.strip()!r}")
    print(f"finish_reason: {response.finish_reason}")
    print(f"usage:         {response.usage.model_dump()}")
    print(f"latency_ms:    {response.latency_ms}")
    print(f"ledger:        calls={client.ledger.calls} tokens={client.ledger.total_tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
