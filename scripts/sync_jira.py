#!/usr/bin/env python
"""Manual Jira sync entrypoint (M1).

    # incremental (default)
    .venv/bin/python scripts/sync_jira.py

    # full resync (also detects deletions)
    .venv/bin/python scripts/sync_jira.py --full

    # machine-readable result
    .venv/bin/python scripts/sync_jira.py --json

Configuration comes from the environment only, via ``connectors.settings``:
``JIRA_BASE_URL``, ``JIRA_EMAIL``, ``JIRA_API_TOKEN``, ``JIRA_PROJECTS`` (+
optional tuning knobs, see ``src/connectors/HANDOFF.md``), and ``DATABASE_URL``
for persistence. Load the dev credentials file in the calling shell, e.g.:

    set -a; source ~/.config/agentic-rag/dev.env; set +a

Exit codes: 0 = clean run, 1 = run finished with collected errors (partial),
2 = the run failed entirely (auth/config/database), 3 = not configured.

The script never prints credentials: ``describe_settings`` only reports non-secret
connector state.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Allow running from a checkout without installing: scripts/ is not a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from connectors import service
from connectors.base import ConnectorConfigError, ConnectorError
from connectors.contracts import SyncRequest, SyncResult
from connectors.settings import get_connector_settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull Jira issues and comments into core.documents."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="full resync (default: incremental from the stored watermark)",
    )
    parser.add_argument("--json", action="store_true", help="print the SyncResult as JSON")
    parser.add_argument(
        "--source",
        default="jira",
        choices=["jira", "notion", "gmail"],
        help="source to sync (notion is not implemented until M2)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="enable debug logging (never logs credentials)"
    )
    return parser.parse_args(argv)


def summarize(result: SyncResult) -> str:
    lines = [
        f"source: {result.source}",
        f"mode: {'full' if result.full else 'incremental'}",
        f"documents upserted: {result.documents_upserted}",
        f"documents unchanged: {result.documents_unchanged}",
        f"documents retired: {result.documents_retired}",
        f"documents skipped: {result.documents_skipped}",
        f"watermark: {result.watermark.isoformat()}",
        f"requests: {result.requests} (retries: {result.retries})",
        f"errors: {len(result.errors)}",
    ]
    for error in result.errors:
        lines.append(
            f"  - [{error.stage}] {error.source_id or '<run>'}: "
            f"{error.error_type}: {error.message}"
        )
    return "\n".join(lines)


async def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = get_connector_settings()
    logging.getLogger("connectors").info(
        "connector config: %s", service.describe_settings(settings)
    )

    try:
        request = SyncRequest(source=args.source, full=args.full)
        result = await service.sync(request, settings=settings)
    except ConnectorConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2 if "DATABASE_URL" in str(exc) else 3
    except NotImplementedError as exc:
        print(f"not implemented: {exc}", file=sys.stderr)
        return 3
    except ConnectorError as exc:
        print(f"sync failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(summarize(result))
    return 0 if result.ok else 1


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
