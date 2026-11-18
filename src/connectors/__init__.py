"""Source connectors (Jira / Notion / Gmail) and normalization (M1+).

Module map:

- ``base`` — shared async httpx framework: env auth, retries with jitter,
  ``Retry-After``, per-run rate budget, typed errors, pagination.
- ``settings`` — connector env config (``JIRA_*`` / ``GMAIL_*`` / ``NOTION_*``).
- ``contracts`` — ``SyncRequest`` / ``SyncResult`` / ``ErrorReport`` and the
  ``Connector`` protocol.
- ``normalizer`` — vendor payload -> ``CanonicalDoc`` -> ``core.documents``.
- ``jira`` — Jira Cloud REST v3 adapter (M1, implemented).
- ``gmail`` — Gmail IMAP adapter (M5, implemented): allowlisted labels only,
  read-only, one document per message.
- ``notion`` — typed stub that raises ``NotImplementedError`` (M2).
- ``service`` — ``sync()`` / ``fetch_targeted()`` entrypoints.
"""
