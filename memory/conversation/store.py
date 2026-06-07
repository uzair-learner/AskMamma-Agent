"""Conversation memory helpers backed by SQLite."""

from __future__ import annotations

from typing import Any

from core.observability import redact_payload
from db.database import get_connection, initialize_database, rows_to_dicts, utc_now


def save_message(session_id: str, role: str, content: str) -> None:
    initialize_database()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO chat_history (session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, role, redact_payload(content), utc_now()),
        )


def get_messages(session_id: str, limit: int = 20) -> list[dict[str, Any]]:
    initialize_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT role, content, created_at
            FROM chat_history
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    return list(reversed(rows_to_dicts(rows)))

