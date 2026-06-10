"""Helpers for conversation, semantic, and audit memory views."""

from __future__ import annotations

import json
from typing import Any

from inventory_pilot_ai.db.database import get_connection, initialize_database, rows_to_dicts


def conversation_memory(session_id: str, limit: int = 20) -> list[dict[str, Any]]:
    initialize_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT session_id, role, content, created_at
            FROM chat_history
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    return list(reversed(rows_to_dicts(rows)))


def semantic_memory(limit: int = 50) -> list[dict[str, Any]]:
    initialize_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT memory_key, memory_value, created_at, updated_at
            FROM long_term_memory
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return rows_to_dicts(rows)


def audit_memory(limit: int = 50) -> list[dict[str, Any]]:
    initialize_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT session_id, selected_agent, user_input, final_answer, created_at, token_usage
            FROM agent_traces
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    records = rows_to_dicts(rows)
    for record in records:
        try:
            record["token_usage"] = json.loads(record["token_usage"] or "{}")
        except json.JSONDecodeError:
            record["token_usage"] = {}
    return records
