"""Audit helpers for agent decisions and tool calls."""

from __future__ import annotations

import json
from typing import Any

from inventory_pilot_ai.observability import redact_payload
from inventory_pilot_ai.db.database import get_connection, initialize_database, rows_to_dicts, utc_now


def write_audit_entry(session_id: str, selected_agent: str, payload: dict[str, Any]) -> dict[str, Any]:
    initialize_database()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO agent_traces (
                session_id, user_input, selected_agent, tools_called, tool_inputs,
                tool_outputs_summary, final_answer, latency_ms, errors, token_usage, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                redact_payload(json.dumps(payload, default=str)),
                selected_agent,
                json.dumps(payload.get("tools_called", [])),
                json.dumps(redact_payload(payload.get("tool_inputs", [])), default=str),
                json.dumps(redact_payload(payload.get("tool_outputs", [])), default=str)[:1000],
                redact_payload(payload.get("answer", "")),
                payload.get("latency_ms", 0),
                payload.get("error"),
                json.dumps(payload.get("trace_metadata", {}), default=str),
                utc_now(),
            ),
        )
    return {"saved": True, "selected_agent": selected_agent}


def get_recent_audit_entries(limit: int = 50) -> list[dict[str, Any]]:
    initialize_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM agent_traces
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return rows_to_dicts(rows)

