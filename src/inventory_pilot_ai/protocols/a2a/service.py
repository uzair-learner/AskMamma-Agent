"""A2A-style task helpers for agent-to-agent communication demos."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_task_record(task_id: str, message: str, metadata: dict[str, Any], from_agent: str | None = None) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "status": "submitted",
        "assigned_agent": from_agent,
        "input_payload": {"message": message},
        "output_payload": None,
        "error_payload": None,
        "metadata": metadata,
        "submitted_at": utcnow(),
        "started_at": None,
        "completed_at": None,
        "failed_at": None,
    }
