"""Compatibility wrapper around the workflow package with lazy imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from inventory_pilot_ai.workflow.graph import AgentResult


def _workflow_module() -> Any:
    from inventory_pilot_ai.workflow import graph

    return graph


def __getattr__(name: str) -> Any:
    if name == "AgentResult":
        return _workflow_module().AgentResult
    raise AttributeError(name)


def get_recent_traces(limit: int = 50):
    return _workflow_module().get_recent_traces(limit)


def get_session_messages(session_id: str, limit: int = 20):
    return _workflow_module().get_session_messages(session_id, limit)


def invoke_agent(user_input: str, session_id: str | None = None):
    return _workflow_module().invoke_agent(user_input, session_id)


def remember_session_value(session_id: str, key: str, value: str) -> None:
    _workflow_module().remember_session_value(session_id, key, value)


def workflow_mermaid() -> str:
    return _workflow_module().workflow_mermaid()


__all__ = ["AgentResult", "get_recent_traces", "get_session_messages", "invoke_agent", "remember_session_value", "workflow_mermaid"]
