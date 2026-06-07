"""Compatibility wrapper around the new workflow package."""

from __future__ import annotations

from workflows.langgraph.workflow import (
    AgentResult,
    get_recent_traces,
    get_session_messages,
    invoke_agent,
    remember_session_value,
    workflow_mermaid,
)

__all__ = [
    "AgentResult",
    "get_recent_traces",
    "get_session_messages",
    "invoke_agent",
    "remember_session_value",
    "workflow_mermaid",
]
