"""Shared agent contracts and execution helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


ToolCallable = Callable[..., Any]


@dataclass(slots=True)
class AgentTool:
    name: str
    description: str
    handler: ToolCallable
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]


@dataclass(slots=True)
class AgentDefinition:
    name: str
    system_prompt: str
    responsibilities: list[str]
    routing_rules: list[str]
    tools: list[AgentTool]
    logging_rules: list[str] = field(default_factory=list)
    trace_tags: list[str] = field(default_factory=list)
    trace_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentExecution:
    agent_name: str
    answer: str
    tools_called: list[str] = field(default_factory=list)
    tool_inputs: list[dict[str, Any]] = field(default_factory=list)
    tool_outputs: list[Any] = field(default_factory=list)
    intermediate_steps: list[dict[str, Any]] = field(default_factory=list)
    review_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
