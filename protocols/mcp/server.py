"""MCP-compatible registries and JSON-RPC handlers."""

from __future__ import annotations

from typing import Any

from agents.catalog import build_agent_catalog
from askmamma.tools import invoke_named_tool, tool_registry


def list_tools() -> list[dict[str, Any]]:
    return [tool.model_dump() for tool in tool_registry()]


def list_resources() -> list[dict[str, Any]]:
    return [
        {
            "name": "agent-catalog",
            "description": "Definitions for supervisor and specialist agents.",
            "uri": "askmamma://agents/catalog",
            "mime_type": "application/json",
        },
        {
            "name": "workflow-mermaid",
            "description": "Mermaid diagram for the LangGraph workflow.",
            "uri": "askmamma://workflows/langgraph/mermaid",
            "mime_type": "text/plain",
        },
        {
            "name": "interview-guide",
            "description": "Interview-ready architecture explanations.",
            "uri": "askmamma://docs/interview-guide",
            "mime_type": "text/markdown",
        },
    ]


def list_prompts() -> list[dict[str, Any]]:
    return [
        {
            "name": "inventory-summary",
            "description": "Prompt for generating an inventory summary from tool evidence.",
            "template": "Summarize the current inventory status using only the provided tool outputs.",
        },
        {
            "name": "forecast-explainer",
            "description": "Prompt for explaining a moving-average forecast to a .NET developer.",
            "template": "Explain the forecast result in simple language with the method, trend, and recommendation.",
        },
        {
            "name": "interview-walkthrough",
            "description": "Prompt for explaining the end-to-end architecture.",
            "template": "Walk through the AskMamma project architecture from UI to agents, RAG, protocols, and ML examples.",
        },
    ]


def rpc_success(request_id: str | int | None, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def rpc_error(request_id: str | int | None, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle_rpc(method: str, request_id: str | int | None, params: dict[str, Any]) -> dict[str, Any]:
    if method == "tools/list":
        return rpc_success(request_id, list_tools())
    if method == "tools/call":
        name = params.get("name")
        if not name:
            return rpc_error(request_id, -32602, "Missing `name` for tools/call.")
        return rpc_success(request_id, invoke_named_tool(name, params.get("arguments", {})))
    if method == "resources/list":
        return rpc_success(request_id, list_resources())
    if method == "prompts/list":
        return rpc_success(request_id, list_prompts())
    if method == "agents/list":
        catalog = build_agent_catalog()
        return rpc_success(
            request_id,
            [
                {
                    "name": agent.name,
                    "responsibilities": agent.responsibilities,
                    "routing_rules": agent.routing_rules,
                    "trace_tags": agent.trace_tags,
                }
                for agent in catalog.values()
            ],
        )
    return rpc_error(request_id, -32601, f"Unknown method `{method}`.")
