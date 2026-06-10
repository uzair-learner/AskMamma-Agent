"""Agent card builder for A2A-style discovery."""

from __future__ import annotations

from typing import Any

from inventory_pilot_ai.agents.catalog import build_agent_catalog
from inventory_pilot_ai import config


def build_agent_card() -> dict[str, Any]:
    catalog = build_agent_catalog()
    return {
        "name": "Inventory Pilot AI Assistant",
        "description": "Learning-focused multi-agent project with LangGraph, LangChain, RAG, MCP, A2A, and ML examples.",
        "version": "3.0.0",
        "endpoint_url": "/agent/chat",
        "endpoint": "/agent/chat",
        "capabilities": {
            "tool_calling": True,
            "rag": True,
            "memory": True,
            "multi_agent": True,
            "task_execution": True,
            "mcp_adapter": True,
            "a2a_tasks": True,
            "langsmith_tracing_optional": bool(config.LANGSMITH_API_KEY),
        },
        "authentication": {"type": "none", "notes": "local development only"},
        "skills": [agent.name for agent in catalog.values()],
        "agents": [
            {
                "name": agent.name,
                "capabilities": agent.responsibilities,
                "skills": [tool.name for tool in agent.tools],
                "endpoint": "/agent/tasks",
            }
            for agent in catalog.values()
        ],
        "supported_input_modes": ["text", "task", "jsonrpc"],
        "supported_output_modes": ["text", "json", "markdown"],
        "examples": [
            {"input": "Which sample demo items are low in availability?", "route": "InventoryAgent"},
            {"input": "What is LangGraph in this project?", "route": "ResearchAgent"},
        ],
    }
