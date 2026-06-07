"""Agent catalog for the learning-oriented AskMamma architecture."""

from __future__ import annotations

from agents.base import AgentDefinition, AgentTool
from agents.document.agent import build_document_agent
from agents.forecast.agent import build_forecast_agent
from agents.inventory.agent import build_inventory_agent
from agents.memory.agent import build_memory_agent
from agents.quality_review.agent import build_quality_review_agent
from agents.reporting.agent import build_reporting_agent
from agents.research.agent import build_research_agent
from agents.supervisor.agent import build_supervisor_agent


def build_agent_catalog() -> dict[str, AgentDefinition]:
    agents = [
        build_supervisor_agent(),
        build_inventory_agent(),
        build_forecast_agent(),
        build_document_agent(),
        build_reporting_agent(),
        build_quality_review_agent(),
        build_memory_agent(),
        build_research_agent(),
    ]
    return {agent.name: agent for agent in agents}


def flatten_agent_tools() -> list[AgentTool]:
    tools: list[AgentTool] = []
    seen: set[str] = set()
    for agent in build_agent_catalog().values():
        for tool in agent.tools:
            if tool.name not in seen:
                seen.add(tool.name)
                tools.append(tool)
    return tools
