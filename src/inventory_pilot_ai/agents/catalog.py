"""Agent catalog for the learning-oriented Inventory Pilot AI architecture."""

from __future__ import annotations

from inventory_pilot_ai.agents.base import AgentDefinition, AgentTool
from inventory_pilot_ai.agents.document import build_document_agent
from inventory_pilot_ai.agents.forecasting import build_forecast_agent
from inventory_pilot_ai.agents.inventory import build_inventory_agent
from inventory_pilot_ai.agents.memory_agent import build_memory_agent
from inventory_pilot_ai.agents.quality_review import build_quality_review_agent
from inventory_pilot_ai.agents.reporting import build_reporting_agent
from inventory_pilot_ai.agents.research import build_research_agent
from inventory_pilot_ai.agents.supervisor import build_supervisor_agent


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
