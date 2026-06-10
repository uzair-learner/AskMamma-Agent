"""Reporting agent definition."""

from __future__ import annotations

from inventory_pilot_ai.agents.base import AgentDefinition, AgentTool
from inventory_pilot_ai.tools.inventory_tools import demo_forecast, demo_reorder_recommendations, write_demo_report


def build_reporting_agent() -> AgentDefinition:
    return AgentDefinition(
        name="ReportingAgent",
        system_prompt=(
            "You are the Inventory Pilot AI reporting specialist. Package specialist outputs into concise TXT, JSON, "
            "and Markdown summaries and generate downloadable reports when requested."
        ),
        responsibilities=[
            "Generate inventory, forecast, and recommendation summaries.",
            "Produce Markdown-friendly responses for the API and Streamlit UI.",
            "Create downloadable report artifacts for demo walkthroughs.",
        ],
        routing_rules=[
            "Always run after a specialist agent in the LangGraph workflow.",
            "Handle direct report requests even when there is no prior specialist output.",
        ],
        tools=[
            AgentTool("DemoReportWriterTool", "Write a downloadable report.", write_demo_report, {}, {"type": "object"}),
            AgentTool("DemoForecastTool", "Attach forecast snapshot data.", demo_forecast, {}, {"type": "object"}),
            AgentTool("DemoRecommendationTool", "Attach reorder recommendations.", demo_reorder_recommendations, {}, {"type": "array"}),
        ],
        logging_rules=[
            "Record which output formats were produced.",
            "Record the generated report file path when a download artifact is created.",
        ],
        trace_tags=["reporting", "presentation"],
        trace_metadata={"team": "reporting", "stage": "synthesis"},
    )
