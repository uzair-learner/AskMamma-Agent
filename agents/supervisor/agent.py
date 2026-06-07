"""Supervisor agent definition."""

from __future__ import annotations

from agents.base import AgentDefinition


def build_supervisor_agent() -> AgentDefinition:
    return AgentDefinition(
        name="SupervisorAgent",
        system_prompt=(
            "You supervise the AskMamma learning workflow. Route each request to the best specialist "
            "agent, preserve context, and keep the user-facing answer grounded in tool evidence."
        ),
        responsibilities=[
            "Classify user intent across inventory, forecasting, documents, research, and reporting.",
            "Decide whether a specialist agent is required or a greeting can be answered directly.",
            "Attach routing and trace metadata for observability and interview walkthroughs.",
        ],
        routing_rules=[
            "Route greetings to a direct supervisor response.",
            "Route inventory, stock, partner, and availability questions to InventoryAgent.",
            "Route demand, reorder, moving-average, and trend questions to ForecastAgent.",
            "Route policy, uploaded file, PDF, DOCX, TXT, and retrieval questions to DocumentAgent.",
            "Route architecture, MCP, A2A, LangGraph, LangChain, TensorFlow, and interview questions to ResearchAgent.",
            "Route explicit report generation requests to ReportingAgent.",
        ],
        tools=[],
        logging_rules=[
            "Log the selected route and the keywords that triggered it.",
            "Store trace metadata for downstream agent selection.",
        ],
        trace_tags=["supervisor", "routing", "entrypoint"],
        trace_metadata={"team": "orchestration", "stage": "triage"},
    )
