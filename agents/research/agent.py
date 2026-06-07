"""Research agent definition."""

from __future__ import annotations

from agents.base import AgentDefinition


def build_research_agent() -> AgentDefinition:
    return AgentDefinition(
        name="ResearchAgent",
        system_prompt=(
            "You are the AskMamma learning and interview-prep specialist. Explain AI agents, LangChain, "
            "LangGraph, RAG, MCP, A2A, TensorFlow, PyTorch, and architecture tradeoffs in simple language."
        ),
        responsibilities=[
            "Answer architecture and interview-prep questions in plain English.",
            "Translate the project structure into a teachable walkthrough.",
            "Compare multi-agent patterns, MCP, A2A, and ML examples for learners.",
        ],
        routing_rules=[
            "Use for interview prep, architecture explanation, and AI framework concept questions."
        ],
        tools=[],
        logging_rules=[
            "Record which concept area was explained.",
            "Record whether the answer referenced project architecture, protocols, or ML examples.",
        ],
        trace_tags=["research", "interview", "education"],
        trace_metadata={"team": "learning", "stage": "specialist"},
    )
