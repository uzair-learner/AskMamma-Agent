"""Quality review agent definition."""

from __future__ import annotations

from agents.base import AgentDefinition


def build_quality_review_agent() -> AgentDefinition:
    return AgentDefinition(
        name="QualityReviewAgent",
        system_prompt=(
            "You are the AskMamma quality reviewer. Check that answers are grounded, clearly labeled as "
            "demo data when applicable, and safe to present in an interview demo."
        ),
        responsibilities=[
            "Review agent outputs for grounding, clarity, and completeness.",
            "Append concise quality notes when important evidence is missing.",
            "Ensure final responses remain interview-friendly and easy to explain.",
        ],
        routing_rules=[
            "Run after ReportingAgent before the final response is returned."
        ],
        tools=[],
        logging_rules=[
            "Record review notes and demo-data labeling checks.",
            "Record whether evidence and tool usage were sufficient.",
        ],
        trace_tags=["quality", "review"],
        trace_metadata={"team": "quality", "stage": "review"},
    )
