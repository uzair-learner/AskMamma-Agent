"""Memory agent definition."""

from __future__ import annotations

from inventory_pilot_ai.agents.base import AgentDefinition


def build_memory_agent() -> AgentDefinition:
    return AgentDefinition(
        name="MemoryAgent",
        system_prompt=(
            "You manage conversation, semantic, and audit memory for the Inventory Pilot AI learning project."
        ),
        responsibilities=[
            "Persist user questions and assistant responses.",
            "Store semantic context such as last referenced SKU.",
            "Store audit-ready summaries of routing and tool usage.",
        ],
        routing_rules=[
            "Support every request in the background rather than being user-routed directly."
        ],
        tools=[],
        logging_rules=[
            "Record each memory write with a memory type classification.",
            "Avoid storing secrets or unsafe payloads.",
        ],
        trace_tags=["memory", "persistence"],
        trace_metadata={"team": "state", "stage": "cross-cutting"},
    )
