"""Inventory agent definition."""

from __future__ import annotations

from inventory_pilot_ai.agents.base import AgentDefinition, AgentTool
from inventory_pilot_ai.tools.inventory_tools import demo_item_lookup, demo_partner_lookup, demo_reorder_recommendations, demo_status


def build_inventory_agent() -> AgentDefinition:
    return AgentDefinition(
        name="InventoryAgent",
        system_prompt=(
            "You are the Inventory Pilot AI inventory specialist. Use tool evidence for stock, item, partner, and "
            "reorder answers. Clearly label sample/demo data."
        ),
        responsibilities=[
            "Answer item lookup and inventory availability questions.",
            "Explain partner sourcing for seeded sample items.",
            "Provide low-stock and reorder context when it helps the user act.",
        ],
        routing_rules=[
            "Use for SKU, item, supplier, stock, availability, and restock-oriented questions.",
            "Escalate to ForecastAgent when the user asks for future demand or trends.",
        ],
        tools=[
            AgentTool("DemoItemLookupTool", "Search seeded items.", demo_item_lookup, {}, {"type": "array"}),
            AgentTool("DemoAvailabilityTool", "Inspect stock levels.", demo_status, {}, {"type": "object"}),
            AgentTool("DemoPartnerLookupTool", "Inspect supplier details.", demo_partner_lookup, {}, {"type": "object"}),
            AgentTool("DemoRecommendationTool", "Generate reorder suggestions.", demo_reorder_recommendations, {}, {"type": "array"}),
        ],
        logging_rules=[
            "Record the item identifier or query terms used for lookup.",
            "Record when partner or recommendation tools were needed.",
        ],
        trace_tags=["inventory", "tool-calling"],
        trace_metadata={"team": "operations", "stage": "specialist"},
    )
