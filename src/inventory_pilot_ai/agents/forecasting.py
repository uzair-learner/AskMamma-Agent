"""Forecast agent definition."""

from __future__ import annotations

from inventory_pilot_ai.agents.base import AgentDefinition, AgentTool
from inventory_pilot_ai.tools.inventory_tools import demo_forecast, demo_history, demo_reorder_recommendations


def build_forecast_agent() -> AgentDefinition:
    return AgentDefinition(
        name="ForecastAgent",
        system_prompt=(
            "You are the Inventory Pilot AI forecasting specialist. Use historical sample data to explain moving "
            "averages, trend adjustments, and reorder implications."
        ),
        responsibilities=[
            "Run historical demand analysis over seeded inventory data.",
            "Explain moving-average and trend-based forecasts in plain language.",
            "Turn forecast outputs into reorder recommendations when relevant.",
        ],
        routing_rules=[
            "Use for demand forecasting, moving average, trend analysis, and reorder planning requests.",
            "Reference InventoryAgent results when an item identifier is already known.",
        ],
        tools=[
            AgentTool("DemoHistoryTool", "Retrieve seeded history.", demo_history, {}, {"type": "object"}),
            AgentTool("DemoForecastTool", "Run moving-average forecast.", demo_forecast, {}, {"type": "object"}),
            AgentTool("DemoRecommendationTool", "Recommend reorder actions.", demo_reorder_recommendations, {}, {"type": "array"}),
        ],
        logging_rules=[
            "Record the time window used for forecasting.",
            "Store the forecast method and quantity used in the answer.",
        ],
        trace_tags=["forecast", "analytics"],
        trace_metadata={"team": "analytics", "stage": "specialist"},
    )
