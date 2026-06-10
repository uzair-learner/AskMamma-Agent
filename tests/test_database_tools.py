from inventory_pilot_ai.tools.inventory_tools import (
    demo_forecast,
    demo_item_lookup,
    demo_partner_lookup,
    demo_status,
    invoke_named_tool,
    tool_registry,
)
from scripts.seed_data import seed


def setup_module():
    seed()


def test_demo_item_lookup_finds_seeded_item():
    results = demo_item_lookup("USB-C")
    assert any(item["sku"] == "TECH-001" for item in results)


def test_demo_status_marks_low_availability():
    result = demo_status("Copy Paper")
    assert result["found"] is True
    assert result["low_stock"] is True


def test_demo_partner_lookup_returns_partner():
    result = demo_partner_lookup("USB-C Cable")
    assert result["found"] is True
    assert result["partner"]["name"] == "TechCore Components"


def test_demo_forecast_uses_sample_history():
    result = demo_forecast("Packing Tape")
    assert result["found"] is True
    assert result["predicted_quantity"] > 0


def test_tool_registry_has_input_and_output_schema():
    tool = next(item for item in tool_registry() if item.name == "DemoForecastTool")
    assert tool.input_schema["type"] == "object"
    assert tool.output_schema["type"] == "object"


def test_invoke_named_tool_runs_langchain_tool():
    result = invoke_named_tool("DemoAvailabilityTool", {"identifier": "Copy Paper"})
    assert result["found"] is True
