from scripts.seed_data import seed
from askmamma.tools import demo_forecast, demo_status, demo_item_search, demo_partner_lookup


def setup_module():
    seed()


def test_demo_item_search_finds_seeded_item():
    results = demo_item_search("USB-C")
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
