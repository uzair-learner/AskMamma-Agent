from scripts.seed_data import seed
from askmamma.tools import demand_forecast, inventory_status, product_search, supplier_lookup


def setup_module():
    seed()


def test_product_search_finds_seeded_product():
    results = product_search("USB-C")
    assert any(item["sku"] == "TECH-001" for item in results)


def test_inventory_status_marks_low_stock():
    result = inventory_status("Copy Paper")
    assert result["found"] is True
    assert result["low_stock"] is True


def test_supplier_lookup_returns_supplier():
    result = supplier_lookup("USB-C Cable")
    assert result["found"] is True
    assert result["supplier"]["name"] == "TechCore Components"


def test_demand_forecast_uses_sales_history():
    result = demand_forecast("Packing Tape")
    assert result["found"] is True
    assert result["predicted_quantity"] > 0
