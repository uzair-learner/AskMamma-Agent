"""Forecasting tools and deterministic demand calculations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from db.database import find_product, get_connection, rows_to_dicts, utc_now
from tools.inventory_tools.service import low_stock_products, out_of_stock_products


def demo_history(identifier: str | None = None, months: int = 6) -> dict[str, Any]:
    params: list[Any] = [f"-{months * 30} days"]
    product_clause = ""
    item = None
    if identifier:
        item = find_product(identifier)
        if not item:
            return {"found": False, "message": f"No sample demo item found for `{identifier}`."}
        product_clause = "AND s.product_id = ?"
        params.append(item["id"])
    with get_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT s.*, p.name AS product_name, p.category
            FROM sales_history s
            JOIN products p ON p.id = s.product_id
            WHERE date(s.sale_date) >= date('now', ?)
            {product_clause}
            ORDER BY s.sale_date
            """,
            params,
        ).fetchall()
    records = rows_to_dicts(rows)
    return {"found": bool(records), "item": item, "records": records}


def demo_forecast(identifier: str | None = None, months: int = 6) -> dict[str, Any]:
    history = demo_history(identifier, months)
    if not history.get("found"):
        return {"found": False, "message": "Insufficient sample history for a numeric forecast. Use the demo threshold instead."}
    monthly: dict[str, int] = defaultdict(int)
    item_totals: dict[str, int] = defaultdict(int)
    for record in history["records"]:
        month = record["sale_date"][:7]
        monthly[month] += int(record["quantity_sold"])
        item_totals[record["product_name"]] += int(record["quantity_sold"])
    ordered_months = sorted(monthly)
    values = [monthly[month] for month in ordered_months]
    moving_average = sum(values[-3:]) / min(3, len(values))
    trend = 0.0 if len(values) < 2 else (values[-1] - values[0]) / max(1, len(values) - 1)
    prediction = max(0.0, moving_average + trend)
    method = "3-month moving average with simple trend adjustment"
    explanation = (
        f"Used {len(values)} monthly demo history buckets. Last values: {values[-3:]}. "
        f"Trend adjustment: {trend:.1f}. Predicted next-month demand: {prediction:.1f} units."
    )
    with get_connection() as connection:
        product_id = history.get("item", {}).get("id") if history.get("item") else None
        connection.execute(
            """
            INSERT INTO forecasts (product_id, category, forecast_period, predicted_quantity, method, explanation, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (product_id, None, "next_month", prediction, method, explanation, utc_now()),
        )
    return {
        "found": True,
        "predicted_quantity": round(prediction, 2),
        "method": method,
        "explanation": explanation,
        "monthly_sales": dict(monthly),
        "top_items": sorted(item_totals.items(), key=lambda item: item[1], reverse=True)[:5],
    }


def demo_reorder_recommendations(identifier: str | None = None) -> list[dict[str, Any]]:
    products = [find_product(identifier)] if identifier else low_stock_products() + out_of_stock_products()
    recommendations: list[dict[str, Any]] = []
    for item in [product for product in products if product]:
        forecast = demo_forecast(item["sku"], months=6)
        predicted = forecast.get("predicted_quantity", 0) if forecast.get("found") else item["reorder_quantity"]
        target = max(item["reorder_quantity"], int(predicted) + item["reorder_level"])
        needed = max(0, target - item["stock_quantity"])
        recommendations.append(
            {
                "item_id": item["id"],
                "sku": item["sku"],
                "name": item["name"],
                "current_stock": item["stock_quantity"],
                "reorder_level": item["reorder_level"],
                "recommended_quantity": needed,
                "supplier": item.get("supplier_name"),
                "reason": f"Target quantity {target} based on demo threshold policy and demo forecast.",
            }
        )
    return recommendations
