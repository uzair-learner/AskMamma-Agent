"""Typed tool functions used by AskMamma agents and exposed through APIs."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core import config
from db.database import (
    find_product,
    get_connection,
    get_product,
    list_products,
    low_stock_products,
    out_of_stock_products,
    rows_to_dicts,
    utc_now,
)


class ToolCall(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] = Field(default_factory=dict)


def tool_registry() -> list[ToolCall]:
    return [
        ToolCall(name="ProductSearchTool", description="Search AskMamma demo items by name, SKU, category, supplier, or description.", input_schema={"query": "string"}),
        ToolCall(name="AvailabilityStatusTool", description="Return quantity, threshold, low-availability status, and unavailable status.", input_schema={"identifier": "string optional"}),
        ToolCall(name="SupplierLookupTool", description="Find supplier information for a product.", input_schema={"identifier": "string"}),
        ToolCall(name="SalesHistoryTool", description="Retrieve previous sales for a product or category.", input_schema={"identifier": "string optional", "months": "integer"}),
        ToolCall(name="DemandForecastTool", description="Predict future demand using moving average and trend adjustment.", input_schema={"identifier": "string optional", "months": "integer"}),
        ToolCall(name="ReorderRecommendationTool", description="Recommend reorder quantities from stock, reorder levels, sales, and lead time.", input_schema={"identifier": "string optional"}),
        ToolCall(name="DocumentSearchTool", description="Search indexed documents using local RAG retrieval.", input_schema={"query": "string"}),
        ToolCall(name="ReportWriterTool", description="Generate an AskMamma operations report and save it under outputs/reports.", input_schema={"title": "string optional"}),
        ToolCall(name="AddInventoryMovementTool", description="Record stock-in, stock-out, or adjustment movement.", input_schema={"product_id": "integer", "movement_type": "string", "quantity": "integer", "reason": "string"}),
        ToolCall(name="AuditLogTool", description="Save agent actions and tool calls.", input_schema={"session_id": "string", "message": "string"}),
    ]


def product_search(query: str) -> list[dict[str, Any]]:
    return list_products(search=query, limit=25)


def inventory_status(identifier: str | None = None) -> dict[str, Any]:
    if identifier:
        product = find_product(identifier)
        if not product:
            return {"found": False, "message": f"No product found for `{identifier}`."}
        return {
            "found": True,
            "product": product,
            "low_stock": product["stock_quantity"] > 0 and product["stock_quantity"] <= product["reorder_level"],
            "out_of_stock": product["stock_quantity"] <= 0,
        }
    return {"low_stock": low_stock_products(), "out_of_stock": out_of_stock_products()}


def supplier_lookup(identifier: str) -> dict[str, Any]:
    product = find_product(identifier)
    if not product:
        return {"found": False, "message": f"No product found for `{identifier}`."}
    return {
        "found": True,
        "product": product["name"],
        "supplier": {
            "name": product.get("supplier_name"),
            "email": product.get("contact_email"),
            "lead_time_days": product.get("lead_time_days"),
        },
    }


def sales_history(identifier: str | None = None, months: int = 6) -> dict[str, Any]:
    params: list[Any] = [f"-{months * 30} days"]
    product_clause = ""
    product = None
    if identifier:
        product = find_product(identifier)
        if not product:
            return {"found": False, "message": f"No product found for `{identifier}`."}
        product_clause = "AND s.product_id = ?"
        params.append(product["id"])
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
    return {"found": bool(records), "product": product, "records": records}


def demand_forecast(identifier: str | None = None, months: int = 6) -> dict[str, Any]:
    history = sales_history(identifier, months)
    if not history.get("found"):
        return {
            "found": False,
            "message": "Insufficient sales history for a numeric forecast. Use reorder levels as a fallback.",
        }

    monthly: dict[str, int] = defaultdict(int)
    product_totals: dict[str, int] = defaultdict(int)
    for record in history["records"]:
        month = record["sale_date"][:7]
        monthly[month] += int(record["quantity_sold"])
        product_totals[record["product_name"]] += int(record["quantity_sold"])

    ordered_months = sorted(monthly)
    values = [monthly[month] for month in ordered_months]
    if len(values) < 2:
        prediction = float(values[0])
        trend = 0.0
    else:
        moving_average = sum(values[-3:]) / min(3, len(values))
        trend = (values[-1] - values[0]) / max(1, len(values) - 1)
        prediction = max(0.0, moving_average + trend)

    method = "3-month moving average with simple trend adjustment"
    explanation = (
        f"Used {len(values)} monthly sales buckets. Last values: {values[-3:]}. "
        f"Trend adjustment: {trend:.1f}. Predicted next-month demand: {prediction:.1f} units."
    )
    with get_connection() as connection:
        product_id = history.get("product", {}).get("id") if history.get("product") else None
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
        "top_products": sorted(product_totals.items(), key=lambda item: item[1], reverse=True)[:5],
    }


def reorder_recommendations(identifier: str | None = None) -> list[dict[str, Any]]:
    products = [find_product(identifier)] if identifier else low_stock_products() + out_of_stock_products()
    recommendations: list[dict[str, Any]] = []
    for product in [item for item in products if item]:
        forecast = demand_forecast(product["sku"], months=6)
        predicted = forecast.get("predicted_quantity", 0) if forecast.get("found") else product["reorder_quantity"]
        target = max(product["reorder_quantity"], int(predicted) + product["reorder_level"])
        needed = max(0, target - product["stock_quantity"])
        recommendations.append(
            {
                "product_id": product["id"],
                "sku": product["sku"],
                "name": product["name"],
                "current_stock": product["stock_quantity"],
                "reorder_level": product["reorder_level"],
                "recommended_quantity": needed,
                "supplier": product.get("supplier_name"),
                "reason": f"Target stock {target} based on reorder policy and demand forecast.",
            }
        )
    return recommendations


def add_inventory_movement(product_id: int, movement_type: str, quantity: int, reason: str = "") -> dict[str, Any]:
    if movement_type not in {"stock_in", "stock_out", "adjustment"}:
        raise ValueError("movement_type must be stock_in, stock_out, or adjustment")
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    product = get_product(product_id)
    if not product:
        raise ValueError(f"Product {product_id} not found")

    delta = quantity if movement_type == "stock_in" else -quantity
    if movement_type == "adjustment":
        delta = quantity - product["stock_quantity"]

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO inventory_movements (product_id, movement_type, quantity, reason, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (product_id, movement_type, quantity, reason, utc_now()),
        )
        connection.execute(
            "UPDATE products SET stock_quantity = stock_quantity + ?, updated_at = ? WHERE id = ?",
            (delta, utc_now(), product_id),
        )
    return {"product": get_product(product_id), "movement_type": movement_type, "quantity": quantity}


def write_inventory_report(title: str = "AskMamma Operations Report") -> dict[str, Any]:
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    low = low_stock_products()
    out = out_of_stock_products()
    recs = reorder_recommendations()
    forecast = demand_forecast(months=6)
    file_name = f"askmamma-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    path = config.REPORT_DIR / file_name
    lines = [
        f"# {title}",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Executive Summary",
        f"- Low-stock products: {len(low)}",
        f"- Out-of-stock products: {len(out)}",
        f"- Forecast: {forecast.get('explanation', forecast.get('message'))}",
        "",
        "## Low-Stock Products",
        *[f"- {p['sku']} {p['name']}: {p['stock_quantity']} on hand, reorder level {p['reorder_level']}" for p in low],
        "",
        "## Out-of-Stock Products",
        *[f"- {p['sku']} {p['name']}" for p in out],
        "",
        "## Reorder Recommendations",
        *[f"- {r['sku']} {r['name']}: order {r['recommended_quantity']} from {r['supplier']}" for r in recs],
        "",
        "## Risks",
        "- Products below reorder level may stock out before the next supplier delivery.",
        "",
        "## Next Actions",
        "- Review recommended reorder quantities.",
        "- Confirm supplier lead times before purchase orders.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    with get_connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO long_term_memory (memory_key, memory_value, created_at, updated_at)
            VALUES (?, ?, COALESCE((SELECT created_at FROM long_term_memory WHERE memory_key = ?), ?), ?)
            """,
            ("last_report_path", str(path), "last_report_path", utc_now(), utc_now()),
        )
    return {"path": str(path), "summary": f"Saved report to {path}"}


def audit_log(session_id: str, message: str) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO agent_traces (session_id, user_input, selected_agent, final_answer, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, message, "AuditLogTool", message, utc_now()),
        )


def summarize_tools_for_trace(tool_outputs: list[Any]) -> str:
    summaries = []
    for output in tool_outputs:
        text = json.dumps(output, default=str)
        summaries.append(text[:500])
    return json.dumps(summaries)
