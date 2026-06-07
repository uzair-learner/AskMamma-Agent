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
        ToolCall(name="DemoItemSearchTool", description="Search sample AskMamma demo items by name, SKU, category, partner, or description.", input_schema={"query": "string"}),
        ToolCall(name="DemoAvailabilityTool", description="Return quantity, threshold, low-availability status, and unavailable status for sample demo items.", input_schema={"identifier": "string optional"}),
        ToolCall(name="DemoPartnerLookupTool", description="Find partner information for a sample demo item.", input_schema={"identifier": "string"}),
        ToolCall(name="DemoHistoryTool", description="Retrieve sample history for a demo item or category.", input_schema={"identifier": "string optional", "months": "integer"}),
        ToolCall(name="DemoForecastTool", description="Predict demo demand using moving average and trend adjustment based on sample history.", input_schema={"identifier": "string optional", "months": "integer"}),
        ToolCall(name="DemoRecommendationTool", description="Recommend sample replenishment quantities from demo availability, thresholds, history, and lead time.", input_schema={"identifier": "string optional"}),
        ToolCall(name="DocumentSearchTool", description="Search indexed documents using local RAG retrieval.", input_schema={"query": "string"}),
        ToolCall(name="DemoReportWriterTool", description="Generate a sample AskMamma operations report and save it under outputs/reports.", input_schema={"title": "string optional"}),
        ToolCall(name="DemoMovementTool", description="Record sample stock-in, stock-out, or adjustment movement.", input_schema={"product_id": "integer", "movement_type": "string", "quantity": "integer", "reason": "string"}),
        ToolCall(name="AuditLogTool", description="Save agent actions and tool calls.", input_schema={"session_id": "string", "message": "string"}),
    ]


def demo_item_search(query: str) -> list[dict[str, Any]]:
    return list_products(search=query, limit=25)


def demo_status(identifier: str | None = None) -> dict[str, Any]:
    if identifier:
        product = find_product(identifier)
        if not product:
            return {"found": False, "message": f"No sample demo item found for `{identifier}`."}
        return {
            "found": True,
            "item": product,
            "low_stock": product["stock_quantity"] > 0 and product["stock_quantity"] <= product["reorder_level"],
            "out_of_stock": product["stock_quantity"] <= 0,
        }
    return {"low_stock": low_stock_products(), "out_of_stock": out_of_stock_products()}


def demo_partner_lookup(identifier: str) -> dict[str, Any]:
    product = find_product(identifier)
    if not product:
        return {"found": False, "message": f"No sample item found for `{identifier}`."}
    return {
        "found": True,
        "item": product["name"],
        "partner": {
            "name": product.get("supplier_name"),
            "email": product.get("contact_email"),
            "lead_time_days": product.get("lead_time_days"),
        },
    }


def demo_history(identifier: str | None = None, months: int = 6) -> dict[str, Any]:
    params: list[Any] = [f"-{months * 30} days"]
    product_clause = ""
    product = None
    if identifier:
        product = find_product(identifier)
        if not product:
            return {"found": False, "message": f"No sample item found for `{identifier}`."}
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
    return {"found": bool(records), "item": product, "records": records}


def demo_forecast(identifier: str | None = None, months: int = 6) -> dict[str, Any]:
    history = demo_history(identifier, months)
    if not history.get("found"):
        return {
            "found": False,
            "message": "Insufficient sample history for a numeric forecast. Use the demo threshold as a fallback.",
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
        "top_items": sorted(product_totals.items(), key=lambda item: item[1], reverse=True)[:5],
    }


def demo_reorder_recommendations(identifier: str | None = None) -> list[dict[str, Any]]:
    products = [find_product(identifier)] if identifier else low_stock_products() + out_of_stock_products()
    recommendations: list[dict[str, Any]] = []
    for product in [item for item in products if item]:
        forecast = demo_forecast(product["sku"], months=6)
        predicted = forecast.get("predicted_quantity", 0) if forecast.get("found") else product["reorder_quantity"]
        target = max(product["reorder_quantity"], int(predicted) + product["reorder_level"])
        needed = max(0, target - product["stock_quantity"])
        recommendations.append(
            {
                "item_id": product["id"],
                "sku": product["sku"],
                "name": product["name"],
                "current_stock": product["stock_quantity"],
                "reorder_level": product["reorder_level"],
                "recommended_quantity": needed,
                "supplier": product.get("supplier_name"),
                "reason": f"Target quantity {target} based on demo threshold policy and demo forecast.",
            }
        )
    return recommendations


def add_demo_movement(product_id: int, movement_type: str, quantity: int, reason: str = "") -> dict[str, Any]:
    if movement_type not in {"stock_in", "stock_out", "adjustment"}:
        raise ValueError("movement_type must be stock_in, stock_out, or adjustment")
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    product = get_product(product_id)
    if not product:
        raise ValueError(f"Demo item {product_id} not found")

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
    return {"item": get_product(product_id), "movement_type": movement_type, "quantity": quantity}


def write_demo_report(title: str = "AskMamma Operations Report") -> dict[str, Any]:
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    low = low_stock_products()
    out = out_of_stock_products()
    recs = demo_reorder_recommendations()
    forecast = demo_forecast(months=6)
    file_name = f"askmamma-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    path = config.REPORT_DIR / file_name
    lines = [
        f"# {title}",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Executive Summary",
        f"- Low-availability demo items: {len(low)}",
        f"- Unavailable demo items: {len(out)}",
        f"- Forecast snapshot: {forecast.get('explanation', forecast.get('message'))}",
        "",
        "## Low-Availability Demo Items",
        *[f"- {p['sku']} {p['name']}: {p['stock_quantity']} on hand, demo threshold {p['reorder_level']} (sample)" for p in low],
        "",
        "## Unavailable Demo Items",
        *[f"- {p['sku']} {p['name']} (sample)" for p in out],
        "",
        "## Demo Replenishment Recommendations",
        *[f"- {r.get('sku')} {r.get('name')}: replenish {r.get('recommended_quantity')} from {r.get('supplier')} (sample)" for r in recs],
        "",
        "## Risks",
        "- Demo items below their threshold may become unavailable before the next sample partner delivery.",
        "",
        "## Next Actions",
        "- Review recommended replenishment quantities.",
        "- Confirm sample partner lead times before acting on the demo scenario.",
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
