"""Typed AskMamma demo tools exposed to LangChain, FastAPI, and MCP adapters."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from typing import Any

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from core import config
from core.observability import redact_payload
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
from rag.retrieval import document_search


class ToolCall(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]


class DemoItemLookupInput(BaseModel):
    query: str = Field(..., min_length=1, description="Demo item name, SKU, category, or descriptive search.")


class DemoAvailabilityInput(BaseModel):
    identifier: str | None = Field(default=None, description="Optional sample demo item SKU or name.")


class DemoPartnerLookupInput(BaseModel):
    identifier: str = Field(..., min_length=1, description="Sample demo item SKU or name.")


class DemoHistoryInput(BaseModel):
    identifier: str | None = Field(default=None, description="Optional sample demo item SKU or name.")
    months: int = Field(default=6, ge=1, le=24, description="How many months of demo history to inspect.")


class DemoForecastInput(BaseModel):
    identifier: str | None = Field(default=None, description="Optional sample demo item SKU or name.")
    months: int = Field(default=6, ge=1, le=24, description="How many months of history to use for the forecast.")


class DemoRecommendationInput(BaseModel):
    identifier: str | None = Field(default=None, description="Optional sample demo item SKU or name.")


class DocumentSearchInput(BaseModel):
    query: str = Field(..., min_length=1, description="Question or search string for AskMamma documents.")
    limit: int = Field(default=5, ge=1, le=10, description="Maximum number of retrieved chunks.")


class DemoReportInput(BaseModel):
    title: str = Field(default="AskMamma Operations Report", min_length=3, max_length=120)
    output_format: str = Field(default="md", description="One of xlsx, txt, json, or md.")


class DemoMovementInput(BaseModel):
    product_id: int = Field(..., ge=1, description="Sample demo item ID.")
    movement_type: str = Field(..., description="One of stock_in, stock_out, or adjustment.")
    quantity: int = Field(..., gt=0, description="Quantity to add, remove, or set.")
    reason: str = Field(default="demo adjustment", max_length=200)
    confirm: bool = Field(default=False, description="Must be true for any movement write.")


class AuditLogInput(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1, max_length=1000)


def demo_item_lookup(query: str) -> list[dict[str, Any]]:
    return list_products(search=query, limit=25)


def demo_item_search(query: str) -> list[dict[str, Any]]:
    return demo_item_lookup(query)


def demo_status(identifier: str | None = None) -> dict[str, Any]:
    if identifier:
        item = find_product(identifier)
        if not item:
            return {"found": False, "message": f"No sample demo item found for `{identifier}`."}
        return {
            "found": True,
            "item": item,
            "low_stock": item["stock_quantity"] > 0 and item["stock_quantity"] <= item["reorder_level"],
            "out_of_stock": item["stock_quantity"] <= 0,
        }
    return {
        "found": True,
        "low_stock": low_stock_products(),
        "out_of_stock": out_of_stock_products(),
    }


def demo_partner_lookup(identifier: str) -> dict[str, Any]:
    item = find_product(identifier)
    if not item:
        return {"found": False, "message": f"No sample demo item found for `{identifier}`."}
    return {
        "found": True,
        "item": item["name"],
        "partner": {
            "name": item.get("supplier_name"),
            "email": item.get("contact_email"),
            "lead_time_days": item.get("lead_time_days"),
        },
    }


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
        return {
            "found": False,
            "message": "Insufficient sample history for a numeric forecast. Use the demo threshold instead.",
        }

    monthly: dict[str, int] = defaultdict(int)
    item_totals: dict[str, int] = defaultdict(int)
    for record in history["records"]:
        month = record["sale_date"][:7]
        monthly[month] += int(record["quantity_sold"])
        item_totals[record["product_name"]] += int(record["quantity_sold"])

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


def add_demo_movement(
    product_id: int,
    movement_type: str,
    quantity: int,
    reason: str = "",
    confirm: bool = False,
) -> dict[str, Any]:
    if not confirm:
        raise ValueError("Human confirmation is required before writing a demo movement.")
    if movement_type not in {"stock_in", "stock_out", "adjustment"}:
        raise ValueError("movement_type must be stock_in, stock_out, or adjustment")
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    item = get_product(product_id)
    if not item:
        raise ValueError(f"Demo item {product_id} not found")

    delta = quantity if movement_type == "stock_in" else -quantity
    if movement_type == "adjustment":
        delta = quantity - item["stock_quantity"]

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


def write_demo_report(title: str = "AskMamma Operations Report", output_format: str = "md") -> dict[str, Any]:
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    low = low_stock_products()
    out = out_of_stock_products()
    recs = demo_reorder_recommendations()
    forecast = demo_forecast(months=6)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_format = output_format.lower()
    file_name = f"askmamma-report-{timestamp}.{output_format}"
    path = config.REPORT_DIR / file_name
    summary_rows = [
        {"Metric": "Report Title", "Value": title},
        {"Metric": "Generated At", "Value": utc_now()},
        {"Metric": "Low-Availability Demo Items", "Value": len(low)},
        {"Metric": "Unavailable Demo Items", "Value": len(out)},
        {"Metric": "Forecast Snapshot", "Value": forecast.get("explanation", forecast.get("message"))},
        {"Metric": "Notes", "Value": "Calculated from local inventory/demo data. AI may explain outputs but does not invent stock or forecast numbers."},
    ]
    if output_format == "json":
        path.write_text(
            json.dumps(
                {
                    "summary": summary_rows,
                    "low_stock": low,
                    "out_of_stock": out,
                    "recommendations": recs,
                    "forecast": forecast,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    elif output_format == "md":
        lines = [f"# {title}", "", f"Generated: {utc_now()}", "", "## Summary"]
        lines.extend([f"- {row['Metric']}: {row['Value']}" for row in summary_rows])
        lines.extend(["", "## Reorder Recommendations"])
        lines.extend([f"- {item['sku']} {item['name']}: recommend {item['recommended_quantity']}" for item in recs] or ["- None"])
        path.write_text("\n".join(lines), encoding="utf-8")
    elif output_format == "txt":
        lines = [title, f"Generated: {utc_now()}", ""]
        lines.extend([f"{row['Metric']}: {row['Value']}" for row in summary_rows])
        path.write_text("\n".join(lines), encoding="utf-8")
    else:
        if output_format != "xlsx":
            output_format = "xlsx"
            path = config.REPORT_DIR / f"askmamma-report-{timestamp}.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
            pd.DataFrame(low or [{"message": "No low-availability demo items"}]).to_excel(
                writer,
                sheet_name="Low Stock",
                index=False,
            )
            pd.DataFrame(out or [{"message": "No out-of-stock demo items"}]).to_excel(
                writer,
                sheet_name="Out of Stock",
                index=False,
            )
            pd.DataFrame(recs or [{"message": "No reorder recommendations"}]).to_excel(
                writer,
                sheet_name="Reorder Recommendations",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "method": forecast.get("method"),
                        "predicted_quantity": forecast.get("predicted_quantity"),
                        "explanation": forecast.get("explanation", forecast.get("message")),
                    }
                ]
            ).to_excel(writer, sheet_name="Forecast", index=False)
    with get_connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO long_term_memory (memory_key, memory_value, created_at, updated_at)
            VALUES (?, ?, COALESCE((SELECT created_at FROM long_term_memory WHERE memory_key = ?), ?), ?)
            """,
            ("last_report_path", str(path), "last_report_path", utc_now(), utc_now()),
        )
    return {
        "path": str(path),
        "file_name": path.name,
        "summary": f"Saved report to {path}",
        "download_name": path.name,
    }


def audit_log(session_id: str, message: str) -> dict[str, Any]:
    payload = {"session_id": session_id, "message": message}
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO agent_traces (session_id, user_input, selected_agent, final_answer, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, redact_payload(message), "AuditLogTool", redact_payload(message), utc_now()),
        )
    return {"saved": True, "entry": payload}


def summarize_tools_for_trace(tool_outputs: list[Any]) -> str:
    summaries = []
    for output in tool_outputs:
        text = json.dumps(redact_payload(output), default=str)
        summaries.append(text[:500])
    return json.dumps(summaries)


def tool_registry() -> list[ToolCall]:
    return [
        ToolCall(
            name="DemoItemLookupTool",
            description="Search sample AskMamma demo items by name, SKU, category, partner, or description.",
            input_schema=DemoItemLookupInput.model_json_schema(),
            output_schema={"type": "array", "items": {"type": "object"}},
        ),
        ToolCall(
            name="DemoAvailabilityTool",
            description="Return quantity, threshold, low-availability status, and unavailable status for sample demo items.",
            input_schema=DemoAvailabilityInput.model_json_schema(),
            output_schema={"type": "object"},
        ),
        ToolCall(
            name="DemoPartnerLookupTool",
            description="Find partner information for a sample demo item.",
            input_schema=DemoPartnerLookupInput.model_json_schema(),
            output_schema={"type": "object"},
        ),
        ToolCall(
            name="DemoHistoryTool",
            description="Retrieve sample history for a demo item or category.",
            input_schema=DemoHistoryInput.model_json_schema(),
            output_schema={"type": "object"},
        ),
        ToolCall(
            name="DemoForecastTool",
            description="Predict demo demand using moving average and trend adjustment based on sample history.",
            input_schema=DemoForecastInput.model_json_schema(),
            output_schema={"type": "object"},
        ),
        ToolCall(
            name="DemoRecommendationTool",
            description="Recommend sample replenishment quantities from demo availability, thresholds, history, and lead time.",
            input_schema=DemoRecommendationInput.model_json_schema(),
            output_schema={"type": "array", "items": {"type": "object"}},
        ),
        ToolCall(
            name="DocumentSearchTool",
            description="Search indexed AskMamma documents using local embedding retrieval.",
            input_schema=DocumentSearchInput.model_json_schema(),
            output_schema={"type": "object"},
        ),
        ToolCall(
            name="DemoReportWriterTool",
            description="Generate a sample AskMamma operations Excel report and save it under outputs/reports.",
            input_schema=DemoReportInput.model_json_schema(),
            output_schema={"type": "object"},
        ),
        ToolCall(
            name="DemoMovementTool",
            description="Record sample stock-in, stock-out, or adjustment movement after explicit human confirmation.",
            input_schema=DemoMovementInput.model_json_schema(),
            output_schema={"type": "object"},
        ),
        ToolCall(
            name="AuditLogTool",
            description="Persist a sanitized local audit record for the current AskMamma session.",
            input_schema=AuditLogInput.model_json_schema(),
            output_schema={"type": "object"},
        ),
    ]


def langchain_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            name="DemoItemLookupTool",
            description="Search sample AskMamma demo items by name, SKU, category, partner, or description.",
            func=demo_item_lookup,
            args_schema=DemoItemLookupInput,
        ),
        StructuredTool.from_function(
            name="DemoAvailabilityTool",
            description="Return quantity, threshold, low-availability status, and unavailable status for sample demo items.",
            func=demo_status,
            args_schema=DemoAvailabilityInput,
        ),
        StructuredTool.from_function(
            name="DemoPartnerLookupTool",
            description="Find partner information for a sample demo item.",
            func=demo_partner_lookup,
            args_schema=DemoPartnerLookupInput,
        ),
        StructuredTool.from_function(
            name="DemoHistoryTool",
            description="Retrieve sample history for a demo item or category.",
            func=demo_history,
            args_schema=DemoHistoryInput,
        ),
        StructuredTool.from_function(
            name="DemoForecastTool",
            description="Predict demo demand using moving average and trend adjustment based on sample history.",
            func=demo_forecast,
            args_schema=DemoForecastInput,
        ),
        StructuredTool.from_function(
            name="DemoRecommendationTool",
            description="Recommend sample replenishment quantities from demo availability, thresholds, history, and lead time.",
            func=demo_reorder_recommendations,
            args_schema=DemoRecommendationInput,
        ),
        StructuredTool.from_function(
            name="DocumentSearchTool",
            description="Search indexed AskMamma documents using local embedding retrieval.",
            func=document_search,
            args_schema=DocumentSearchInput,
        ),
        StructuredTool.from_function(
            name="DemoReportWriterTool",
            description="Generate a sample AskMamma operations Excel report and save it under outputs/reports.",
            func=write_demo_report,
            args_schema=DemoReportInput,
        ),
        StructuredTool.from_function(
            name="DemoMovementTool",
            description="Record sample stock-in, stock-out, or adjustment movement after explicit human confirmation.",
            func=add_demo_movement,
            args_schema=DemoMovementInput,
        ),
        StructuredTool.from_function(
            name="AuditLogTool",
            description="Persist a sanitized local audit record for the current AskMamma session.",
            func=audit_log,
            args_schema=AuditLogInput,
        ),
    ]


def get_tool_by_name(name: str) -> StructuredTool:
    for tool in langchain_tools():
        if tool.name == name:
            return tool
    raise KeyError(f"Unknown tool `{name}`")


def invoke_named_tool(name: str, arguments: dict[str, Any]) -> Any:
    tool = get_tool_by_name(name)
    return tool.invoke(arguments)
