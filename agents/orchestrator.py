"""Hierarchical inventory agent orchestration with memory and tracing."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from db.database import get_connection, initialize_database, list_products, rows_to_dicts, utc_now
from rag.retrieval import document_search
from inventory.tools import (
    demand_forecast,
    inventory_status,
    product_search,
    reorder_recommendations,
    sales_history,
    supplier_lookup,
    summarize_tools_for_trace,
    write_inventory_report,
)


SYSTEM_PROMPT = """
You are AskMamma Inventory Assistant, a tool-using AI agent for inventory management.
Never invent product data, stock levels, supplier details, or forecast numbers.
Use tools for inventory, supplier, sales, forecast, document, and report answers.
If sales history is insufficient, say so and provide a conservative fallback.
Confirm before destructive actions such as deleting products or major stock adjustments.
Do not reveal secrets or environment variables.
""".strip()


def save_message(session_id: str, role: str, content: str) -> None:
    initialize_database()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO chat_history (session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, role, content, utc_now()),
        )


def get_session_messages(session_id: str, limit: int = 20) -> list[dict[str, Any]]:
    initialize_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT role, content, created_at
            FROM chat_history
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    return list(reversed(rows_to_dicts(rows)))


def get_recent_traces(limit: int = 50) -> list[dict[str, Any]]:
    initialize_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM agent_traces
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return rows_to_dicts(rows)


def _extract_identifier(message: str, session_id: str) -> str | None:
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", message)
    if quoted:
        return quoted[0]
    lowered = message.lower()
    for product in list_products(limit=500):
        if product["sku"].lower() in lowered or product["name"].lower() in lowered:
            return product["sku"]
    stop = {
        "which",
        "products",
        "product",
        "are",
        "is",
        "low",
        "stock",
        "supplier",
        "provides",
        "available",
        "have",
        "do",
        "we",
        "what",
        "about",
        "its",
        "that",
        "item",
        "forecast",
        "demand",
        "next",
        "month",
        "for",
        "based",
        "previous",
        "expect",
        "this",
        "week",
    }
    words = [word for word in re.findall(r"[a-zA-Z0-9-]+", lowered) if word not in stop]
    if words:
        candidate = " ".join(words[-3:])
        if len(candidate) > 2:
            return candidate

    for previous in reversed(get_session_messages(session_id)):
        content = previous["content"]
        content_lower = content.lower()
        for product in list_products(limit=500):
            if product["sku"].lower() in content_lower or product["name"].lower() in content_lower:
                return product["sku"]
        match = re.search(r"(?:SKU\s+)?([A-Z]{2,}-\d{3})", content)
        if match:
            return match.group(1)
    return None


@dataclass
class AgentResult:
    answer: str
    selected_agent: str
    tools_called: list[str] = field(default_factory=list)
    tool_inputs: list[dict[str, Any]] = field(default_factory=list)
    tool_outputs: list[Any] = field(default_factory=list)


class InventoryAgent:
    name = "InventoryAgent"

    def run(self, message: str, session_id: str) -> AgentResult:
        lowered = message.lower()
        identifier = _extract_identifier(message, session_id)
        if "supplier" in lowered and identifier:
            result = supplier_lookup(identifier)
            if not result.get("found"):
                answer = result["message"]
            else:
                supplier = result["supplier"]
                answer = (
                    f"**{result['product']}** is supplied by {supplier['name']} "
                    f"(email: {supplier['email']}, lead time: {supplier['lead_time_days']} days)."
                )
            return AgentResult(answer, self.name, ["SupplierLookupTool"], [{"identifier": identifier}], [result])

        if "low" in lowered and "stock" in lowered:
            result = inventory_status()
            low = result["low_stock"]
            if not low:
                answer = "No products are currently below reorder level."
            else:
                answer = "Low-stock products:\n" + "\n".join(
                    f"- {p['sku']} **{p['name']}**: {p['stock_quantity']} on hand, reorder level {p['reorder_level']}"
                    for p in low
                )
            return AgentResult(answer, self.name, ["InventoryStatusTool"], [{"identifier": None}], [result])

        if "out" in lowered and "stock" in lowered:
            result = inventory_status()
            out = result["out_of_stock"]
            answer = "Out-of-stock products:\n" + "\n".join(
                f"- {p['sku']} **{p['name']}**" for p in out
            ) if out else "No products are out of stock."
            return AgentResult(answer, self.name, ["InventoryStatusTool"], [{"identifier": None}], [result])

        if identifier:
            result = inventory_status(identifier)
            if not result.get("found"):
                search = product_search(identifier)
                answer = "No exact stock match found. Similar products:\n" + "\n".join(
                    f"- {p['sku']} **{p['name']}** ({p['stock_quantity']} on hand)" for p in search[:5]
                )
                return AgentResult(answer, self.name, ["InventoryStatusTool", "ProductSearchTool"], [{"identifier": identifier}, {"query": identifier}], [result, search])
            product = result["product"]
            answer = (
                f"**{product['name']}** ({product['sku']}) has {product['stock_quantity']} units on hand. "
                f"Reorder level: {product['reorder_level']}. Price: ${product['price']:.2f}. "
                f"Supplier: {product.get('supplier_name')}. "
                f"Status: {'out of stock' if result['out_of_stock'] else 'low stock' if result['low_stock'] else 'in stock'}."
            )
            return AgentResult(answer, self.name, ["InventoryStatusTool"], [{"identifier": identifier}], [result])

        result = product_search(message)
        answer = "Matching products:\n" + "\n".join(
            f"- {p['sku']} **{p['name']}**: {p['stock_quantity']} on hand" for p in result[:10]
        )
        return AgentResult(answer, self.name, ["ProductSearchTool"], [{"query": message}], [result])


class ForecastAgent:
    name = "ForecastAgent"

    def run(self, message: str, session_id: str) -> AgentResult:
        identifier = _extract_identifier(message, session_id)
        history = sales_history(identifier, months=6)
        forecast = demand_forecast(identifier, months=6)
        if not forecast.get("found"):
            answer = forecast["message"]
        else:
            answer = (
                f"Expected next-month demand is **{forecast['predicted_quantity']} units**. "
                f"{forecast['explanation']}"
            )
        return AgentResult(
            answer,
            self.name,
            ["SalesHistoryTool", "DemandForecastTool"],
            [{"identifier": identifier, "months": 6}, {"identifier": identifier, "months": 6}],
            [history, forecast],
        )


class DocumentAgent:
    name = "DocumentAgent"

    def run(self, message: str, session_id: str) -> AgentResult:
        result = document_search(message)
        if not result.get("found"):
            answer = result["message"]
        else:
            lines = ["I found these document references:"]
            for item in result["results"][:3]:
                page = f", page {item['page_number']}" if item.get("page_number") else ""
                lines.append(f"- {item['file_name']}{page}: {item['text'][:220].strip()}...")
            answer = "\n".join(lines)
        return AgentResult(answer, self.name, ["DocumentSearchTool"], [{"query": message}], [result])


class ReportAgent:
    name = "ReportAgent"

    def run(self, message: str, session_id: str) -> AgentResult:
        report = write_inventory_report("Inventory Management Report")
        recs = reorder_recommendations()
        answer = f"{report['summary']}. It includes {len(recs)} reorder recommendations."
        return AgentResult(
            answer,
            self.name,
            ["ReportWriterTool", "ReorderRecommendationTool"],
            [{"title": "Inventory Management Report"}, {"identifier": None}],
            [report, recs],
        )


class QualityReviewAgent:
    name = "QualityReviewAgent"

    def run(self, result: AgentResult) -> AgentResult:
        if result.selected_agent in {"ForecastAgent", "ReportAgent"} and not result.tools_called:
            result.answer += "\n\nQuality note: no tool evidence was available for this answer."
        if "forecast" in result.answer.lower() and "units" not in result.answer.lower():
            result.answer += "\n\nQuality note: forecast answers should include quantities when data is available."
        result.tools_called.append("QualityReviewAgent")
        result.tool_inputs.append({"answer": result.answer[:300]})
        result.tool_outputs.append({"review": "passed"})
        return result


class SupervisorAgent:
    name = "SupervisorAgent"

    def __init__(self) -> None:
        self.inventory_agent = InventoryAgent()
        self.forecast_agent = ForecastAgent()
        self.document_agent = DocumentAgent()
        self.report_agent = ReportAgent()
        self.quality_agent = QualityReviewAgent()

    def route(self, message: str) -> str:
        lowered = message.lower()
        if lowered.strip() in {"hi", "hello", "hey", "good morning", "good afternoon"}:
            return "greeting"
        if any(word in lowered for word in ["document", "policy", "manual", "contract", "return", "uploaded"]):
            return "document"
        if any(word in lowered for word in ["forecast", "demand", "next month", "sales history", "high-demand"]):
            return "forecast"
        if any(word in lowered for word in ["report", "summary"]):
            return "report"
        if any(word in lowered for word in ["reorder", "restock", "stockout", "supplier", "stock", "available", "product", "sku"]):
            return "inventory"
        return "inventory"

    def run(self, message: str, session_id: str) -> AgentResult:
        route = self.route(message)
        if route == "greeting":
            return AgentResult(
                "Hello. I can help with stock checks, suppliers, low-stock analysis, demand forecasts, reports, and document search.",
                self.name,
            )
        if route == "document":
            result = self.document_agent.run(message, session_id)
        elif route == "forecast":
            result = self.forecast_agent.run(message, session_id)
            result = self.quality_agent.run(result)
        elif route == "report":
            result = self.report_agent.run(message, session_id)
            result = self.quality_agent.run(result)
        else:
            result = self.inventory_agent.run(message, session_id)
        return result


def trace_run(
    session_id: str,
    user_input: str,
    result: AgentResult,
    latency_ms: int,
    error: str | None = None,
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO agent_traces (
                session_id, user_input, selected_agent, tools_called, tool_inputs,
                tool_outputs_summary, final_answer, latency_ms, errors, token_usage, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                user_input,
                result.selected_agent,
                json.dumps(result.tools_called),
                json.dumps(result.tool_inputs, default=str),
                summarize_tools_for_trace(result.tool_outputs),
                result.answer,
                latency_ms,
                error,
                "{}",
                utc_now(),
            ),
        )


def invoke_agent(user_input: str, session_id: str | None = None) -> dict[str, Any]:
    initialize_database()
    session_id = session_id or str(uuid.uuid4())
    start = time.perf_counter()
    save_message(session_id, "user", user_input)
    try:
        result = SupervisorAgent().run(user_input, session_id)
        error = None
    except Exception as exc:
        result = AgentResult(f"Sorry, I could not complete that request: {exc}", "SupervisorAgent")
        error = str(exc)
    latency_ms = int((time.perf_counter() - start) * 1000)
    save_message(session_id, "assistant", result.answer)
    trace_run(session_id, user_input, result, latency_ms, error)
    return {
        "session_id": session_id,
        "answer": result.answer,
        "selected_agent": result.selected_agent,
        "tools_called": result.tools_called,
        "tool_inputs": result.tool_inputs,
        "latency_ms": latency_ms,
    }
