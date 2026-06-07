"""Hierarchical AskMamma agent orchestration with memory and tracing."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from db.database import get_connection, initialize_database, list_products, rows_to_dicts, utc_now
from rag.retrieval import document_search
from askmamma.tools import (
    demo_forecast,
    demo_status,
    demo_item_search,
    demo_reorder_recommendations,
    demo_history,
    demo_partner_lookup,
    summarize_tools_for_trace,
    write_demo_report,
)


SYSTEM_PROMPT = """
You are AskMamma Assistant, a tool-using AI agent for AskMamma demo operations.
Never invent sample demo item data, availability levels, partner details, or forecast numbers.
Use tools for demo availability, partners, history, forecasts, documents, and reports.
If demo history is insufficient, say so and provide a conservative fallback.
Confirm before destructive actions such as deleting demo items or major availability adjustments.
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


def remember_session_value(session_id: str, key: str, value: str) -> None:
    initialize_database()
    memory_key = f"session:{session_id}:{key}"
    with get_connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO long_term_memory (memory_key, memory_value, created_at, updated_at)
            VALUES (?, ?, COALESCE((SELECT created_at FROM long_term_memory WHERE memory_key = ?), ?), ?)
            """,
            (memory_key, value, memory_key, utc_now(), utc_now()),
        )


def recall_session_value(session_id: str, key: str) -> str | None:
    initialize_database()
    memory_key = f"session:{session_id}:{key}"
    with get_connection() as connection:
        row = connection.execute(
            "SELECT memory_value FROM long_term_memory WHERE memory_key = ?",
            (memory_key,),
        ).fetchone()
    return row["memory_value"] if row else None


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
    for item in list_products(limit=500):
        if item["sku"].lower() in lowered or item["name"].lower() in lowered:
            return item["sku"]
    stop = {
        "which",
        "products",
        "product",
        "items",
        "item",
        "are",
        "is",
        "low",
        "stock",
        "availability",
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
        for item in list_products(limit=500):
            if item["sku"].lower() in content_lower or item["name"].lower() in content_lower:
                return item["sku"]
        match = re.search(r"(?:SKU\s+)?([A-Z]{2,}-\d{3})", content)
        if match:
            return match.group(1)
    remembered = recall_session_value(session_id, "last_sku")
    if remembered:
        return remembered
    return None


@dataclass
class AgentResult:
    answer: str
    selected_agent: str
    tools_called: list[str] = field(default_factory=list)
    tool_inputs: list[dict[str, Any]] = field(default_factory=list)
    tool_outputs: list[Any] = field(default_factory=list)


class AskMammaActionAgent:
    name = "AskMammaActionAgent"

    def run(self, message: str, session_id: str) -> AgentResult:
        lowered = message.lower()
        identifier = _extract_identifier(message, session_id)
        if ("supplier" in lowered or "partner" in lowered) and identifier:
            result = demo_partner_lookup(identifier)
            if not result.get("found"):
                answer = result["message"]
            else:
                item_info = demo_status(identifier).get("item")
                if item_info:
                    remember_session_value(session_id, "last_sku", item_info["sku"])
                partner = result.get("partner")
                answer = (
                    f"In the sample demo catalog, **{result.get('item')}** is supplied by {partner.get('name')} "
                    f"(email: {partner.get('email')}, lead time: {partner.get('lead_time_days')} days)."
                )
            return AgentResult(answer, self.name, ["DemoPartnerLookupTool"], [{"identifier": identifier}], [result])

        if "low" in lowered and ("stock" in lowered or "availability" in lowered):
            result = demo_status()
            low = result["low_stock"]
            if not low:
                answer = "No sample demo items are currently below their demo threshold."
            else:
                answer = "Low-availability demo items:\n" + "\n".join(
                    f"- {p['sku']} **{p['name']}**: {p['stock_quantity']} on hand, demo threshold {p['reorder_level']}"
                    for p in low
                )
            return AgentResult(answer, self.name, ["DemoAvailabilityTool"], [{"identifier": None}], [result])

        if ("out" in lowered or "unavailable" in lowered) and ("stock" in lowered or "availability" in lowered):
            result = demo_status()
            out = result["out_of_stock"]
            answer = "Unavailable demo items:\n" + "\n".join(
                f"- {p['sku']} **{p['name']}**" for p in out
            ) if out else "No sample demo items are unavailable."
            return AgentResult(answer, self.name, ["DemoAvailabilityTool"], [{"identifier": None}], [result])

        if identifier:
            result = demo_status(identifier)
            if not result.get("found"):
                search = demo_item_search(identifier)
                answer = "No exact demo item match found. Similar demo items:\n" + "\n".join(
                    f"- {p['sku']} **{p['name']}** ({p['stock_quantity']} on hand)" for p in search[:5]
                )
                return AgentResult(answer, self.name, ["DemoAvailabilityTool", "DemoItemSearchTool"], [{"identifier": identifier}, {"query": identifier}], [result, search])
            item = result.get("item")
            remember_session_value(session_id, "last_sku", item["sku"])
            answer = (
                f"In the sample demo catalog, **{item['name']}** ({item['sku']}) has {item['stock_quantity']} units on hand. "
                f"Demo threshold: {item['reorder_level']}. Price: ${item['price']:.2f}. "
                f"Partner: {item.get('supplier_name')}. "
                f"Status: {'unavailable' if result['out_of_stock'] else 'low availability' if result['low_stock'] else 'available'}."
            )
            return AgentResult(answer, self.name, ["DemoAvailabilityTool"], [{"identifier": identifier}], [result])

        result = demo_item_search(message)
        answer = "Matching demo items:\n" + "\n".join(
            f"- {p['sku']} **{p['name']}**: {p['stock_quantity']} on hand" for p in result[:10]
        )
        return AgentResult(answer, self.name, ["DemoItemSearchTool"], [{"query": message}], [result])


class ForecastAgent:
    name = "ForecastAgent"

    def run(self, message: str, session_id: str) -> AgentResult:
        identifier = _extract_identifier(message, session_id)
        history = demo_history(identifier, months=6)
        forecast = demo_forecast(identifier, months=6)
        if history.get("item"):
            remember_session_value(session_id, "last_sku", history["item"]["sku"])
        if not forecast.get("found"):
            answer = forecast["message"]
        else:
            answer = (
                f"Based on sample demo history, expected next-month demand is **{forecast['predicted_quantity']} units**. "
                f"{forecast['explanation']}"
            )
        return AgentResult(
            answer,
            self.name,
            ["DemoHistoryTool", "DemoForecastTool"],
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
        report = write_demo_report("AskMamma Operations Report")
        recs = demo_reorder_recommendations()
        answer = f"{report['summary']}. It includes {len(recs)} sample demo replenishment recommendations."
        return AgentResult(
            answer,
            self.name,
            ["DemoReportWriterTool", "DemoRecommendationTool"],
            [{"title": "AskMamma Operations Report"}, {"identifier": None}],
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
        self.action_agent = AskMammaActionAgent()
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
        if any(word in lowered for word in ["forecast", "demand", "next month", "sales history", "history", "high-demand"]):
            return "forecast"
        if any(word in lowered for word in ["report", "summary"]):
            return "report"
        if any(word in lowered for word in ["reorder", "restock", "stockout", "supplier", "partner", "stock", "availability", "available", "product", "item", "sku"]):
            return "actions"
        return "actions"

    def run(self, message: str, session_id: str) -> AgentResult:
        route = self.route(message)
        if route == "greeting":
            return AgentResult(
                "Hello. I can help with AskMamma demo items, availability checks, partners, forecasts, reports, and document search.",
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
            result = self.action_agent.run(message, session_id)
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
