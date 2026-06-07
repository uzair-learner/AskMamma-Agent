"""LangGraph-based AskMamma orchestration with deterministic fallback."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from askmamma.tools import (
    audit_log,
    demo_forecast,
    demo_history,
    demo_item_lookup,
    demo_partner_lookup,
    demo_reorder_recommendations,
    demo_status,
    langchain_tools,
    summarize_tools_for_trace,
    write_demo_report,
)
from core.llm_provider import current_runtime_status, get_chat_model, supports_langchain_agents
from core.observability import configure_langsmith, redact_payload, safe_error_message, tracing_backend_name
from db.database import get_connection, initialize_database, list_products, rows_to_dicts, utc_now
from rag.retrieval import document_search as document_search_tool


SYSTEM_PROMPT = """
You are AskMamma Assistant, a tool-using AI agent for AskMamma demo operations.
Never invent sample demo item data, availability levels, partner details, or forecast numbers.
Use tools for demo availability, partners, history, forecasts, documents, and reports.
If demo history is insufficient, say so and provide a conservative fallback.
Confirm before destructive actions such as deleting demo items or major availability adjustments.
Do not reveal secrets or environment variables.
Clearly label sample or demo data when answering item, availability, partner, forecast, or report questions.
""".strip()
LOGGER = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    user_input: str
    session_id: str
    route: str
    selected_agent: str
    answer: str
    tools_called: list[str]
    tool_inputs: list[dict[str, Any]]
    tool_outputs: list[Any]
    intermediate_steps: list[dict[str, Any]]
    route_path: list[str]
    review_notes: list[str]
    errors: list[str]
    trace_backend: str
    provider: str
    model: str
    llm_used: bool
    fallback_used: bool
    response_time_ms: int


@dataclass
class AgentResult:
    answer: str
    selected_agent: str
    tools_called: list[str] = field(default_factory=list)
    tool_inputs: list[dict[str, Any]] = field(default_factory=list)
    tool_outputs: list[Any] = field(default_factory=list)
    intermediate_steps: list[dict[str, Any]] = field(default_factory=list)
    route_path: list[str] = field(default_factory=list)
    review_notes: list[str] = field(default_factory=list)
    trace_backend: str = "sqlite"
    task_status: str = "completed"
    provider: str = "Deterministic fallback"
    model: str = "None"
    llm_used: bool = False
    fallback_used: bool = True
    response_time_ms: int = 0


def save_message(session_id: str, role: str, content: str) -> None:
    initialize_database()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO chat_history (session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, role, redact_payload(content), utc_now()),
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
    traces = rows_to_dicts(rows)
    for trace in traces:
        if isinstance(trace.get("tools_called"), str):
            try:
                trace["tools_called"] = json.loads(trace["tools_called"])
            except json.JSONDecodeError:
                pass
        try:
            metadata = json.loads(trace.get("token_usage") or "{}")
        except json.JSONDecodeError:
            metadata = {}
        trace["provider"] = metadata.get("provider", "Deterministic fallback")
        trace["model"] = metadata.get("model", "None")
        trace["llm_used"] = metadata.get("llm_used", False)
        trace["fallback_used"] = metadata.get("fallback_used", True)
        trace["response_time_ms"] = metadata.get("response_time_ms", trace.get("latency_ms", 0))
        trace["tools_called"] = metadata.get("tools_called", trace.get("tools_called", []))
    return traces


def _history_as_messages(session_id: str) -> list[Any]:
    messages: list[Any] = []
    for record in get_session_messages(session_id, limit=6):
        if record["role"] == "user":
            messages.append(HumanMessage(content=record["content"]))
        elif record["role"] == "assistant":
            messages.append(AIMessage(content=record["content"]))
    return messages


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
        "partner",
        "provides",
        "available",
        "have",
        "do",
        "we",
        "what",
        "about",
        "its",
        "that",
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
        "demo",
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


def _append_step(state: GraphState, step: dict[str, Any]) -> list[dict[str, Any]]:
    return [*state.get("intermediate_steps", []), redact_payload(step)]


def _append_route(state: GraphState, node_name: str) -> list[str]:
    return [*state.get("route_path", []), node_name]


def _route_message(message: str) -> str:
    lowered = message.lower()
    if lowered.strip() in {"hi", "hello", "hey", "good morning", "good afternoon"}:
        return "greeting"
    if any(word in lowered for word in ["document", "policy", "manual", "contract", "return", "uploaded"]):
        return "document"
    if any(word in lowered for word in ["forecast", "demand", "next month", "sales history", "history", "high-demand"]):
        return "forecast"
    if any(word in lowered for word in ["report", "summary"]):
        return "report"
    return "actions"


def _lc_tools(names: set[str]):
    return [tool for tool in langchain_tools() if tool.name in names]


def _run_langchain_tool_agent(
    agent_name: str,
    system_prompt: str,
    tool_names: set[str],
    message: str,
    session_id: str,
) -> AgentResult | None:
    if not supports_langchain_agents():
        return None

    llm = get_chat_model()
    if llm is None:
        return None

    tools = _lc_tools(tool_names)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        name=agent_name,
    )
    result = agent.invoke({"messages": [*_history_as_messages(session_id), HumanMessage(content=message)]})
    messages = result.get("messages", [])
    intermediate_steps: list[dict[str, Any]] = []
    tools_called = []
    tool_inputs = []
    tool_outputs = []
    pending_calls: dict[str, dict[str, Any]] = {}
    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in getattr(msg, "tool_calls", []) or []:
                tool_name = call.get("name", "unknown")
                tool_args = call.get("args", {})
                tools_called.append(tool_name)
                tool_inputs.append(tool_args if isinstance(tool_args, dict) else {"input": tool_args})
                step = {"agent": agent_name, "tool": tool_name, "tool_input": tool_args}
                pending_calls[call.get("id", tool_name)] = step
                intermediate_steps.append(step)
        elif isinstance(msg, ToolMessage):
            tool_outputs.append(msg.content)
            if msg.tool_call_id in pending_calls:
                pending_calls[msg.tool_call_id]["observation"] = redact_payload(msg.content)

    final_answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            final_answer = msg.text() if hasattr(msg, "text") else str(msg.content)
            break
    if not final_answer and messages:
        final_answer = str(messages[-1].content)
    return AgentResult(
        answer=final_answer,
        selected_agent=agent_name,
        tools_called=tools_called,
        tool_inputs=tool_inputs,
        tool_outputs=tool_outputs,
        intermediate_steps=intermediate_steps,
        provider=current_runtime_status()["provider"],
        model=current_runtime_status()["model"],
        llm_used=True,
        fallback_used=False,
    )


def _deterministic_action(message: str, session_id: str) -> AgentResult:
    lowered = message.lower()
    identifier = _extract_identifier(message, session_id)
    if "low" in lowered and ("stock" in lowered or "availability" in lowered):
        result = demo_status()
        low = result["low_stock"]
        if not low:
            answer = "No sample demo items are currently below their demo threshold."
        else:
            answer = "In the sample demo catalog, low-availability demo items are:\n" + "\n".join(
                f"- {p['sku']} **{p['name']}**: {p['stock_quantity']} on hand, demo threshold {p['reorder_level']}"
                for p in low
            )
        return AgentResult(answer, "AskMammaActionAgent", ["DemoAvailabilityTool"], [{"identifier": None}], [result])

    if ("out" in lowered or "unavailable" in lowered) and ("stock" in lowered or "availability" in lowered):
        result = demo_status()
        out = result["out_of_stock"]
        answer = (
            "In the sample demo catalog, unavailable demo items are:\n"
            + "\n".join(f"- {p['sku']} **{p['name']}**" for p in out)
            if out
            else "No sample demo items are unavailable."
        )
        return AgentResult(answer, "AskMammaActionAgent", ["DemoAvailabilityTool"], [{"identifier": None}], [result])

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
        return AgentResult(answer, "AskMammaActionAgent", ["DemoPartnerLookupTool"], [{"identifier": identifier}], [result])

    if identifier:
        result = demo_status(identifier)
        if not result.get("found"):
            search = demo_item_lookup(identifier)
            answer = "No exact demo item match found. Similar demo items:\n" + "\n".join(
                f"- {p['sku']} **{p['name']}** ({p['stock_quantity']} on hand)" for p in search[:5]
            )
            return AgentResult(
                answer,
                "AskMammaActionAgent",
                ["DemoAvailabilityTool", "DemoItemLookupTool"],
                [{"identifier": identifier}, {"query": identifier}],
                [result, search],
            )
        item = result.get("item")
        remember_session_value(session_id, "last_sku", item["sku"])
        answer = (
            f"In the sample demo catalog, **{item['name']}** ({item['sku']}) has {item['stock_quantity']} units on hand. "
            f"Demo threshold: {item['reorder_level']}. Price: ${item['price']:.2f}. "
            f"Partner: {item.get('supplier_name')}. "
            f"Status: {'unavailable' if result['out_of_stock'] else 'low availability' if result['low_stock'] else 'available'}."
        )
        return AgentResult(answer, "AskMammaActionAgent", ["DemoAvailabilityTool"], [{"identifier": identifier}], [result])

    result = demo_item_lookup(message)
    answer = "Matching demo items:\n" + "\n".join(
        f"- {p['sku']} **{p['name']}**: {p['stock_quantity']} on hand" for p in result[:10]
    )
    return AgentResult(answer, "AskMammaActionAgent", ["DemoItemLookupTool"], [{"query": message}], [result])


def _deterministic_forecast(message: str, session_id: str) -> AgentResult:
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
        "ForecastAgent",
        ["DemoHistoryTool", "DemoForecastTool"],
        [{"identifier": identifier, "months": 6}, {"identifier": identifier, "months": 6}],
        [history, forecast],
    )


def _deterministic_document(message: str) -> AgentResult:
    result = document_search_tool(message)
    if not result.get("found"):
        answer = result["message"]
    else:
        lines = ["I found these document references:"]
        for item in result["results"][:3]:
            page = f", page {item['page_number']}" if item.get("page_number") else ""
            lines.append(f"- {item['file_name']}{page}: {item['text'][:220].strip()}...")
        answer = "\n".join(lines)
    return AgentResult(answer, "DocumentAgent", ["DocumentSearchTool"], [{"query": message, "limit": 5}], [result])


def _deterministic_report() -> AgentResult:
    report = write_demo_report("AskMamma Operations Report")
    recs = demo_reorder_recommendations()
    answer = f"{report['summary']}. It includes {len(recs)} sample demo replenishment recommendations."
    return AgentResult(
        answer,
        "ReportAgent",
        ["DemoReportWriterTool", "DemoRecommendationTool"],
        [{"title": "AskMamma Operations Report"}, {"identifier": None}],
        [report, recs],
    )


def _result_to_state(state: GraphState, result: AgentResult, node_name: str) -> GraphState:
    route_path = _append_route(state, node_name)
    steps = state.get("intermediate_steps", [])
    for step in result.intermediate_steps:
        steps = [*steps, redact_payload(step)]
    if not result.intermediate_steps:
        steps = [*steps, {"agent": result.selected_agent, "mode": "deterministic_fallback"}]
    return {
        "selected_agent": result.selected_agent,
        "answer": result.answer,
        "tools_called": result.tools_called,
        "tool_inputs": redact_payload(result.tool_inputs),
        "tool_outputs": redact_payload(result.tool_outputs),
        "intermediate_steps": steps,
        "route_path": route_path,
        "trace_backend": tracing_backend_name(),
        "provider": result.provider,
        "model": result.model,
        "llm_used": result.llm_used,
        "fallback_used": result.fallback_used,
    }


def supervisor_node(state: GraphState) -> GraphState:
    route = _route_message(state["user_input"])
    answer = None
    if route == "greeting":
        answer = "Hello. I can help with AskMamma demo items, availability checks, partners, forecasts, reports, and document search."
    return {
        "route": route,
        "answer": answer,
        "selected_agent": "SupervisorAgent" if route == "greeting" else "",
        "intermediate_steps": _append_step(state, {"agent": "SupervisorAgent", "route": route}),
        "route_path": _append_route(state, "SupervisorAgent"),
        "trace_backend": tracing_backend_name(),
        "provider": "Deterministic fallback",
        "model": "None",
        "llm_used": False,
        "fallback_used": True,
    }


def action_node(state: GraphState) -> GraphState:
    prompt = (
        SYSTEM_PROMPT
        + "\nYou are the AskMamma action specialist. Use demo item lookup, availability, and partner tools as needed."
    )
    result = _run_langchain_tool_agent(
        "AskMammaActionAgent",
        prompt,
        {"DemoItemLookupTool", "DemoAvailabilityTool", "DemoPartnerLookupTool", "DemoRecommendationTool"},
        state["user_input"],
        state["session_id"],
    ) or _deterministic_action(state["user_input"], state["session_id"])
    return _result_to_state(state, result, "AskMammaActionAgent")


def forecast_node(state: GraphState) -> GraphState:
    prompt = (
        SYSTEM_PROMPT
        + "\nYou are the AskMamma forecasting specialist. Use demo history, forecast, and recommendation tools."
    )
    result = _run_langchain_tool_agent(
        "ForecastAgent",
        prompt,
        {"DemoHistoryTool", "DemoForecastTool", "DemoRecommendationTool"},
        state["user_input"],
        state["session_id"],
    ) or _deterministic_forecast(state["user_input"], state["session_id"])
    return _result_to_state(state, result, "ForecastAgent")


def document_node(state: GraphState) -> GraphState:
    prompt = (
        SYSTEM_PROMPT
        + "\nYou are the AskMamma document specialist. Use only the document search tool and answer from retrieved evidence."
    )
    result = _run_langchain_tool_agent(
        "DocumentAgent",
        prompt,
        {"DocumentSearchTool"},
        state["user_input"],
        state["session_id"],
    ) or _deterministic_document(state["user_input"])
    return _result_to_state(state, result, "DocumentAgent")


def report_node(state: GraphState) -> GraphState:
    prompt = (
        SYSTEM_PROMPT
        + "\nYou are the AskMamma reporting specialist. Use report and recommendation tools to generate concise demo reports."
    )
    result = _run_langchain_tool_agent(
        "ReportAgent",
        prompt,
        {"DemoReportWriterTool", "DemoRecommendationTool", "DemoForecastTool"},
        state["user_input"],
        state["session_id"],
    ) or _deterministic_report()
    return _result_to_state(state, result, "ReportAgent")


def quality_node(state: GraphState) -> GraphState:
    answer = state.get("answer", "")
    notes = list(state.get("review_notes", []))
    if state.get("selected_agent") in {"ForecastAgent", "ReportAgent"} and not state.get("tools_called"):
        notes.append("Quality note: no tool evidence was available for this answer.")
    if "forecast" in answer.lower() and "units" not in answer.lower():
        notes.append("Quality note: forecast answers should include quantities when data is available.")
    if state.get("selected_agent") != "DocumentAgent" and "sample demo" not in answer.lower():
        notes.append("Quality note: answers about demo data should say they are based on sample/demo data.")

    if notes:
        answer = answer + "\n\n" + "\n".join(notes)

    audit_message = json.dumps(
        {
            "selected_agent": state.get("selected_agent"),
            "route_path": state.get("route_path", []),
            "tools_called": state.get("tools_called", []),
        },
        default=str,
    )
    audit_result = audit_log(state["session_id"], audit_message)

    return {
        "answer": answer,
        "review_notes": notes,
        "route_path": _append_route(state, "QualityReviewAgent"),
        "intermediate_steps": _append_step(state, {"agent": "QualityReviewAgent", "notes": notes}),
        "tool_outputs": [*state.get("tool_outputs", []), audit_result],
    }


def _route_after_supervisor(state: GraphState) -> str:
    return state.get("route", "actions")


def _build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("action", action_node)
    graph.add_node("forecast", forecast_node)
    graph.add_node("document", document_node)
    graph.add_node("report", report_node)
    graph.add_node("quality", quality_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "greeting": END,
            "actions": "action",
            "forecast": "forecast",
            "document": "document",
            "report": "report",
        },
    )
    graph.add_edge("action", "quality")
    graph.add_edge("forecast", "quality")
    graph.add_edge("document", "quality")
    graph.add_edge("report", "quality")
    graph.add_edge("quality", END)
    return graph.compile()


GRAPH = _build_graph()


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
                redact_payload(user_input),
                result.selected_agent,
                json.dumps(result.tools_called),
                json.dumps(redact_payload(result.tool_inputs), default=str),
                summarize_tools_for_trace(
                    [
                        *result.tool_outputs,
                        {
                            "intermediate_steps": result.intermediate_steps,
                            "route_path": result.route_path,
                            "provider": result.provider,
                            "model": result.model,
                            "llm_used": result.llm_used,
                            "fallback_used": result.fallback_used,
                            "response_time_ms": latency_ms,
                        },
                    ]
                ),
                redact_payload(result.answer),
                latency_ms,
                error,
                json.dumps(
                    {
                        "trace_backend": result.trace_backend,
                        "provider": result.provider,
                        "model": result.model,
                        "llm_used": result.llm_used,
                        "fallback_used": result.fallback_used,
                        "response_time_ms": latency_ms,
                        "selected_agent": result.selected_agent,
                        "tools_called": result.tools_called,
                    }
                ),
                utc_now(),
            ),
        )
    LOGGER.info(
        "agent_response provider=%s model=%s llm_used=%s fallback_used=%s selected_agent=%s tools_called=%s response_time_ms=%s",
        result.provider,
        result.model,
        result.llm_used,
        result.fallback_used,
        result.selected_agent,
        ",".join(result.tools_called),
        latency_ms,
    )


def _state_to_result(state: GraphState) -> AgentResult:
    return AgentResult(
        answer=state.get("answer", ""),
        selected_agent=state.get("selected_agent", "SupervisorAgent"),
        tools_called=state.get("tools_called", []),
        tool_inputs=state.get("tool_inputs", []),
        tool_outputs=state.get("tool_outputs", []),
        intermediate_steps=state.get("intermediate_steps", []),
        route_path=state.get("route_path", []),
        review_notes=state.get("review_notes", []),
        trace_backend=state.get("trace_backend", tracing_backend_name()),
        provider=state.get("provider", "Deterministic fallback"),
        model=state.get("model", "None"),
        llm_used=state.get("llm_used", False),
        fallback_used=state.get("fallback_used", True),
        response_time_ms=state.get("response_time_ms", 0),
    )


def invoke_agent(user_input: str, session_id: str | None = None) -> dict[str, Any]:
    initialize_database()
    configure_langsmith()
    session_id = session_id or str(uuid.uuid4())
    start = time.perf_counter()
    save_message(session_id, "user", user_input)
    try:
        runtime = current_runtime_status()
        final_state = GRAPH.invoke(
            {
                "user_input": user_input,
                "session_id": session_id,
                "tools_called": [],
                "tool_inputs": [],
                "tool_outputs": [],
                "intermediate_steps": [],
                "route_path": [],
                "review_notes": [],
                "errors": [],
                "trace_backend": tracing_backend_name(),
                "provider": runtime["provider"] if runtime["llm_used"] else "Deterministic fallback",
                "model": runtime["model"] if runtime["llm_used"] else "None",
                "llm_used": runtime["llm_used"],
                "fallback_used": runtime["fallback_used"],
            }
        )
        result = _state_to_result(final_state)
        error = None
    except Exception as exc:
        result = AgentResult(
            f"Sorry, I could not complete that request: {safe_error_message(exc)}",
            "SupervisorAgent",
            trace_backend=tracing_backend_name(),
            provider="Deterministic fallback",
            model="None",
            llm_used=False,
            fallback_used=True,
        )
        error = safe_error_message(exc)
    latency_ms = int((time.perf_counter() - start) * 1000)
    result.response_time_ms = latency_ms
    save_message(session_id, "assistant", result.answer)
    trace_run(session_id, user_input, result, latency_ms, error)
    return {
        "session_id": session_id,
        "answer": result.answer,
        "provider": result.provider,
        "model": result.model,
        "llm_used": result.llm_used,
        "fallback_used": result.fallback_used,
        "selected_agent": result.selected_agent,
        "tools_called": result.tools_called,
        "tool_inputs": result.tool_inputs,
        "intermediate_steps": result.intermediate_steps,
        "route_path": result.route_path,
        "review_notes": result.review_notes,
        "trace_backend": result.trace_backend,
        "latency_ms": latency_ms,
        "response_time_ms": latency_ms,
    }
