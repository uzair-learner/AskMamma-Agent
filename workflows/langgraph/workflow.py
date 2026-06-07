"""Production-style LangGraph workflow for the AskMamma learning project."""

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

from agents.catalog import build_agent_catalog
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
from workflows.routing.classifier import classify_route


LOGGER = logging.getLogger(__name__)
AGENT_CATALOG = build_agent_catalog()


class GraphState(TypedDict, total=False):
    user_input: str
    session_id: str
    route: str
    selected_agent: str
    answer: str
    specialist_output: dict[str, Any]
    report_bundle: dict[str, Any]
    tools_called: list[str]
    tool_inputs: list[dict[str, Any]]
    tool_outputs: list[Any]
    intermediate_steps: list[dict[str, Any]]
    route_path: list[str]
    review_notes: list[str]
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
    report_bundle: dict[str, Any] = field(default_factory=dict)
    trace_backend: str = "sqlite"
    provider: str = "Deterministic fallback"
    model: str = "None"
    llm_used: bool = False
    fallback_used: bool = True
    response_time_ms: int = 0


def workflow_mermaid() -> str:
    return "\n".join(
        [
            "flowchart TD",
            "    U[User] --> S[SupervisorAgent]",
            "    S -->|inventory| I[InventoryAgent]",
            "    S -->|forecast| F[ForecastAgent]",
            "    S -->|document| D[DocumentAgent]",
            "    S -->|research| R[ResearchAgent]",
            "    S -->|report| P[ReportingAgent]",
            "    I --> P",
            "    F --> P",
            "    D --> P",
            "    R --> P",
            "    P --> Q[QualityReviewAgent]",
            "    Q --> A[Final Response]",
        ]
    )


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
        trace["route_path"] = metadata.get("route_path", [])
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
    remembered = recall_session_value(session_id, "last_sku")
    if remembered:
        return remembered
    return None


def _append_step(state: GraphState, step: dict[str, Any]) -> list[dict[str, Any]]:
    return [*state.get("intermediate_steps", []), redact_payload(step)]


def _append_route(state: GraphState, node_name: str) -> list[str]:
    return [*state.get("route_path", []), node_name]


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
    agent = create_agent(
        model=llm,
        tools=_lc_tools(tool_names),
        system_prompt=system_prompt,
        name=agent_name,
    )
    result = agent.invoke({"messages": [*_history_as_messages(session_id), HumanMessage(content=message)]})
    messages = result.get("messages", [])
    intermediate_steps: list[dict[str, Any]] = []
    tools_called: list[str] = []
    tool_inputs: list[dict[str, Any]] = []
    tool_outputs: list[Any] = []
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
    return AgentResult(
        answer=final_answer or str(messages[-1].content),
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


def _inventory_fallback(message: str, session_id: str) -> AgentResult:
    lowered = message.lower()
    identifier = _extract_identifier(message, session_id)
    if "low" in lowered and ("stock" in lowered or "availability" in lowered):
        result = demo_status()
        low = result["low_stock"]
        answer = (
            "In the sample demo inventory, the low-availability items are:\n"
            + "\n".join(
                f"- {p['sku']} {p['name']}: {p['stock_quantity']} on hand, reorder level {p['reorder_level']}"
                for p in low
            )
        )
        return AgentResult(answer=answer, selected_agent="InventoryAgent", tools_called=["DemoAvailabilityTool"], tool_inputs=[{"identifier": None}], tool_outputs=[result])
    if ("supplier" in lowered or "partner" in lowered) and identifier:
        result = demo_partner_lookup(identifier)
        item = demo_status(identifier).get("item")
        if result.get("found") and item:
            remember_session_value(session_id, "last_sku", item["sku"])
            partner = result["partner"]
            answer = (
                f"In the sample demo inventory, {result['item']} is supplied by {partner['name']} "
                f"with a lead time of {partner['lead_time_days']} days."
            )
        else:
            answer = result["message"]
        return AgentResult(answer=answer, selected_agent="InventoryAgent", tools_called=["DemoPartnerLookupTool"], tool_inputs=[{"identifier": identifier}], tool_outputs=[result])
    if identifier:
        result = demo_status(identifier)
        if result.get("found") and (item := result.get("item")):
            remember_session_value(session_id, "last_sku", item["sku"])
            answer = (
                f"In the sample demo inventory, {item['name']} ({item['sku']}) has {item['stock_quantity']} units on hand "
                f"and a reorder level of {item['reorder_level']}."
            )
            return AgentResult(answer=answer, selected_agent="InventoryAgent", tools_called=["DemoAvailabilityTool"], tool_inputs=[{"identifier": identifier}], tool_outputs=[result])
    result = demo_item_lookup(message)
    answer = "Matching sample inventory items:\n" + "\n".join(
        f"- {item['sku']} {item['name']} ({item['stock_quantity']} on hand)" for item in result[:10]
    )
    return AgentResult(answer=answer, selected_agent="InventoryAgent", tools_called=["DemoItemLookupTool"], tool_inputs=[{"query": message}], tool_outputs=[result])


def _forecast_fallback(message: str, session_id: str) -> AgentResult:
    identifier = _extract_identifier(message, session_id)
    history = demo_history(identifier, months=6)
    forecast = demo_forecast(identifier, months=6)
    if history.get("item"):
        remember_session_value(session_id, "last_sku", history["item"]["sku"])
    answer = (
        f"Based on seeded history, expected next-month demand is {forecast['predicted_quantity']} units. "
        f"{forecast['explanation']}"
        if forecast.get("found")
        else forecast["message"]
    )
    return AgentResult(
        answer=answer,
        selected_agent="ForecastAgent",
        tools_called=["DemoHistoryTool", "DemoForecastTool"],
        tool_inputs=[{"identifier": identifier, "months": 6}, {"identifier": identifier, "months": 6}],
        tool_outputs=[history, forecast],
    )


def _document_fallback(message: str) -> AgentResult:
    result = document_search_tool(message)
    if not result.get("found"):
        answer = result["message"]
    else:
        answer = "I found these document references:\n" + "\n".join(
            f"- {item['file_name']}: {item['text'][:180].strip()}..." for item in result["results"][:3]
        )
    return AgentResult(answer=answer, selected_agent="DocumentAgent", tools_called=["DocumentSearchTool"], tool_inputs=[{"query": message, "limit": 5}], tool_outputs=[result])


def _research_fallback(message: str) -> AgentResult:
    topic = classify_route(message)
    answer = (
        "This project uses a supervisor-led multi-agent design. LangGraph manages routing, LangChain provides "
        "tool calling and prompt abstractions, RAG grounds answers in documents, MCP exposes tools/resources/prompts, "
        "and A2A-style endpoints show how agents can publish capabilities and accept tasks."
    )
    return AgentResult(
        answer=answer,
        selected_agent="ResearchAgent",
        intermediate_steps=[{"agent": "ResearchAgent", "topic": topic or "architecture"}],
    )


def _report_markdown(state: GraphState) -> dict[str, Any]:
    worker = state.get("specialist_output", {})
    route = state.get("route", "inventory")
    report = write_demo_report("AskMamma Learning Project Report", output_format="md") if route == "report" else None
    recommendations = demo_reorder_recommendations() if route in {"inventory", "forecast", "report"} else []
    forecast = demo_forecast(months=6) if route in {"forecast", "report"} else {}
    summary = worker.get("answer", state.get("answer", "")) or (report.get("summary") if report else "")
    tool_names = worker.get("tools_called", state.get("tools_called", [])) or (["DemoReportWriterTool"] if report else [])
    specialist_name = worker.get("selected_agent", state.get("selected_agent")) or ("ReportingAgent" if report else "SupervisorAgent")
    markdown = "\n".join(
        [
            "# AskMamma Agent Summary",
            f"- Route: {route}",
            f"- Specialist: {specialist_name}",
            f"- Tools: {', '.join(tool_names) or 'None'}",
            f"- Summary: {summary}",
            f"- Recommendations: {len(recommendations)}",
        ]
    )
    txt = markdown.replace("# ", "").replace("- ", "* ")
    bundle = {
        "markdown": markdown,
        "txt": txt,
        "json": {
            "route": route,
            "specialist": worker.get("selected_agent", state.get("selected_agent")),
            "tools_called": worker.get("tools_called", state.get("tools_called", [])),
            "recommendations_count": len(recommendations),
            "forecast_snapshot": forecast,
            "generated_report": report,
        },
    }
    if report:
        bundle["download"] = report
    return bundle


def supervisor_node(state: GraphState) -> GraphState:
    route = classify_route(state["user_input"])
    answer = None
    if route == "greeting":
        answer = "Hello from AskMamma. I can help with inventory, forecasting, documents, reports, MCP, A2A, and interview-style project walkthroughs."
    return {
        "route": route,
        "answer": answer,
        "selected_agent": "SupervisorAgent" if route == "greeting" else "",
        "intermediate_steps": _append_step(state, {"agent": "SupervisorAgent", "route": route}),
        "route_path": _append_route(state, "SupervisorAgent"),
    }


def inventory_node(state: GraphState) -> GraphState:
    definition = AGENT_CATALOG["InventoryAgent"]
    result = _run_langchain_tool_agent(
        "InventoryAgent",
        definition.system_prompt,
        {"DemoItemLookupTool", "DemoAvailabilityTool", "DemoPartnerLookupTool", "DemoRecommendationTool"},
        state["user_input"],
        state["session_id"],
    ) or _inventory_fallback(state["user_input"], state["session_id"])
    return _result_to_state(state, result, "InventoryAgent")


def forecast_node(state: GraphState) -> GraphState:
    definition = AGENT_CATALOG["ForecastAgent"]
    result = _run_langchain_tool_agent(
        "ForecastAgent",
        definition.system_prompt,
        {"DemoHistoryTool", "DemoForecastTool", "DemoRecommendationTool"},
        state["user_input"],
        state["session_id"],
    ) or _forecast_fallback(state["user_input"], state["session_id"])
    return _result_to_state(state, result, "ForecastAgent")


def document_node(state: GraphState) -> GraphState:
    definition = AGENT_CATALOG["DocumentAgent"]
    result = _run_langchain_tool_agent(
        "DocumentAgent",
        definition.system_prompt,
        {"DocumentSearchTool"},
        state["user_input"],
        state["session_id"],
    ) or _document_fallback(state["user_input"])
    return _result_to_state(state, result, "DocumentAgent")


def research_node(state: GraphState) -> GraphState:
    result = _research_fallback(state["user_input"])
    return _result_to_state(state, result, "ResearchAgent")


def report_node(state: GraphState) -> GraphState:
    bundle = _report_markdown(state)
    report_answer = bundle["markdown"]
    tools_called = list(state.get("tools_called", []))
    tool_outputs = list(state.get("tool_outputs", []))
    if state.get("route") == "report" and bundle.get("download"):
        tools_called.append("DemoReportWriterTool")
        tool_outputs.append(bundle["download"])
    return {
        "answer": report_answer,
        "report_bundle": bundle,
        "route_path": _append_route(state, "ReportingAgent"),
        "intermediate_steps": _append_step(
            state,
            {"agent": "ReportingAgent", "formats": list(bundle.keys()), "has_download": "download" in bundle},
        ),
        "selected_agent": state.get("selected_agent") or "ReportingAgent",
        "tools_called": tools_called,
        "tool_outputs": tool_outputs,
    }


def quality_node(state: GraphState) -> GraphState:
    answer = state.get("answer", "")
    notes = list(state.get("review_notes", []))
    if state.get("selected_agent") in {"InventoryAgent", "ForecastAgent"} and "sample demo" not in answer.lower():
        notes.append("Quality note: label seeded outputs as sample/demo data.")
    if state.get("selected_agent") == "DocumentAgent" and "document references" not in answer.lower():
        notes.append("Quality note: document answers should cite retrieved evidence.")
    if state.get("selected_agent") == "ResearchAgent" and "LangGraph" not in answer:
        notes.append("Quality note: architecture explanations should mention the graph workflow.")
    final_answer = answer if not notes else answer + "\n\n" + "\n".join(notes)
    audit_result = audit_log(
        state["session_id"],
        json.dumps(
            {
                "selected_agent": state.get("selected_agent"),
                "route_path": state.get("route_path", []),
                "tools_called": state.get("tools_called", []),
            }
        ),
    )
    return {
        "answer": final_answer,
        "review_notes": notes,
        "route_path": _append_route(state, "QualityReviewAgent"),
        "intermediate_steps": _append_step(state, {"agent": "QualityReviewAgent", "notes": notes}),
        "tool_outputs": [*state.get("tool_outputs", []), audit_result],
    }


def _result_to_state(state: GraphState, result: AgentResult, node_name: str) -> GraphState:
    route_path = _append_route(state, node_name)
    steps = state.get("intermediate_steps", [])
    for step in result.intermediate_steps or [{"agent": result.selected_agent, "mode": "deterministic_fallback"}]:
        steps = [*steps, redact_payload(step)]
    return {
        "selected_agent": result.selected_agent,
        "answer": result.answer,
        "specialist_output": {
            "selected_agent": result.selected_agent,
            "answer": result.answer,
            "tools_called": result.tools_called,
        },
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


def _route_after_supervisor(state: GraphState) -> str:
    return state.get("route", "inventory")


def _build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("inventory", inventory_node)
    graph.add_node("forecast", forecast_node)
    graph.add_node("document", document_node)
    graph.add_node("research", research_node)
    graph.add_node("report", report_node)
    graph.add_node("quality", quality_node)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "greeting": END,
            "inventory": "inventory",
            "forecast": "forecast",
            "document": "document",
            "research": "research",
            "report": "report",
        },
    )
    graph.add_edge("inventory", "report")
    graph.add_edge("forecast", "report")
    graph.add_edge("document", "report")
    graph.add_edge("research", "report")
    graph.add_edge("report", "quality")
    graph.add_edge("quality", END)
    return graph.compile()


GRAPH = _build_graph()


def trace_run(session_id: str, user_input: str, result: AgentResult, latency_ms: int, error: str | None = None) -> None:
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
                            "route_path": result.route_path,
                            "provider": result.provider,
                            "model": result.model,
                            "llm_used": result.llm_used,
                            "fallback_used": result.fallback_used,
                            "response_time_ms": latency_ms,
                            "report_bundle": result.report_bundle,
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
                        "route_path": result.route_path,
                        "tools_called": result.tools_called,
                    }
                ),
                utc_now(),
            ),
        )
    LOGGER.info(
        "agent_response selected_agent=%s tools=%s latency_ms=%s",
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
        report_bundle=state.get("report_bundle", {}),
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
    save_message(session_id, "user", user_input)
    start = time.perf_counter()
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
            answer=f"Sorry, I could not complete that request: {safe_error_message(exc)}",
            selected_agent="SupervisorAgent",
            trace_backend=tracing_backend_name(),
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
        "response_time_ms": latency_ms,
        "report_bundle": result.report_bundle,
        "graph_mermaid": workflow_mermaid(),
    }
