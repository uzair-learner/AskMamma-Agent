"""FastAPI backend for the AskMamma agent system."""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from askmamma.tools import (
    add_demo_movement,
    demo_forecast,
    demo_reorder_recommendations,
    demo_history,
    tool_registry,
)
from core import config
from core.llm_provider import LLM_UNAVAILABLE_MESSAGE, current_runtime_status, get_llm_provider
from core.observability import configure_langsmith, safe_error_message
from db.database import (
    create_product,
    dashboard_stats,
    delete_product,
    find_product,
    get_product,
    initialize_database,
    list_ai_generation_events,
    list_suppliers,
    list_products,
    log_ai_generation_event,
    low_stock_products,
    out_of_stock_products,
    update_product,
)
from memory.service import audit_memory, conversation_memory, semantic_memory
from protocols.a2a.service import create_task_record, utcnow
from protocols.agent_cards.builder import build_agent_card
from protocols.mcp.server import handle_rpc, list_prompts, list_resources, list_tools

app = FastAPI(
    title="Inventory Pilot AI API",
    description=(
        "Local AI agent demo with FastAPI, Streamlit, LangGraph routing, LangChain tool calling, "
        "embedding-backed RAG, memory, tracing, tests, SQLite runtime state, and clearly labeled sample demo data."
    ),
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RATE_LIMIT_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
FRONTEND_DIST_DIR = config.ROOT_DIR / "frontend" / "dist"
TASK_STORE: dict[str, dict[str, Any]] = {}
AI_UNAVAILABLE_MESSAGE = "Ollama is unavailable. AI content was not generated."
REPORT_UNAVAILABLE_MESSAGE = "Ollama is unavailable. Report was not generated."


class DemoItemPayload(BaseModel):
    sku: str
    name: str
    category: str
    description: str = ""
    supplier_id: int | None = None
    price: float
    cost: float
    stock_quantity: int
    reorder_level: int
    reorder_quantity: int
    location: str | None = None
    expiry_date: str | None = None
    confirm: bool = False


class DemoAvailabilityPayload(BaseModel):
    product_id: int
    quantity: int = Field(gt=0)
    reason: str = "restock"
    confirm: bool = False


class DemoForecastPayload(BaseModel):
    identifier: str | None = None
    months: int = Field(default=6, ge=1, le=24)


class ChatPayload(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None


class TaskPayload(BaseModel):
    task_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1, max_length=4000)
    from_agent: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: str = "submitted"


class DocumentSearchPayload(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=10)


class MCPRpcPayload(BaseModel):
    jsonrpc: str = "2.0"
    id: str | int | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


@app.on_event("startup")
def startup() -> None:
    initialize_database()
    configure_langsmith()


def _enforce_rate_limit(key: str) -> None:
    window = RATE_LIMIT_BUCKETS[key]
    now = time.time()
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= config.RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please retry in a minute.")
    window.append(now)


def _request_key(request: Request, suffix: str) -> str:
    client = request.client.host if request.client else "local"
    return f"{client}:{suffix}"


def _orchestrator():
    from agents import orchestrator

    return orchestrator


def _rag():
    from rag import retrieval

    return retrieval


def _online_reports_snapshot(limit: int = 20) -> list[dict[str, Any]]:
    return list_ai_generation_events(limit=limit, feature_name="online_report")


def _ai_payload(
    *,
    provider: str,
    model: str,
    llm_used: bool,
    generated_at: str,
    status: str,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "ai_source": "Ollama",
        "provider": provider,
        "model": model,
        "llm_used": llm_used,
        "generated_at": generated_at,
        "status": status,
        "error_message": error_message,
    }


def _generate_ai_content(
    *,
    feature_name: str,
    prompt: str,
    content_key: str,
    unavailable_message: str,
    track_event: bool = True,
) -> dict[str, Any]:
    generated_at = utcnow()
    provider = "Ollama"
    model = config.OLLAMA_MODEL
    try:
        runtime = current_runtime_status()
        provider = runtime.get("provider") or provider
        model = runtime.get("model") or model
    except Exception as exc:
        runtime = {
            "provider": provider,
            "model": model,
            "llm_used": False,
            "runtime_error": safe_error_message(exc),
        }

    if not runtime.get("llm_used"):
        error_message = runtime.get("runtime_error") or unavailable_message
        if track_event:
            log_ai_generation_event(
                feature_name=feature_name,
                provider=provider,
                model=model,
                llm_used=False,
                prompt=prompt,
                response=None,
                created_at=generated_at,
                status="failed",
                error_message=error_message,
            )
        return {
            content_key: unavailable_message,
            **_ai_payload(
                provider=provider,
                model=model,
                llm_used=False,
                generated_at=generated_at,
                status="failed",
                error_message=error_message,
            ),
        }

    try:
        response = get_llm_provider().generate(prompt)
        if track_event:
            log_ai_generation_event(
                feature_name=feature_name,
                provider=provider,
                model=model,
                llm_used=True,
                prompt=prompt,
                response=response,
                created_at=generated_at,
                status="success",
            )
        return {
            content_key: response,
            **_ai_payload(
                provider=provider,
                model=model,
                llm_used=True,
                generated_at=generated_at,
                status="success",
            ),
        }
    except Exception as exc:
        error_message = safe_error_message(exc)
        if track_event:
            log_ai_generation_event(
                feature_name=feature_name,
                provider=provider,
                model=model,
                llm_used=False,
                prompt=prompt,
                response=None,
                created_at=generated_at,
                status="failed",
                error_message=error_message,
            )
        return {
            content_key: unavailable_message,
            **_ai_payload(
                provider=provider,
                model=model,
                llm_used=False,
                generated_at=generated_at,
                status="failed",
                error_message=error_message,
            ),
        }


def _ai_message(prompt: str) -> dict[str, Any]:
    return _generate_ai_content(
        feature_name="insight",
        prompt=prompt,
        content_key="message",
        unavailable_message=AI_UNAVAILABLE_MESSAGE,
        track_event=False,
    )


def _ai_explanation_message(
    prompt: str,
    *,
    feature_name: str,
    unavailable_message: str = AI_UNAVAILABLE_MESSAGE,
) -> dict[str, Any]:
    return _generate_ai_content(
        feature_name=feature_name,
        prompt=prompt,
        content_key="ai_explanation",
        unavailable_message=unavailable_message,
        track_event=True,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "environment": config.APP_ENV}


@app.get("/dashboard")
def dashboard() -> dict[str, Any]:
    return dashboard_stats()


@app.get("/ai/insights/dashboard")
def ai_dashboard_insight() -> dict[str, Any]:
    data = dashboard_stats()
    prompt = (
        "You are explaining an operations dashboard. Use only the provided JSON. "
        "Do not invent any numbers. Give a concise plain-English summary of stock pressure, "
        "high-demand signals, and the biggest operational risk in 3 short bullets.\n\n"
        f"{json.dumps(data, default=str)}"
    )
    return _ai_message(prompt)


@app.get("/demo/items", summary="List sample demo items")
def demo_items(search: str | None = None, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    return list_products(search=search, limit=limit, offset=offset)


@app.get("/ai/insights/products")
def ai_products_insight(filter: str = "all", product_id: int | None = None) -> dict[str, Any]:
    products = list_products(limit=100)
    high_demand_names = {item["name"] for item in dashboard_stats().get("predicted_high_demand_products", [])}
    if filter == "low-stock":
        products = [item for item in products if item["stock_quantity"] > 0 and item["stock_quantity"] <= item["reorder_level"]]
    elif filter == "out-of-stock":
        products = [item for item in products if item["stock_quantity"] <= 0]
    elif filter == "high-demand":
        products = [item for item in products if item["name"] in high_demand_names]
    selected = get_product(product_id) if product_id else (products[0] if products else None)
    payload = {
        "filter": filter,
        "selected_product": selected,
        "visible_products_sample": products[:8],
        "high_demand_names": sorted(high_demand_names),
    }
    prompt = (
        "You are explaining an operations product view. Use only the provided JSON. "
        "If the filter is high-demand, explain why the selected item is in that group. "
        "If the item is low on stock or unusual, explain the risk briefly. Keep it to 3 short bullets.\n\n"
        f"{json.dumps(payload, default=str)}"
    )
    return _ai_message(prompt)


@app.get("/demo/items/{item_id}", summary="Get one sample demo item")
def demo_item(item_id: int) -> dict[str, Any]:
    item = get_product(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Demo item not found")
    return item


@app.post("/demo/items", summary="Create a sample demo item")
def demo_item_create(payload: DemoItemPayload) -> dict[str, Any]:
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to create or overwrite demo item data.")
    return create_product(payload.model_dump(exclude={"confirm"}))


@app.put("/demo/items/{item_id}", summary="Update a sample demo item")
def demo_item_update(item_id: int, payload: DemoItemPayload) -> dict[str, Any]:
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to update demo item data.")
    item = update_product(item_id, payload.model_dump(exclude={"confirm"}))
    if not item:
        raise HTTPException(status_code=404, detail="Demo item not found")
    return item


@app.delete("/demo/items/{item_id}", summary="Delete a sample demo item")
def demo_item_delete(item_id: int, confirm: bool = False) -> dict[str, Any]:
    if not confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to delete a demo item.")
    return {"deleted": delete_product(item_id)}


@app.get("/demo/availability/low", summary="List low-availability sample demo items")
def demo_availability_low() -> list[dict[str, Any]]:
    return low_stock_products()


@app.get("/demo/availability/out", summary="List unavailable sample demo items")
def demo_availability_out() -> list[dict[str, Any]]:
    return out_of_stock_products()


@app.get("/demo/suppliers", summary="List sample demo suppliers")
def demo_suppliers() -> list[dict[str, Any]]:
    return list_suppliers()


@app.get("/demo/recommendations/reorder", summary="List sample demo reorder recommendations")
def demo_recommendations_reorder() -> list[dict[str, Any]]:
    return demo_reorder_recommendations()


@app.get("/ai/insights/forecasts")
def ai_forecasts_insight() -> dict[str, Any]:
    recommendations = demo_reorder_recommendations()[:8]
    top_forecast = demo_forecast(recommendations[0]["sku"], months=6) if recommendations else demo_forecast(months=6)
    payload = {
        "recommendations": recommendations,
        "forecast_example": top_forecast,
    }
    prompt = (
        "You are explaining forecast results for an operations page. Use only the provided JSON. "
        "Explain in plain English why demand appears to be increasing or decreasing, what the reorder risk is, "
        "and what the team should watch next. Keep it concise.\n\n"
        f"{json.dumps(payload, default=str)}"
    )
    return _ai_message(prompt)


@app.get("/forecast/ai-explanation")
def forecast_ai_explanation() -> dict[str, Any]:
    recommendations = demo_reorder_recommendations()[:8]
    top_forecast = demo_forecast(recommendations[0]["sku"], months=6) if recommendations else demo_forecast(months=6)
    payload = {
        "recommendations": recommendations,
        "forecast_example": top_forecast,
    }
    prompt = (
        "You are explaining forecast results for an operations page. Use only the provided JSON. "
        "Explain in plain English why demand is increasing or decreasing, which products are risky, "
        "which items may need attention soon, and give a concise forecast summary.\n\n"
        f"{json.dumps(payload, default=str)}"
    )
    return _ai_explanation_message(prompt, feature_name="forecast_explanation")


@app.get("/ai/insights/reorder")
def ai_reorder_insight() -> dict[str, Any]:
    recommendations = demo_reorder_recommendations()[:10]
    prompt = (
        "You are explaining reorder recommendations. Use only the provided JSON. "
        "Prioritize the biggest reorder risks and explain why these items should be reordered. "
        "Keep it short and operational.\n\n"
        f"{json.dumps(recommendations, default=str)}"
    )
    return _ai_message(prompt)


@app.get("/reorder/ai-explanation")
def reorder_ai_explanation() -> dict[str, Any]:
    recommendations = demo_reorder_recommendations()[:10]
    prompt = (
        "You are explaining reorder recommendations. Use only the provided JSON. "
        "Explain why each item should be reordered, identify the highest priority reorder item, "
        "describe the risk if reorder is delayed, and give a concise purchase recommendation.\n\n"
        f"{json.dumps(recommendations, default=str)}"
    )
    return _ai_explanation_message(prompt, feature_name="reorder_explanation")


@app.get("/ai/insights/suppliers")
def ai_suppliers_insight() -> dict[str, Any]:
    suppliers = list_suppliers()[:10]
    prompt = (
        "You are explaining supplier performance and risk. Use only the provided JSON. "
        "Summarize the biggest supplier risks, especially low-stock exposure and out-of-stock exposure. "
        "Keep it to 3 short bullets.\n\n"
        f"{json.dumps(suppliers, default=str)}"
    )
    return _ai_message(prompt)


@app.post("/demo/availability/restock", summary="Record sample demo availability replenishment")
def demo_availability_restock(payload: DemoAvailabilityPayload) -> dict[str, Any]:
    try:
        return add_demo_movement(payload.product_id, "stock_in", payload.quantity, payload.reason, payload.confirm)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=safe_error_message(exc)) from exc


@app.post("/demo/forecast", summary="Run a sample demo forecast")
def demo_forecast_run(payload: DemoForecastPayload) -> dict[str, Any]:
    return demo_forecast(payload.identifier, payload.months)


@app.post("/agent/chat")
def agent_chat(payload: ChatPayload, request: Request) -> dict[str, Any]:
    _enforce_rate_limit(_request_key(request, "agent-chat"))
    result = _orchestrator().invoke_agent(payload.message, payload.session_id)
    generated_at = utcnow()
    if result.get("llm_used"):
        log_ai_generation_event(
            feature_name="chat",
            provider=result.get("provider") or "Ollama",
            model=result.get("model") or config.OLLAMA_MODEL,
            llm_used=True,
            prompt=payload.message,
            response=result.get("answer"),
            created_at=generated_at,
            status="success",
        )
        return {
            **result,
            **_ai_payload(
                provider=result.get("provider") or "Ollama",
                model=result.get("model") or config.OLLAMA_MODEL,
                llm_used=True,
                generated_at=generated_at,
                status="success",
            ),
        }

    error_message = result.get("answer") or AI_UNAVAILABLE_MESSAGE
    log_ai_generation_event(
        feature_name="chat",
        provider=result.get("provider") or "Ollama",
        model=result.get("model") or config.OLLAMA_MODEL,
        llm_used=False,
        prompt=payload.message,
        response=None,
        created_at=generated_at,
        status="failed",
        error_message=error_message,
    )
    return {
        **result,
        "answer": AI_UNAVAILABLE_MESSAGE,
        **_ai_payload(
            provider=result.get("provider") or "Ollama",
            model=result.get("model") or config.OLLAMA_MODEL,
            llm_used=False,
            generated_at=generated_at,
            status="failed",
            error_message=error_message,
        ),
    }


@app.post("/agent/run-task")
def agent_run_task(payload: ChatPayload, request: Request) -> dict[str, Any]:
    _enforce_rate_limit(_request_key(request, "agent-run-task"))
    return _orchestrator().invoke_agent(payload.message, payload.session_id)


@app.get("/agent/sessions/{session_id}")
def agent_session(session_id: str) -> dict[str, Any]:
    return {"session_id": session_id, "messages": _orchestrator().get_session_messages(session_id, limit=100)}


@app.get("/agent/traces")
def agent_traces(limit: int = 50) -> list[dict[str, Any]]:
    return _orchestrator().get_recent_traces(limit)


@app.get("/agent/graph")
def agent_graph() -> dict[str, Any]:
    return {"format": "mermaid", "graph": _orchestrator().workflow_mermaid()}


@app.get("/admin/diagnostics")
def admin_diagnostics() -> dict[str, Any]:
    try:
        runtime = current_runtime_status()
    except Exception as exc:
        runtime = {
            "provider": "Unavailable",
            "model": "Unavailable",
            "ollama_base_url": config.OLLAMA_BASE_URL,
            "ollama_reachable": False,
            "error": safe_error_message(exc),
        }
    try:
        recent_requests = _orchestrator().get_recent_traces(20)
    except Exception:
        recent_requests = []
    ai_events = list_ai_generation_events(20)
    return {
        "provider": runtime["provider"],
        "model": runtime["model"],
        "ollama_base_url": runtime["ollama_base_url"],
        "ollama_reachable": runtime["ollama_reachable"],
        "runtime_error": runtime.get("runtime_error"),
        "recent_requests": recent_requests,
        "ai_events": ai_events,
    }


@app.get("/agent/tools")
def agent_tools() -> list[dict[str, Any]]:
    return list_tools()


@app.get("/mcp/tools")
def mcp_tools() -> list[dict[str, Any]]:
    return list_tools()


@app.get("/mcp/resources")
def mcp_resources() -> list[dict[str, Any]]:
    return list_resources()


@app.get("/mcp/prompts")
def mcp_prompts() -> list[dict[str, Any]]:
    return list_prompts()


@app.get("/mcp/metadata")
def mcp_metadata() -> dict[str, Any]:
    return {
        "server_name": "Inventory Pilot AI MCP Server",
        "transport": "http+jsonrpc",
        "methods": ["tools/list", "tools/call", "resources/list", "prompts/list", "agents/list"],
        "tool_count": len(tool_registry()),
        "resource_count": len(list_resources()),
        "prompt_count": len(list_prompts()),
    }


@app.post("/mcp/rpc")
def mcp_rpc(payload: MCPRpcPayload, request: Request) -> dict[str, Any]:
    _enforce_rate_limit(_request_key(request, "mcp-rpc"))
    try:
        return handle_rpc(payload.method, payload.id, payload.params)
    except Exception as exc:
        return {"jsonrpc": "2.0", "id": payload.id, "error": {"code": -32000, "message": safe_error_message(exc)}}


@app.post("/agent/tasks")
def agent_tasks(payload: TaskPayload, request: Request) -> dict[str, Any]:
    _enforce_rate_limit(_request_key(request, "agent-tasks"))
    TASK_STORE[payload.task_id] = create_task_record(payload.task_id, payload.message, payload.metadata, payload.from_agent)
    TASK_STORE[payload.task_id]["status"] = "running"
    TASK_STORE[payload.task_id]["started_at"] = utcnow()

    try:
        result = _orchestrator().invoke_agent(payload.message, payload.task_id)
        TASK_STORE[payload.task_id]["status"] = "completed"
        TASK_STORE[payload.task_id]["assigned_agent"] = result.get("selected_agent") or payload.from_agent
        TASK_STORE[payload.task_id]["output_payload"] = result
        TASK_STORE[payload.task_id]["completed_at"] = utcnow()
    except Exception as exc:
        TASK_STORE[payload.task_id]["status"] = "failed"
        TASK_STORE[payload.task_id]["error_payload"] = {"message": safe_error_message(exc)}
        TASK_STORE[payload.task_id]["failed_at"] = utcnow()
        raise

    return TASK_STORE[payload.task_id]


@app.get("/agent/tasks/{task_id}")
def agent_task_status(task_id: str) -> dict[str, Any]:
    task = TASK_STORE.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.post("/documents/upload")
async def documents_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read()
    try:
        return _rag().save_uploaded_document(file.filename or "upload.txt", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=safe_error_message(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=safe_error_message(exc)) from exc


@app.post("/documents/reindex")
def documents_reindex() -> dict[str, Any]:
    try:
        return _rag().reindex_documents()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=safe_error_message(exc)) from exc


@app.post("/documents/search")
def documents_search(payload: DocumentSearchPayload) -> dict[str, Any]:
    try:
        return _rag().document_search(payload.query, payload.limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=safe_error_message(exc)) from exc


@app.get("/reports/demo", summary="Generate a sample demo report")
def reports_demo() -> dict[str, Any]:
    return reports_askmamma()


@app.get("/reports/askmamma")
def reports_askmamma() -> dict[str, Any]:
    dashboard = dashboard_stats()
    recommendations = demo_reorder_recommendations()
    forecast = demo_forecast(months=6)
    suppliers = list_suppliers()
    products = list_products(limit=200)
    payload = {
        "dashboard": dashboard,
        "forecast": forecast,
        "reorder_recommendations": recommendations,
        "high_demand_products": dashboard.get("predicted_high_demand_products", []),
        "low_stock_products": [item for item in products if item["stock_quantity"] > 0 and item["stock_quantity"] <= item["reorder_level"]],
        "out_of_stock_products": [item for item in products if item["stock_quantity"] <= 0],
        "suppliers": suppliers,
    }
    prompt = (
        "You are writing an online inventory operations report. Use only the provided JSON. "
        "Write a concise report with sections for dashboard summary, forecast risks, reorder priorities, "
        "high-demand products, stock risks, supplier risks, and recommended next actions. "
        "Do not invent any data.\n\n"
        f"{json.dumps(payload, default=str)}"
    )
    report = _generate_ai_content(
        feature_name="online_report",
        prompt=prompt,
        content_key="report_content",
        unavailable_message=REPORT_UNAVAILABLE_MESSAGE,
        track_event=True,
    )
    return {
        **report,
        "title": "Inventory Pilot AI Online Report",
    }


@app.get("/ai/insights/reports")
def ai_reports_insight() -> dict[str, Any]:
    payload = {
        "dashboard": dashboard_stats(),
        "recommendations": demo_reorder_recommendations()[:8],
        "suppliers": list_suppliers()[:8],
        "reports": _online_reports_snapshot(5),
    }
    prompt = (
        "You are writing a concise executive-style report summary. Use only the provided JSON. "
        "Summarize the main inventory risks, supplier issues, and report-ready talking points in 3 short bullets.\n\n"
        f"{json.dumps(payload, default=str)}"
    )
    return _ai_message(prompt)


@app.get("/reports/demo-forecast", summary="Generate a sample demo forecast snapshot")
def reports_demo_forecast() -> dict[str, Any]:
    return demo_forecast(months=6)


@app.get("/reports")
def reports_list() -> list[dict[str, Any]]:
    return _online_reports_snapshot()


@app.get("/.well-known/agent-card.json")
def agent_card() -> dict[str, Any]:
    return build_agent_card()


@app.get("/memory/conversation/{session_id}")
def memory_conversation_view(session_id: str, limit: int = 20) -> dict[str, Any]:
    return {"session_id": session_id, "records": conversation_memory(session_id, limit=limit)}


@app.get("/memory/semantic")
def memory_semantic_view(limit: int = 50) -> dict[str, Any]:
    return {"records": semantic_memory(limit=limit)}


@app.get("/memory/audit")
def memory_audit_view(limit: int = 50) -> dict[str, Any]:
    return {"records": audit_memory(limit=limit)}


def _frontend_path(path: str) -> Path:
    return FRONTEND_DIST_DIR / path


@app.get("/", include_in_schema=False, response_model=None)
def frontend_index() -> Response:
    index_path = _frontend_path("index.html")
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse(
        "<h1>Inventory Pilot AI frontend not built yet.</h1><p>Run the launcher so the React app is built before startup.</p>",
        status_code=503,
    )


@app.get("/{full_path:path}", include_in_schema=False, response_model=None)
def frontend_assets(full_path: str) -> Response:
    requested = _frontend_path(full_path)
    if requested.is_file():
        return FileResponse(requested)

    index_path = _frontend_path("index.html")
    if index_path.exists():
        return FileResponse(index_path)

    return HTMLResponse(
        "<h1>Inventory Pilot AI frontend not built yet.</h1><p>Run the launcher so the React app is built before startup.</p>",
        status_code=503,
    )
