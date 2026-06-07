"""FastAPI backend for the AskMamma agent system."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from agents.orchestrator import get_recent_traces, get_session_messages, invoke_agent
from askmamma.tools import (
    add_demo_movement,
    demo_forecast,
    demo_reorder_recommendations,
    invoke_named_tool,
    tool_registry,
    write_demo_report,
)
from core import config
from core.observability import configure_langsmith, safe_error_message
from db.database import (
    create_product,
    dashboard_stats,
    delete_product,
    get_product,
    initialize_database,
    list_suppliers,
    list_products,
    low_stock_products,
    out_of_stock_products,
    update_product,
)
from rag.retrieval import document_search, reindex_documents, save_uploaded_document


app = FastAPI(
    title="AskMamma Agent API",
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


class TaskRecord(BaseModel):
    task_id: str
    status: str
    assigned_agent: str | None = None
    input_payload: dict[str, Any]
    output_payload: dict[str, Any] | None = None
    error_payload: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    submitted_at: str
    started_at: str | None = None
    completed_at: str | None = None
    failed_at: str | None = None


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


def _rpc_success(request_id: str | int | None, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _rpc_error(request_id: str | int | None, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "environment": config.APP_ENV}


@app.get("/dashboard")
def dashboard() -> dict[str, Any]:
    return dashboard_stats()


@app.get("/demo/items", summary="List sample demo items")
def demo_items(search: str | None = None, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    return list_products(search=search, limit=limit, offset=offset)


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
    return invoke_agent(payload.message, payload.session_id)


@app.post("/agent/run-task")
def agent_run_task(payload: ChatPayload, request: Request) -> dict[str, Any]:
    _enforce_rate_limit(_request_key(request, "agent-run-task"))
    return invoke_agent(payload.message, payload.session_id)


@app.get("/agent/sessions/{session_id}")
def agent_session(session_id: str) -> dict[str, Any]:
    return {"session_id": session_id, "messages": get_session_messages(session_id, limit=100)}


@app.get("/agent/traces")
def agent_traces(limit: int = 50) -> list[dict[str, Any]]:
    return get_recent_traces(limit)


@app.get("/agent/tools")
def agent_tools() -> list[dict[str, Any]]:
    return [tool.model_dump() for tool in tool_registry()]


@app.get("/mcp/tools")
def mcp_tools() -> list[dict[str, Any]]:
    return agent_tools()


@app.get("/mcp/metadata")
def mcp_metadata() -> dict[str, Any]:
    return {
        "server_name": "AskMamma MCP Adapter",
        "transport": "http+jsonrpc",
        "methods": ["tools/list", "tools/call"],
        "tool_count": len(tool_registry()),
    }


@app.post("/mcp/rpc")
def mcp_rpc(payload: MCPRpcPayload, request: Request) -> dict[str, Any]:
    _enforce_rate_limit(_request_key(request, "mcp-rpc"))
    try:
        if payload.method == "tools/list":
            return _rpc_success(payload.id, agent_tools())
        if payload.method == "tools/call":
            tool_name = payload.params.get("name")
            arguments = payload.params.get("arguments", {})
            if not tool_name:
                return _rpc_error(payload.id, -32602, "Missing `name` for tools/call.")
            return _rpc_success(payload.id, invoke_named_tool(tool_name, arguments))
        return _rpc_error(payload.id, -32601, f"Unknown method `{payload.method}`.")
    except Exception as exc:
        return _rpc_error(payload.id, -32000, safe_error_message(exc))


@app.post("/agent/tasks")
def agent_tasks(payload: TaskPayload, request: Request) -> dict[str, Any]:
    _enforce_rate_limit(_request_key(request, "agent-tasks"))
    task_record = TaskRecord(
        task_id=payload.task_id,
        status="submitted",
        assigned_agent=payload.from_agent,
        input_payload={"message": payload.message},
        metadata=payload.metadata,
        submitted_at=_utcnow(),
    )
    TASK_STORE[payload.task_id] = task_record.model_dump()
    TASK_STORE[payload.task_id]["status"] = "running"
    TASK_STORE[payload.task_id]["started_at"] = _utcnow()

    try:
        result = invoke_agent(payload.message, payload.task_id)
        TASK_STORE[payload.task_id]["status"] = "completed"
        TASK_STORE[payload.task_id]["assigned_agent"] = result.get("selected_agent") or payload.from_agent
        TASK_STORE[payload.task_id]["output_payload"] = result
        TASK_STORE[payload.task_id]["completed_at"] = _utcnow()
    except Exception as exc:
        TASK_STORE[payload.task_id]["status"] = "failed"
        TASK_STORE[payload.task_id]["error_payload"] = {"message": safe_error_message(exc)}
        TASK_STORE[payload.task_id]["failed_at"] = _utcnow()
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
    return save_uploaded_document(file.filename or "upload.txt", content)


@app.post("/documents/reindex")
def documents_reindex() -> dict[str, Any]:
    return reindex_documents()


@app.post("/documents/search")
def documents_search(payload: DocumentSearchPayload) -> dict[str, Any]:
    return document_search(payload.query, payload.limit)


@app.get("/reports/demo", summary="Generate a sample demo report")
def reports_demo() -> dict[str, Any]:
    return write_demo_report()


@app.get("/reports/askmamma")
def reports_askmamma() -> dict[str, Any]:
    return write_demo_report("AskMamma Operations Report")


@app.get("/reports/demo-forecast", summary="Generate a sample demo forecast snapshot")
def reports_demo_forecast() -> dict[str, Any]:
    return demo_forecast(months=6)


@app.get("/reports")
def reports_list() -> list[dict[str, Any]]:
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in sorted(config.REPORT_DIR.glob("*.md"), key=lambda item: item.stat().st_mtime, reverse=True):
        stats = path.stat()
        reports.append(
            {
                "file_name": path.name,
                "path": str(path),
                "updated_at": datetime.fromtimestamp(stats.st_mtime, timezone.utc).isoformat(),
                "size_bytes": stats.st_size,
            }
        )
    return reports


@app.get("/.well-known/agent-card.json")
def agent_card() -> dict[str, Any]:
    return {
        "name": "AskMamma Assistant",
        "description": "AI-powered AskMamma agent with LangGraph routing, LangChain tools, embedding-backed RAG, memory, demo/sample data, reports, and traces.",
        "version": "2.0.0",
        "endpoint_url": "/agent/chat",
        "endpoint": "/agent/chat",
        "capabilities": {
            "tool_calling": True,
            "rag": True,
            "memory": True,
            "multi_agent": True,
            "task_execution": True,
            "mcp_adapter": True,
            "langsmith_tracing_optional": bool(config.LANGSMITH_API_KEY),
        },
        "authentication": {"type": "none", "notes": "local development only"},
        "skills": [
            "demo_item_lookup",
            "demo_availability_analysis",
            "demo_forecast",
            "document_search",
            "demo_report_generation",
            "quality_review",
        ],
        "supported_input_modes": ["text", "task", "jsonrpc"],
        "supported_output_modes": ["text", "json", "markdown"],
        "examples": [
            {"input": "Which sample demo items are low in availability right now?", "route": "AskMammaActionAgent"},
            {"input": "What does the return policy say about unopened items?", "route": "DocumentAgent"},
        ],
    }


def _frontend_path(path: str) -> Path:
    return FRONTEND_DIST_DIR / path


@app.get("/", include_in_schema=False, response_model=None)
def frontend_index() -> Response:
    index_path = _frontend_path("index.html")
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse(
        "<h1>AskMamma frontend not built yet.</h1><p>Run the launcher so the React app is built before startup.</p>",
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
        "<h1>AskMamma frontend not built yet.</h1><p>Run the launcher so the React app is built before startup.</p>",
        status_code=503,
    )
