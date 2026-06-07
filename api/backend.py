"""FastAPI backend for the AskMamma agent system."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core import config
from agents.orchestrator import get_recent_traces, get_session_messages, invoke_agent
from db.database import (
    create_product,
    dashboard_stats,
    delete_product,
    get_product,
    initialize_database,
    list_products,
    low_stock_products,
    out_of_stock_products,
    update_product,
)
from rag.retrieval import document_search, reindex_documents, save_uploaded_document
from askmamma.tools import (
    add_demo_movement,
    demo_forecast,
    demo_reorder_recommendations,
    tool_registry,
    write_demo_report,
)


app = FastAPI(
    title="AskMamma Agent API",
    description="Local AI agent demo with FastAPI, Streamlit, RAG, tool calling, memory, tracing, tests, SQLite runtime state, and clearly labeled sample demo data.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class DemoAvailabilityPayload(BaseModel):
    product_id: int
    quantity: int = Field(gt=0)
    reason: str = "restock"


class DemoForecastPayload(BaseModel):
    identifier: str | None = None
    months: int = Field(default=6, ge=1, le=24)


class ChatPayload(BaseModel):
    message: str
    session_id: str | None = None


class TaskPayload(BaseModel):
    task_id: str
    message: str
    from_agent: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: str = "submitted"


class DocumentSearchPayload(BaseModel):
    query: str
    limit: int = 5


@app.on_event("startup")
def startup() -> None:
    initialize_database()


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
    return create_product(payload.model_dump())


@app.put("/demo/items/{item_id}", summary="Update a sample demo item")
def demo_item_update(item_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    item = update_product(item_id, payload)
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


@app.post("/demo/availability/restock", summary="Record sample demo availability replenishment")
def demo_availability_restock(payload: DemoAvailabilityPayload) -> dict[str, Any]:
    return add_demo_movement(payload.product_id, "stock_in", payload.quantity, payload.reason)


@app.post("/demo/forecast", summary="Run a sample demo forecast")
def demo_forecast_run(payload: DemoForecastPayload) -> dict[str, Any]:
    return demo_forecast(payload.identifier, payload.months)


@app.post("/agent/chat")
def agent_chat(payload: ChatPayload) -> dict[str, Any]:
    return invoke_agent(payload.message, payload.session_id)


@app.post("/agent/run-task")
def agent_run_task(payload: ChatPayload) -> dict[str, Any]:
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


@app.post("/agent/tasks")
def agent_tasks(payload: TaskPayload) -> dict[str, Any]:
    result = invoke_agent(payload.message, payload.task_id)
    return {
        "task_id": payload.task_id,
        "from_agent": payload.from_agent,
        "metadata": payload.metadata,
        "status": "completed",
        "response": result,
    }


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


@app.get("/.well-known/agent-card.json")
def agent_card() -> dict[str, Any]:
    return {
        "name": "AskMamma Assistant",
        "description": "AI-powered AskMamma agent with tools, memory, RAG, demo/sample data, reports, and traces.",
        "version": "1.0.0",
        "endpoint": "/agent/chat",
        "supported_input_modes": ["text", "task"],
        "supported_output_modes": ["text", "json", "markdown"],
        "authentication": "none for local development",
        "skills": [
            "demo_item_lookup",
            "demo_availability_analysis",
            "demo_forecast",
            "document_search",
            "demo_report_generation",
        ],
    }
