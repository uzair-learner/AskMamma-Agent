"""FastAPI backend for the inventory management agent system."""

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
from inventory.tools import (
    add_inventory_movement,
    demand_forecast,
    reorder_recommendations,
    tool_registry,
    write_inventory_report,
)


app = FastAPI(title="AskMamma Inventory Agent API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProductPayload(BaseModel):
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


class RestockPayload(BaseModel):
    product_id: int
    quantity: int = Field(gt=0)
    reason: str = "restock"


class ForecastPayload(BaseModel):
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


@app.get("/products")
def products(search: str | None = None, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    return list_products(search=search, limit=limit, offset=offset)


@app.get("/products/{product_id}")
def product(product_id: int) -> dict[str, Any]:
    item = get_product(product_id)
    if not item:
        raise HTTPException(status_code=404, detail="Product not found")
    return item


@app.post("/products")
def product_create(payload: ProductPayload) -> dict[str, Any]:
    return create_product(payload.model_dump())


@app.put("/products/{product_id}")
def product_update(product_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    item = update_product(product_id, payload)
    if not item:
        raise HTTPException(status_code=404, detail="Product not found")
    return item


@app.delete("/products/{product_id}")
def product_delete(product_id: int, confirm: bool = False) -> dict[str, Any]:
    if not confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to delete a product.")
    return {"deleted": delete_product(product_id)}


@app.get("/inventory/low-stock")
def inventory_low_stock() -> list[dict[str, Any]]:
    return low_stock_products()


@app.get("/inventory/out-of-stock")
def inventory_out_of_stock() -> list[dict[str, Any]]:
    return out_of_stock_products()


@app.post("/inventory/restock")
def inventory_restock(payload: RestockPayload) -> dict[str, Any]:
    return add_inventory_movement(payload.product_id, "stock_in", payload.quantity, payload.reason)


@app.post("/forecast/demand")
def forecast_demand(payload: ForecastPayload) -> dict[str, Any]:
    return demand_forecast(payload.identifier, payload.months)


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


@app.get("/reports/inventory")
def reports_inventory() -> dict[str, Any]:
    return write_inventory_report()


@app.get("/reports/forecast")
def reports_forecast() -> dict[str, Any]:
    return demand_forecast(months=6)


@app.get("/.well-known/agent-card.json")
def agent_card() -> dict[str, Any]:
    return {
        "name": "AskMamma Inventory Assistant",
        "description": "AI-powered inventory management agent with tools, memory, RAG, forecasting, reports, and traces.",
        "version": "1.0.0",
        "endpoint": "/agent/chat",
        "supported_input_modes": ["text", "task"],
        "supported_output_modes": ["text", "json", "markdown"],
        "authentication": "none for local development",
        "skills": [
            "inventory_lookup",
            "low_stock_analysis",
            "demand_forecast",
            "document_search",
            "report_generation",
        ],
    }
