from fastapi.testclient import TestClient

from api.backend import app
from core import config
from core.llm_provider import LLM_UNAVAILABLE_MESSAGE
from scripts.seed_data import seed


def setup_module():
    seed()


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_demo_low_availability_endpoint():
    response = client.get("/demo/availability/low")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert response.json()


def test_agent_card_endpoint():
    response = client.get("/.well-known/agent-card.json")
    assert response.status_code == 200
    payload = response.json()
    assert payload["capabilities"]["multi_agent"] is True
    assert "InventoryAgent" in payload["skills"]


def test_tool_registry_endpoint():
    response = client.get("/agent/tools")
    assert response.status_code == 200
    names = {tool["name"] for tool in response.json()}
    assert {"DemoForecastTool", "DocumentSearchTool", "AuditLogTool"}.issubset(names)


def test_chat_endpoint_returns_graph_steps():
    response = client.post(
        "/agent/chat",
        json={"message": "In the sample demo catalog, which items are low in availability?", "session_id": "api-test"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "DemoAvailabilityTool" in payload["tools_called"]
    assert "SupervisorAgent" in payload["route_path"]
    assert payload["intermediate_steps"]
    assert payload["provider"]
    assert payload["model"] is not None
    assert isinstance(payload["llm_used"], bool)
    assert payload["selected_agent"]
    assert isinstance(payload["response_time_ms"], int)
    assert payload["response_time_ms"] >= 0


def test_mcp_rpc_list_and_call():
    list_response = client.post("/mcp/rpc", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    assert list_response.status_code == 200
    assert list_response.json()["result"]

    call_response = client.post(
        "/mcp/rpc",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "DemoAvailabilityTool", "arguments": {"identifier": "Copy Paper"}},
        },
    )
    assert call_response.status_code == 200
    assert call_response.json()["result"]["found"] is True


def test_agent_tasks_endpoint():
    response = client.post(
        "/agent/tasks",
        json={"task_id": "task-123", "message": "Generate a short AskMamma report.", "metadata": {"source": "test"}},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["assigned_agent"] == "ReportingAgent"
    assert response.json()["output_payload"]["selected_agent"] == "ReportingAgent"
    assert response.json()["submitted_at"]
    assert response.json()["started_at"]
    assert response.json()["completed_at"]


def test_agent_tasks_polling_endpoint():
    client.post(
        "/agent/tasks",
        json={"task_id": "task-lookup", "message": "Which sample demo items are low in availability?"},
    )
    response = client.get("/agent/tasks/task-lookup")
    assert response.status_code == 200
    assert response.json()["task_id"] == "task-lookup"
    assert response.json()["status"] == "completed"


def test_stock_movement_requires_confirmation():
    response = client.post("/demo/availability/restock", json={"product_id": 1, "quantity": 2, "reason": "demo"})
    assert response.status_code == 400


def test_mcp_metadata_endpoint():
    response = client.get("/mcp/metadata")
    assert response.status_code == 200
    assert response.json()["transport"] == "http+jsonrpc"
    assert response.json()["resource_count"] >= 1


def test_mcp_resources_and_prompts_endpoints():
    resources = client.get("/mcp/resources")
    prompts = client.get("/mcp/prompts")
    assert resources.status_code == 200
    assert prompts.status_code == 200
    assert resources.json()
    assert prompts.json()


def test_admin_diagnostics_endpoint():
    client.post(
        "/agent/chat",
        json={"message": "Which sample demo items are low in availability?", "session_id": "diagnostics-test"},
    )
    response = client.get("/admin/diagnostics")
    assert response.status_code == 200
    payload = response.json()
    assert "provider" in payload
    assert "model" in payload
    assert "ollama_base_url" in payload
    assert "ollama_reachable" in payload
    assert isinstance(payload["recent_requests"], list)
    if payload["recent_requests"]:
        recent = payload["recent_requests"][0]
        assert "provider" in recent
        assert "model" in recent
        assert "response_time_ms" in recent


def test_chat_endpoint_returns_clear_message_when_llm_unavailable(monkeypatch):
    from workflows.langgraph import workflow

    monkeypatch.setattr(
        workflow,
        "current_runtime_status",
        lambda: {
            "provider": "Ollama",
            "model": config.OLLAMA_MODEL,
            "llm_used": False,
            "ollama_base_url": config.OLLAMA_BASE_URL,
            "ollama_reachable": False,
        },
    )

    response = client.post(
        "/agent/chat",
        json={"message": "Which sample demo items are low in availability?", "session_id": "api-llm-unavailable"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == LLM_UNAVAILABLE_MESSAGE
    assert payload["llm_used"] is False


def test_reports_endpoint_returns_excel_download():
    response = client.get("/reports/askmamma")
    assert response.status_code == 200
    payload = response.json()
    assert payload["title"] == "Inventory Pilot AI Online Report"
    assert payload["ai_source"] == "Ollama"
    assert payload["model"] == config.OLLAMA_MODEL
    assert isinstance(payload["llm_used"], bool)
    assert payload["report_content"]
    assert payload["generated_at"]


def test_reports_list_returns_excel_entries():
    client.get("/reports/askmamma")
    response = client.get("/reports")
    assert response.status_code == 200
    payload = response.json()
    assert payload
    assert payload[0]["feature_name"] == "online_report"
    assert payload[0]["model"] == config.OLLAMA_MODEL
    assert "response" in payload[0]


def test_agent_graph_endpoint():
    response = client.get("/agent/graph")
    assert response.status_code == 200
    assert response.json()["format"] == "mermaid"
    assert "SupervisorAgent" in response.json()["graph"]
