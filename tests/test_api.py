from fastapi.testclient import TestClient

from api.backend import app
from core import config
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
    assert isinstance(payload["fallback_used"], bool)
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
    assert "fallback_mode_active" in payload
    assert isinstance(payload["recent_requests"], list)
    if payload["recent_requests"]:
        recent = payload["recent_requests"][0]
        assert "provider" in recent
        assert "model" in recent
        assert "response_time_ms" in recent


def test_reports_endpoint_returns_excel_download():
    response = client.get("/reports/askmamma")
    assert response.status_code == 200
    payload = response.json()
    assert payload["file_name"].endswith(".md")
    assert payload["download_url"].endswith(payload["file_name"])

    download = client.get(payload["download_url"])
    assert download.status_code == 200
    assert "text/markdown" in download.headers["content-type"] or "text/plain" in download.headers["content-type"]

    report_path = config.REPORT_DIR / payload["file_name"]
    assert report_path.exists()


def test_reports_list_returns_excel_entries():
    client.get("/reports/askmamma")
    response = client.get("/reports")
    assert response.status_code == 200
    payload = response.json()
    assert payload
    assert payload[0]["file_name"].endswith(".md")
    assert payload[0]["download_url"].endswith(payload[0]["file_name"])


def test_agent_graph_endpoint():
    response = client.get("/agent/graph")
    assert response.status_code == 200
    assert response.json()["format"] == "mermaid"
    assert "SupervisorAgent" in response.json()["graph"]
