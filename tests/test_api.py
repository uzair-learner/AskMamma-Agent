from fastapi.testclient import TestClient

from api.backend import app
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
    assert "demo_item_lookup" in payload["skills"]


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
    assert response.json()["assigned_agent"] == "ReportAgent"
    assert response.json()["output_payload"]["selected_agent"] == "ReportAgent"
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
