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
    # skills now include demo/sample annotations
    skills = response.json()["skills"]
    assert any("demo_item_lookup" in s for s in skills)


def test_tool_registry_endpoint():
    response = client.get("/agent/tools")
    assert response.status_code == 200
    names = {tool["name"] for tool in response.json()}
    assert "DemoForecastTool" in names


def test_chat_endpoint():
    response = client.post("/agent/chat", json={"message": "In the sample demo catalog, which items are low in availability?", "session_id": "api-test"})
    assert response.status_code == 200
    assert "DemoAvailabilityTool" in response.json()["tools_called"]
