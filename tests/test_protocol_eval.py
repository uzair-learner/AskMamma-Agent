from fastapi.testclient import TestClient

from api.backend import app
from scripts.evaluate_agent import evaluate_cases
from scripts.seed_data import seed


def setup_module():
    seed()


client = TestClient(app)


def test_mcp_tools_endpoint():
    response = client.get("/mcp/tools")
    assert response.status_code == 200
    assert any(tool["name"] == "DocumentSearchTool" for tool in response.json())


def test_agent_card_has_a2a_style_fields():
    payload = client.get("/.well-known/agent-card.json").json()
    assert {"name", "description", "version", "endpoint", "capabilities", "authentication", "skills"}.issubset(payload)
    assert "jsonrpc" in payload["supported_input_modes"]
    assert payload["examples"]


def test_evaluation_script_passes():
    summary = evaluate_cases()
    assert summary["passed"] == summary["total"]
    assert summary["categories"]["simple_answer"]["passed"] >= 1
