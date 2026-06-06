from agents.orchestrator import get_session_messages, invoke_agent
from rag.retrieval import document_search
from scripts.seed_data import seed


def setup_module():
    seed()


def test_greeting_uses_no_tools():
    result = invoke_agent("Hi", session_id="test-greeting")
    assert result["selected_agent"] == "SupervisorAgent"
    assert result["tools_called"] == []


def test_low_stock_routes_to_action_tool():
    result = invoke_agent("Which products are low in stock?", session_id="test-low-stock")
    assert result["selected_agent"] == "AskMammaActionAgent"
    assert "AvailabilityStatusTool" in result["tools_called"]


def test_document_search_finds_return_policy():
    result = document_search("return policy unopened products")
    assert result["found"] is True
    assert any("return_policy" in item["file_name"] for item in result["results"])


def test_memory_persists_messages():
    session_id = "test-memory"
    invoke_agent("Do we have USB-C Cable 2m available?", session_id=session_id)
    messages = get_session_messages(session_id)
    assert len(messages) >= 2
    assert messages[-1]["role"] == "assistant"
