from agents.orchestrator import get_session_messages, invoke_agent
from core import config
from core.llm_provider import LLM_UNAVAILABLE_MESSAGE
from rag.retrieval import document_search, reindex_documents
from scripts.seed_data import seed


def setup_module():
    seed()


def test_greeting_uses_supervisor_only():
    result = invoke_agent("Hi", session_id="test-greeting")
    assert result["selected_agent"] == "SupervisorAgent"
    assert result["tools_called"] == []
    assert result["route_path"] == ["SupervisorAgent"]
    assert result["trace_backend"] in {"sqlite", "langsmith"}


def test_demo_low_availability_routes_to_action_tool():
    result = invoke_agent("Which sample demo items are low in availability?", session_id="test-low-stock")
    assert result["selected_agent"] == "InventoryAgent"
    assert "DemoAvailabilityTool" in result["tools_called"]
    assert "ReportingAgent" in result["route_path"]
    assert "QualityReviewAgent" in result["route_path"]


def test_document_search_uses_vector_retriever():
    reindex_summary = reindex_documents()
    assert reindex_summary["vector_store"]["indexed_chunks"] > 0
    result = document_search("return policy unopened demo items")
    assert result["found"] is True
    assert result["retriever"] == "faiss+local-hash-embeddings"
    assert any("return_policy" in item["file_name"] for item in result["results"])


def test_memory_persists_messages():
    session_id = "test-memory"
    invoke_agent("Do we have USB-C Cable 2m available?", session_id=session_id)
    messages = get_session_messages(session_id)
    assert len(messages) >= 2
    assert messages[-1]["role"] == "assistant"


def test_forecast_route_has_intermediate_steps():
    result = invoke_agent(
        "Based on the sample demo history, what demand do you expect next month for Packing Tape?",
        session_id="test-forecast",
    )
    assert result["selected_agent"] == "ForecastAgent"
    assert "DemoForecastTool" in result["tools_called"]
    assert "ReportingAgent" in result["route_path"]
    assert result["intermediate_steps"]


def test_langsmith_disabled_uses_local_trace_backend(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    result = invoke_agent("Hi", session_id="test-langsmith-disabled")
    assert result["trace_backend"] == "sqlite"


def test_returns_clear_message_when_llm_unavailable(monkeypatch):
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

    result = invoke_agent("Which sample demo items are low in availability?", session_id="test-llm-unavailable")
    assert result["answer"] == LLM_UNAVAILABLE_MESSAGE
    assert result["llm_used"] is False
    assert result["selected_agent"] == "SupervisorAgent"
