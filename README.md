# AskMamma-Agent

AskMamma-Agent is a clean, interview-ready learning project for AI agents. It is designed for a .NET developer who wants to understand how modern agent systems are structured without getting buried in unnecessary framework complexity.

The project demonstrates:

- LangGraph workflow orchestration
- LangChain tool calling and prompt abstractions
- RAG with `RecursiveCharacterTextSplitter`, FAISS, OpenAI embeddings, and local fallback embeddings
- conversation, semantic, and audit memory
- MCP-style tool/resource/prompt discovery with JSON-RPC
- A2A-style task execution with an agent card
- Streamlit observability UI
- optional TensorFlow and PyTorch learning examples for demand prediction

## Architecture

```text
UI
  -> FastAPI API
    -> LangGraph workflow
      -> SupervisorAgent
      -> InventoryAgent / ForecastAgent / DocumentAgent / ResearchAgent
      -> ReportingAgent
      -> QualityReviewAgent
    -> LangChain tools
    -> RAG layer
    -> SQLite memory + traces
    -> MCP + A2A endpoints
```

Supporting diagrams live in:

- [Architecture](docs/architecture-diagram.md)
- [Agents](docs/agent-diagram.md)
- [Workflow](docs/workflow-diagram.md)
- [Interview Guide](docs/interview-guide.md)

## Project Layout

Key folders:

- [agents](agents/README.md)
- [workflows](workflows/README.md)
- [tools](tools/README.md)
- [rag](rag/README.md)
- [memory](memory/README.md)
- [api](api/README.md)
- [ui](ui/README.md)
- [ml](ml/README.md)
- [protocols](protocols/README.md)
- [evaluation](evaluation/README.md)
- [tests](tests/README.md)
- [docs](docs/README.md)

Each folder contains a small README describing its purpose, responsibilities, and interactions.

## Agents

The main agents are:

- `SupervisorAgent`
- `InventoryAgent`
- `ForecastAgent`
- `DocumentAgent`
- `ReportingAgent`
- `QualityReviewAgent`
- `MemoryAgent`
- `ResearchAgent`

Each agent module includes:

- a system prompt
- responsibilities
- routing rules
- tool definitions
- logging guidance
- trace metadata

## LangGraph Flow

The graph is implemented in `workflows/langgraph/workflow.py`. The runtime flow is:

1. `SupervisorAgent` classifies the request.
2. A specialist agent handles inventory, forecasting, document, research, or direct reporting work.
3. `ReportingAgent` packages the result into Markdown, TXT, and JSON views.
4. `QualityReviewAgent` checks grounding and clarity.
5. Memory and traces are stored.

The API also exposes a Mermaid view at `GET /agent/graph`.

## RAG

The RAG pipeline supports:

- `PDF`
- `TXT`
- `DOCX`
- `MD`
- `CSV`

It uses:

- `RecursiveCharacterTextSplitter`
- `FAISS`
- OpenAI or Azure embeddings when configured
- local deterministic embeddings when running offline

## MCP and A2A

MCP-style endpoints:

- `GET /mcp/tools`
- `GET /mcp/resources`
- `GET /mcp/prompts`
- `POST /mcp/rpc`

A2A-style endpoints:

- `GET /.well-known/agent-card.json`
- `POST /agent/tasks`
- `GET /agent/tasks/{task_id}`

## UI

The Streamlit UI now includes:

- agent activity panel
- tool activity panel
- LangGraph Mermaid view
- memory viewer
- MCP viewer
- forecast chart

Run it through [ui/app.py](ui/app.py) once the backend is up.

## ML Examples

Optional learning examples are included in:

- [ml/tensorflow](ml/tensorflow/README.md)
- [ml/pytorch](ml/pytorch/README.md)

These scripts are intentionally lightweight and are not required for the core app to run.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
python scripts/seed_data.py
python -m uvicorn api.backend:app --host 127.0.0.1 --port 8000 --reload
```

Streamlit:

```powershell
streamlit run ui/app.py
```

One-command launcher:

```powershell
.\start.cmd
```

## Tests and Evaluation

```powershell
python -m pytest -q
python scripts/evaluate_agent.py
```

## Notes

- The inventory and document data are seeded demo data.
- The ML examples are for learning and comparison, not production forecasting.
- If no hosted model is configured, the workflow falls back to deterministic behavior so the project still runs offline.
