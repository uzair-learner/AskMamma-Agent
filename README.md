# AskMamma Agent System

AskMamma-Agent is a local AI agent demo that combines:
- FastAPI backend
- Streamlit UI
- LangGraph multi-agent orchestration
- LangChain tool calling
- embedding-backed RAG over local documents
- SQLite memory and trace storage
- optional LangSmith tracing
- automated tests and evaluation

The seeded catalog, availability, partner, movement, forecast, and report flows are sample demo data. They exist to demonstrate agent patterns and are not the core identity of the project.

## Architecture

In simple terms:

```text
Streamlit UI
  -> FastAPI API
    -> LangGraph supervisor
      -> AskMammaActionAgent
      -> ForecastAgent
      -> DocumentAgent
      -> ReportAgent
      -> QualityReviewAgent
    -> LangChain tools
    -> SQLite state + local trace fallback
    -> FAISS vector store for document retrieval
    -> Optional LangSmith tracing
```

What each layer does:
- `api/backend.py`: public API, task endpoints, MCP-style adapter, and metadata endpoints
- `agents/orchestrator.py`: LangGraph routing plus deterministic fallback when no paid chat model is configured
- `askmamma/tools.py`: typed tools with clear schemas for item lookup, availability, partner lookup, forecast, recommendations, document search, reporting, and audit logging
- `rag/retrieval.py`: document ingestion, chunking, local embeddings, and FAISS retrieval
- `db/database.py`: SQLite schema for demo items, memory, traces, and documents
- `ui/app.py`: Streamlit dashboard and chat interface

## Folder Structure

```text
api/backend.py            FastAPI backend
ui/app.py                 Streamlit dashboard/chat frontend
agents/orchestrator.py    LangGraph multi-agent orchestration
askmamma/tools.py         LangChain-compatible AskMamma demo tools
rag/retrieval.py          Document ingestion and embedding retrieval
db/database.py            SQLite schema and CRUD helpers
core/llm_provider.py      OpenAI, Azure OpenAI, and Ollama provider config
core/observability.py     LangSmith setup and redaction helpers
scripts/seed_data.py      Demo data and document indexing
scripts/evaluate_agent.py Route/tool/answer/intermediate-step evaluation
tests/                    API, RAG, routing, tool, MCP, and evaluation tests
documents/                Sample knowledge-base documents
```

## One-Command Start

From PowerShell, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_all.ps1
```

Or, for the simplest one-command launcher from the project root:

```powershell
.\start.cmd
```

This script:
- creates or repairs `.venv`
- upgrades `pip`
- installs `requirements.txt`
- installs frontend dependencies when needed
- builds the React frontend
- creates `.env` from `.env.example` if needed
- seeds the local database and vector store
- starts the FastAPI backend
- opens `http://127.0.0.1:8000` in your browser

Open:

```text
http://localhost:8501
```

## Manual Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
```

Seed the local database and build the vector index:

```powershell
python scripts/seed_data.py
```

## Environment

The tracked template is `.env.example`.

Important settings:

```text
APP_ENV=development
DATABASE_URL=sqlite:///askmamma.db
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
OPENAI_MODEL=gpt-4o-mini

OPENAI_API_KEY=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT=
AZURE_OPENAI_API_VERSION=2024-10-21

LANGSMITH_API_KEY=
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=askmamma-agent
LANGSMITH_TRACING=true
```

Provider behavior:
- `LLM_PROVIDER=openai`: uses LangChain `ChatOpenAI` if `OPENAI_API_KEY` is set
- `LLM_PROVIDER=azure`: uses LangChain `AzureChatOpenAI` if Azure settings are set
- `LLM_PROVIDER=ollama`: keeps local Ollama generation config available, while the agent falls back to deterministic graph behavior when no tool-calling chat model is configured

## Run Backend

```powershell
python -m uvicorn api.backend:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

## Build Frontend

The React app lives in `frontend/` and is served by FastAPI after build.

Build it with:

```powershell
cd frontend
npm install
npm run build
```

Open:

```text
http://127.0.0.1:8000
```

## Key API Endpoints

- `POST /agent/chat`
- `POST /agent/run-task`
- `POST /agent/tasks`
- `GET /agent/tools`
- `GET /agent/traces`
- `GET /agent/sessions/{session_id}`
- `GET /demo/items`
- `GET /demo/availability/low`
- `GET /demo/availability/out`
- `POST /demo/availability/restock`
- `POST /demo/forecast`
- `POST /documents/upload`
- `POST /documents/reindex`
- `POST /documents/search`
- `GET /mcp/tools`
- `POST /mcp/rpc`
- `GET /.well-known/agent-card.json`

## RAG

The document pipeline now uses:
- `RecursiveCharacterTextSplitter`
- local deterministic embeddings
- `FAISS`
- local vector store persistence in `vector_store/`

Reindex documents:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/documents/reindex
```

Search documents:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/documents/search `
  -ContentType application/json `
  -Body '{"query":"return policy unopened demo items"}'
```

Upload supports `.pdf`, `.txt`, `.md`, and `.csv`.

## MCP and Agent Metadata

This project includes a lightweight MCP-style adapter:
- `GET /mcp/tools` lists available tools
- `POST /mcp/rpc` supports JSON-RPC style `tools/list` and `tools/call`

The agent card at `/.well-known/agent-card.json` includes:
- name
- description
- version
- endpoint
- capabilities
- authentication
- skills
- supported input modes
- supported output modes

## LangSmith

LangSmith tracing is optional.

If you set:
- `LANGSMITH_API_KEY`
- `LANGSMITH_ENDPOINT`
- `LANGSMITH_PROJECT`

the app enables LangSmith tracing automatically. If not configured, the project still stores local traces in SQLite.

## Demo-Only Features

These are intentionally demo/sample features:
- seeded item catalog and availability data
- partner and movement records
- historical sales and forecast examples
- reorder recommendations
- generated markdown reports

They are useful for demonstrating agent tooling, not for representing a production AskMamma business domain.

## Safety and Guardrails

The backend includes:
- Pydantic input validation
- explicit confirmation for destructive writes and movement writes
- in-memory rate limiting for agent and MCP endpoints
- secret redaction in stored traces
- clear demo/sample labels in tool and answer wording
- local SQLite trace fallback when LangSmith is not configured

## Evaluation

Run the evaluation suite:

```powershell
python scripts/evaluate_agent.py
```

It checks:
- final answer quality
- selected route
- expected tools called
- route path and intermediate steps

## Tests

Run:

```powershell
python -m pytest -q
```

Coverage includes:
- embedding-backed RAG retrieval
- tool invocation
- memory persistence
- LangGraph routing
- API endpoints
- MCP/A2A-style metadata endpoints
- evaluation script behavior

## Troubleshooting

- Backend not reachable: start `uvicorn api.backend:app --port 8000`
- Empty demo items or missing vectors: run `python scripts/seed_data.py`
- Document results missing: run `POST /documents/reindex`
- Ollama not reachable: run `ollama serve` and `ollama pull llama3.1`
- LangSmith not tracing: verify `LANGSMITH_API_KEY`, `LANGSMITH_ENDPOINT`, and `LANGSMITH_PROJECT`
- Demo movement blocked: pass `"confirm": true`
