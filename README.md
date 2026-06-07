# AskMamma Agent System

AskMamma-Agent is a local AI agent system demonstrating:
- FastAPI backend
- Streamlit UI
- RAG/document retrieval
- tool/action calling
- memory
- tracing
- tests
- local SQLite runtime state

The repository includes seeded sample demo item and history data used only for demonstration of availability checks, partner lookup, forecasting, and report generation. That sample data is not the core identity of the project.

## Architecture

```text
Streamlit UI
  -> FastAPI backend
    -> SupervisorAgent
      -> AskMammaActionAgent -> AskMamma tools -> SQLite
      -> ForecastAgent -> demo history + forecast tools -> SQLite
      -> DocumentAgent -> local RAG search -> document_chunks
      -> ReportAgent -> markdown reports -> outputs/reports
      -> QualityReviewAgent -> final answer checks
    -> traces + chat memory -> SQLite
```

The core agent is tool-first and works without paid model keys. Ollama, OpenAI, and Azure OpenAI are configurable through `.env` for future LLM-backed response refinement.

## Folder Structure

```text
api/backend.py            FastAPI backend
ui/app.py                 Streamlit dashboard/chat frontend
agents/orchestrator.py    Hierarchical multi-agent orchestration
db/database.py            SQLite schema and CRUD helpers
askmamma/tools.py         Typed AskMamma demo tools, forecast, report, movement, audit tools
rag/retrieval.py          Document ingestion and local retrieval
core/llm_provider.py      Ollama/OpenAI/Azure provider abstraction
core/config.py            Environment configuration
scripts/seed_data.py      Demo data and document indexing
scripts/evaluate_agent.py Route/tool evaluation
tests/                    API, database, tools, RAG, memory, routing tests
documents/                Sample knowledge-base documents
outputs/reports/          Generated reports
```

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.template .env
```

Seed the local database and index sample documents:

```powershell
python scripts/seed_data.py
```

The seed script creates a reproducible local demo database with 12 sample partners, a broad AskMamma sample catalog, 18 months of demo history, movement records, and indexed knowledge-base documents. The generated SQLite file is runtime data and is intentionally ignored by Git.

## Run Ollama

Ollama is optional for the current deterministic tool-backed agent, but the provider is configured for local-first LLM support.

```powershell
ollama serve
ollama pull llama3.1
```

In `.env`:

```text
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
```

To switch providers, set `LLM_PROVIDER=openai` with `OPENAI_API_KEY`, or `LLM_PROVIDER=azure` with Azure OpenAI settings.

## Run Backend

```powershell
python -m uvicorn api.backend:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

## Run Frontend

In another terminal:

```powershell
python -m streamlit run ui/app.py --server.port 8501
```

Open:

```text
http://localhost:8501
```

Or start both with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_all.ps1
```

## Key API Endpoints

- `GET /health`
- `GET /dashboard`
- `GET /demo/items`
- `GET /demo/items/{id}`
- `POST /demo/items`
- `PUT /demo/items/{id}`
- `DELETE /demo/items/{id}?confirm=true`
- `GET /demo/availability/low`
- `GET /demo/availability/out`
- `POST /demo/availability/restock`
- `POST /demo/forecast`
- `POST /agent/chat`
- `POST /agent/run-task`
- `GET /agent/sessions/{session_id}`
- `GET /agent/traces`
- `POST /documents/upload`
- `POST /documents/reindex`
- `POST /documents/search`
- `GET /reports/askmamma`
- `GET /reports/demo`
- `GET /reports/demo-forecast`
- `GET /.well-known/agent-card.json`
- `GET /agent/tools`
- `GET /mcp/tools`
- `POST /agent/tasks`

## Test Chat

Try:

```text
Hi
Which sample demo items are low in availability?
Do we have USB-C Cable 2m available in the sample demo catalog?
Which demo partner provides that item?
Based on the sample demo history, what demand do you expect next month for Packing Tape?
Search uploaded documents and tell me the return policy.
Generate a short AskMamma report.
```

## Documents

Sample docs live in `documents/`. Reindex them:

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

## Traces

Every agent run stores:

- session ID
- user input
- selected agent
- tools called
- tool inputs
- tool output summaries
- final answer
- latency
- errors
- timestamp

View traces:

```powershell
Invoke-RestMethod http://localhost:8000/agent/traces
```

The Streamlit UI also has a `View traces` button.

## Evaluation

```powershell
python scripts/evaluate_agent.py
```

The evaluation checks greetings, demo low-availability routing, partner lookup, forecasting, document search, report generation, and memory follow-ups.

## Tests

```powershell
python -m pytest -q
```

## Docker

```powershell
docker compose up
```

## Troubleshooting

- Backend not reachable: start `uvicorn api.backend:app --port 8000`.
- Frontend error: confirm backend health at `http://localhost:8000/health`.
- Empty demo items: run `python scripts/seed_data.py`.
- Document search has no results: run `POST /documents/reindex`.
- Ollama error: run `ollama serve` and `ollama pull llama3.1`.
- Delete demo item blocked: pass `confirm=true`; destructive actions require explicit confirmation.
