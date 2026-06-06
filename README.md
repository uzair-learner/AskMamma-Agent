# AskMamma Inventory Agent System

AskMamma is a local-first AI-powered inventory management agent system. It supports natural-language stock questions, product lookup, supplier lookup, low-stock alerts, demand forecasting, document Q&A, report generation, memory, tracing, evaluation, and lightweight MCP/A2A-style discovery endpoints.

## Architecture

```text
Streamlit UI
  -> FastAPI backend
    -> SupervisorAgent
      -> InventoryAgent -> inventory tools -> SQLite
      -> ForecastAgent -> sales + forecast tools -> SQLite
      -> DocumentAgent -> local RAG search -> document_chunks
      -> ReportAgent -> markdown reports -> outputs/reports
      -> QualityReviewAgent -> final answer checks
    -> traces + chat memory -> SQLite
```

The core agent is tool-first and works without paid model keys. Ollama, OpenAI, and Azure OpenAI are configurable through `.env` for future LLM-backed response refinement.

## Folder Structure

```text
app.py                    Streamlit dashboard/chat frontend
backend.py                FastAPI backend
agent.py                  Hierarchical multi-agent orchestration
database.py               SQLite schema and CRUD helpers
tools.py                  Typed inventory, forecast, report, movement, audit tools
rag.py                    Document ingestion and local retrieval
llm_provider.py           Ollama/OpenAI/Azure provider abstraction
config.py                 Environment configuration
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
copy .env.example .env
```

Seed the local database and index sample documents:

```powershell
python scripts/seed_data.py
```

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
python -m uvicorn backend:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

## Run Frontend

In another terminal:

```powershell
python -m streamlit run app.py --server.port 8501
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
- `GET /products`
- `GET /products/{id}`
- `POST /products`
- `PUT /products/{id}`
- `DELETE /products/{id}?confirm=true`
- `GET /inventory/low-stock`
- `GET /inventory/out-of-stock`
- `POST /inventory/restock`
- `POST /forecast/demand`
- `POST /agent/chat`
- `POST /agent/run-task`
- `GET /agent/sessions/{session_id}`
- `GET /agent/traces`
- `POST /documents/upload`
- `POST /documents/reindex`
- `POST /documents/search`
- `GET /reports/inventory`
- `GET /reports/forecast`
- `GET /.well-known/agent-card.json`
- `GET /agent/tools`
- `GET /mcp/tools`
- `POST /agent/tasks`

## Test Chat

Try:

```text
Hi
Which products are low in stock?
Do we have USB-C Cable 2m available?
Which supplier provides that item?
Based on previous sales, what demand do you expect next month for Packing Tape?
Search uploaded documents and tell me the return policy.
Generate a short inventory report.
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
  -Body '{"query":"return policy unopened products"}'
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

The evaluation checks greetings, low-stock routing, supplier lookup, forecasting, document search, report generation, and memory follow-ups.

## Tests

```powershell
python -m pytest -q
```

## Docker

```powershell
docker compose up
```

## Troubleshooting

- Backend not reachable: start `uvicorn backend:app --port 8000`.
- Frontend error: confirm backend health at `http://localhost:8000/health`.
- Empty products: run `python scripts/seed_data.py`.
- Document search has no results: run `POST /documents/reindex`.
- Ollama error: run `ollama serve` and `ollama pull llama3.1`.
- Delete product blocked: pass `confirm=true`; destructive actions require explicit confirmation.
