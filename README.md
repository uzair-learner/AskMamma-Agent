# Inventory Pilot AI

Inventory Pilot AI is a small AI-assisted inventory operations app. It helps a user inspect stock, find low-availability items, forecast demand, generate reorder recommendations, search local documents, and create inventory reports.

The app has a FastAPI backend, a React frontend, an optional Streamlit UI, SQLite for local data, LangGraph for agent routing, and optional Ollama/OpenAI/Azure OpenAI support for generated explanations.

The important rule is simple: inventory numbers come from the database and deterministic tools. The LLM may explain the result, but it should not invent stock, supplier, reorder, or forecast data.

## Demo Users

- `admin@example.com` / `AdminPass123!`
- `manager@example.com` / `ManagerPass123!`
- `analyst@example.com` / `AnalystPass123!`
- `viewer@example.com` / `ViewerPass123!`
- `tenantb-viewer@example.com` / `TenantBPass123!`

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe scripts\seed_data.py
```

Run the backend:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m uvicorn inventory_pilot_ai.main:app --host 127.0.0.1 --port 8000 --reload
```

Run the React UI:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

The all-in-one local launcher is:

```powershell
.\scripts\start_all.ps1
```

## Optional Streamlit UI

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m streamlit run src\inventory_pilot_ai\ui\app.py
```

The root `app.py` still works as a small compatibility wrapper:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## How Agents Work

The API sends chat requests to the workflow graph. The supervisor agent chooses the best specialist:

- `InventoryAgent` handles products, suppliers, stock, availability, and reorder context.
- `ForecastAgent` handles sales history, demand forecasts, and reorder planning.
- `DocumentAgent` searches indexed local documents with RAG.
- `ResearchAgent` answers project and architecture questions from local project files.
- `ReportingAgent` packages summaries and report payloads.
- `QualityReviewAgent` checks the answer before it returns to the user.

## Example Request Flow

1. A user asks, “Which products need reorder attention?”
2. The React or Streamlit UI sends the message to FastAPI.
3. FastAPI calls the LangGraph workflow.
4. The supervisor routes the request to the inventory or forecast agent.
5. The specialist agent calls inventory, forecast, document, report, memory, or RAG tools.
6. The reporting step packages the result.
7. Quality review checks grounding and demo-data labeling.
8. FastAPI returns the final answer to the UI.

## Folder Structure

```text
src/
  inventory_pilot_ai/
    main.py
    config.py
    api/
    ui/
    agents/
    workflow/
    memory/
    tools/
    rag/
    db/
    protocols/
tests/
docs/
frontend/
documents/
scripts/
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the detailed file-by-file flow.

## Running With Ollama

Install and start Ollama, then pull the configured model:

```powershell
ollama pull llama3.2
```

Set these in `.env`:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest
```

## Running Without Ollama

Inventory, supplier, reorder, auth, audit, tenant, and document APIs still run. AI endpoints return controlled unavailable responses instead of crashing.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Focused API/security tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_auth.py tests\test_api.py
```
