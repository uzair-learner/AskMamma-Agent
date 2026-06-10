# Inventory Pilot AI Architecture

Inventory Pilot AI is a small local AI application for inventory operations. It keeps the code in one main Python package, uses FastAPI for backend routes, uses React or Streamlit for UI, and sends assistant requests through a LangGraph workflow.

## High-Level Flow

```text
User
  ↓
Streamlit UI
  ↓
FastAPI API
  ↓
Workflow Graph
  ↓
Supervisor Agent
  ↓
Specialist Agent
  ↓
Tools / RAG / Memory
  ↓
Quality Review
  ↓
Final Response
```

The React UI follows the same backend path as Streamlit: user input goes to FastAPI, then to the workflow graph.

## Folder Explanation

```text
src/inventory_pilot_ai/main.py
```

Uvicorn entry point. It exposes the FastAPI `app`.

```text
src/inventory_pilot_ai/config.py
```

Environment settings, paths, model names, JWT settings, upload paths, and vector-store paths.

```text
src/inventory_pilot_ai/api/
```

FastAPI routes, auth, permissions, and request dependencies.

```text
src/inventory_pilot_ai/ui/
```

The optional Streamlit UI.

```text
src/inventory_pilot_ai/agents/
```

One agent per file. These files define the supervisor and specialist agent responsibilities, prompts, routing rules, and tool access.

```text
src/inventory_pilot_ai/workflow/
```

LangGraph state, routing, graph nodes, and final agent invocation.

```text
src/inventory_pilot_ai/memory/
```

Conversation, semantic, and audit-memory readers.

```text
src/inventory_pilot_ai/tools/
```

Inventory, forecast, document, and report tool functions used by agents, API routes, and MCP handlers.

```text
src/inventory_pilot_ai/rag/
```

Document ingestion, embeddings, vector-store rebuilds, and document search.

```text
src/inventory_pilot_ai/db/
```

SQLite schema, seed/reset helpers, data access functions, users, sessions, audit logs, products, suppliers, and reports.

```text
src/inventory_pilot_ai/protocols/
```

Small protocol adapters for MCP, A2A-style task records, and agent-card discovery.

## Request Flow

1. The UI sends requests to FastAPI.
2. FastAPI validates auth and permissions when the endpoint requires it.
3. Inventory and report endpoints call tools or database helpers directly.
4. Chat endpoints call `agents/orchestrator.py`.
5. The orchestrator calls `workflow/graph.py`.
6. The graph returns a structured response with answer, selected agent, tools used, route path, and report bundle.

## Agent Flow

1. `workflow/router.py` classifies the user message.
2. `workflow/graph.py` starts at the supervisor node.
3. The supervisor routes to inventory, forecasting, document, research, or reporting.
4. The specialist agent calls tools through LangChain when an LLM is available.
5. Reporting packages the result.
6. Quality review adds grounding notes when needed.
7. The final response is saved to conversation memory and returned to the API.

## Memory Flow

1. `workflow/graph.py` saves user and assistant messages to SQLite chat history.
2. `tools/inventory_tools.py` writes audit records for agent runs.
3. `memory/service.py` reads conversation, semantic, and audit memory for API display.
4. `/memory/conversation/{session_id}`, `/memory/semantic`, and `/memory/audit` expose memory views.

## UI Flow

1. React UI in `frontend/src/App.tsx` calls FastAPI endpoints.
2. Streamlit UI in `src/inventory_pilot_ai/ui/app.py` calls the same backend.
3. Dashboards read `/dashboard`, `/demo/items`, `/demo/recommendations/reorder`, and related insight endpoints.
4. Chat sends messages to `/agent/chat`.
5. Reports use `/reports/inventory-pilot-ai` and `/reports`.

## API Flow

1. `main.py` imports `api/routes.py`.
2. `api/routes.py` creates the FastAPI app.
3. `api/auth.py` handles login/logout/current-user routes.
4. `api/dependencies.py` checks users, sessions, permissions, and optional auth.
5. Protected routes call `require_permission(...)`.
6. Routes call tools, database helpers, RAG, memory, or the workflow graph.

## Which File Calls Which File

- `main.py` calls `api/routes.py`.
- `api/routes.py` calls `api/auth.py`, `api/dependencies.py`, `db/database.py`, `tools/inventory_tools.py`, `rag/retriever.py`, `agents/orchestrator.py`, and protocol adapters.
- `agents/orchestrator.py` calls `workflow/graph.py`.
- `workflow/graph.py` calls `workflow/router.py`, `agents/catalog.py`, `tools/inventory_tools.py`, `db/database.py`, `llm_provider.py`, and `observability.py`.
- `agents/catalog.py` calls each file in `agents/`.
- `agents/inventory.py`, `agents/forecasting.py`, `agents/document.py`, and `agents/reporting.py` describe which tools each specialist can use.
- `tools/inventory_tools.py` calls `db/database.py` and `rag/retriever.py`.
- `rag/retriever.py` calls `llm_provider.py`, `db/database.py`, and `config.py`.
- `memory/service.py` calls the memory reader files and database-backed memory tables.
- `protocols/mcp/server.py` calls `agents/catalog.py` and `tools/inventory_tools.py`.
