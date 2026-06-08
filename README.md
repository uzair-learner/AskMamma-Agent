# Inventory Pilot AI

Inventory Pilot AI / AskMamma Agent is a demo AI-assisted inventory operations platform built for interview demonstration. It uses a FastAPI backend, React/TypeScript frontend, SQLite data store, LangGraph orchestration, LangChain tool calling, and optional local Ollama explanations.

Inventory calculations come from backend tools and SQLite. Ollama explains results but should not invent inventory, supplier, reorder, or forecast data.

## CV Alignment

- FastAPI backend: inventory, supplier, forecast, report, auth, audit, document, and agent APIs.
- React/TypeScript frontend: Vite React UI converted to TSX with typed user/product/supplier/reorder/AI response models.
- JWT authentication: `/auth/login`, `/auth/me`, `/auth/logout`, signed JWTs, backend session validation.
- RBAC: backend-enforced roles for admin, manager, analyst, and viewer.
- Inventory tracking: product catalog, stock levels, low/out-of-stock views, stock movement endpoint.
- Reorder intelligence: backend reorder recommendations and supplier reorder request workflow.
- Supplier workflow: reorder requests associated to suppliers with Draft, Submitted, Approved, Rejected, and Completed statuses.
- Audit logging: product writes, stock movements, login/logout, reorder request changes, and AI generation events.
- Local LLM/Ollama integration: page insights, reports, and assistant responses.
- Multi-agent assistant workflow: LangGraph supervisor routes to Inventory, Forecast, Document, Reporting, Research, and Quality Review agents.
- Forecasting concept: deterministic demo forecasting and clear AI explanation layer; TensorFlow sample is optional and not required for app startup.
- Tenant-aware architecture concept: users belong to tenants, and protected product, supplier, report, document, audit, and AI data paths filter by tenant.

## Demo Users

- `admin@example.com` / `AdminPass123!`
- `manager@example.com` / `ManagerPass123!`
- `analyst@example.com` / `AnalystPass123!`
- `viewer@example.com` / `ViewerPass123!`
- `tenantb-viewer@example.com` / `TenantBPass123!`

## Backend Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe scripts\seed_data.py
.\.venv\Scripts\python.exe -m uvicorn api.backend:app --host 127.0.0.1 --port 8000 --reload
```

## Frontend Setup

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

The all-in-one local launcher is:

```powershell
.\scripts\start_all.ps1
```

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

The inventory, supplier, reorder, auth, audit, and tenant APIs still run. AI endpoints return controlled unavailable responses instead of crashing the app.

## Tests

Focused fast/security tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_auth.py tests\test_api.py
```

Targeted RAG test:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_agent_rag.py::test_document_search_uses_vector_retriever
```

Full test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Some full-suite tests call local Ollama and can be slow.

## Required Environment Variables

- `DATABASE_URL`
- `LLM_PROVIDER`
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `JWT_SECRET`
- `JWT_ISSUER`
- `JWT_AUDIENCE`
- `JWT_EXPIRY_MINUTES`
- `JWT_ALGORITHM`

Use a strong private `JWT_SECRET` outside local demo work.

## Limitations / Future Enhancements

- Full production SaaS billing is not implemented.
- Production cloud deployment and infrastructure are documented concepts, not provisioned infrastructure.
- Advanced ML forecasting is not production-grade; current forecasting is a concept/demo workflow.
- Enterprise SSO is not implemented.
- Production monitoring, alerting, and SIEM integration are not implemented.
