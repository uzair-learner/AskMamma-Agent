# Inventory Pilot AI

Inventory Pilot AI is a demo FastAPI + React inventory platform with SQLite-backed inventory data, LangGraph orchestration, LangChain tool calling, and optional local Ollama explanations.

## What Is Real In This Demo

- Inventory, reorder, supplier, report, and forecast numbers come from backend tools and SQLite.
- Ollama explains backend-calculated results. It should not invent stock, forecast, or supplier data.
- JWT authentication, RBAC, and tenant filtering are backend-enforced demo security patterns.
- The ResearchAgent is limited to internal project/document research unless external search is added later.

## Demo Users

- `admin@example.com` / `AdminPass123!`
- `manager@example.com` / `ManagerPass123!`
- `analyst@example.com` / `AnalystPass123!`
- `viewer@example.com` / `ViewerPass123!`
- `tenantb-viewer@example.com` / `TenantBPass123!`

## Run

```powershell
.\scripts\start_all.ps1
```

Backend and frontend are served at `http://127.0.0.1:8000`.

## Frontend Only

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Environment

Copy `.env.example` to `.env` and set:

- `LLM_PROVIDER`
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `JWT_SECRET`
- `JWT_ISSUER`
- `JWT_AUDIENCE`
- `JWT_EXPIRY_MINUTES`
- `JWT_ALGORITHM`

Use a strong private `JWT_SECRET` outside demo/local work.
