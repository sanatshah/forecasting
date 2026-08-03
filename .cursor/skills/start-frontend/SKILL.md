---
name: start-frontend
description: Starts the Macy's-themed React dashboard (Vite) and its FastAPI backend for the retail forecasting pipeline. Use when the user asks to start the frontend, run the dashboard UI, open the web app, or npm run dev for the forecast dashboard.
---

# Start Frontend Dashboard

## Scope

Macy's-themed React dashboard under `retail_forecasting_optimization/frontend/`.

- **UI**: Vite + React on http://localhost:5173
- **API**: FastAPI (`src.dashboard_api`) on http://127.0.0.1:8000
- Vite proxies `/api` → port 8000
- API-only: use the **start-backend** skill

## Prerequisites

1. **Pipeline outputs**: `retail_forecasting_optimization/outputs/*.csv` must exist (API reads them). If missing, run the **run-pipeline** skill first (`python main.py` or `--quick`).
2. **Working directory for API**: `retail_forecasting_optimization/`
3. **Working directory for UI**: `retail_forecasting_optimization/frontend/`
4. **Node**: Node.js + npm available
5. **Python venv**: `retail_forecasting_optimization/.venv` with deps installed

## Workflow checklist

Copy and track progress:

```
Frontend start:
- [ ] Confirm outputs/ CSVs exist (or run pipeline)
- [ ] Check terminals — reuse API/UI if already running
- [ ] Start API on :8000 (background) if needed
- [ ] npm install if node_modules missing
- [ ] Start Vite dev server on :5173 (background)
- [ ] Confirm UI URL and tell the user
```

## Start commands

**Check existing terminals first.** If uvicorn or `npm run dev` is already running, reuse those processes — do not start duplicates.

### 1. API (required for data)

From `retail_forecasting_optimization/`:

```bash
# macOS / Linux
./.venv/bin/uvicorn src.dashboard_api:app --reload --port 8000

# Windows / PowerShell
.\.venv\Scripts\python.exe -m uvicorn src.dashboard_api:app --reload --port 8000
```

Run in the **background** (`block_until_ms: 0`). Ready when logs show Uvicorn listening on port 8000.

### 2. UI

From `retail_forecasting_optimization/frontend/`:

```bash
# First time / after package.json changes
npm install

npm run dev
```

Run in the **background**. Ready when Vite prints the local URL (default http://localhost:5173).

## Success criteria

- API responds (e.g. http://127.0.0.1:8000/api/health or Vite proxy via UI)
- Vite reachable at **http://localhost:5173**
- Tell the user both URLs; note that the UI is the entry point

## Failure handling

| Symptom | Action |
|---------|--------|
| Empty / error UI, API unreachable | Start uvicorn on :8000; confirm Vite proxy in `frontend/vite.config.ts` |
| Port 8000 or 5173 in use | Reuse existing process, or stop the occupant and restart |
| Missing CSVs / empty data | Run pipeline (`run-pipeline` skill), then restart API if needed |
| `npm` / module errors | `npm install` in `frontend/` |
| Import / uvicorn missing | `./.venv/bin/pip install -r requirements.txt` from `retail_forecasting_optimization/` |

## Do not

- Commit `node_modules/`, build artifacts, or regenerated `outputs/`
- Block the agent session on long-running servers — always background them
- Start a second copy if one is already healthy in a terminal
