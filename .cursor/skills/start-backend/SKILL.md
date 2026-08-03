---
name: start-backend
description: Starts the FastAPI dashboard API that serves retail forecasting pipeline CSV outputs. Use when the user asks to start the backend, API, dashboard API, or uvicorn for the forecast dashboard.
---

# Start Backend (Dashboard API)

## Scope

FastAPI app in `retail_forecasting_optimization/src/dashboard_api.py`.

- Serves JSON over `/api/*` from live `outputs/*.csv`
- Default: http://127.0.0.1:8000
- CORS allows the Vite UI on http://localhost:5173
- For the React UI, use the **start-frontend** skill (it starts this API too)

## Prerequisites

1. **Working directory**: `retail_forecasting_optimization/`
2. **Virtual env**: `.venv` with deps (`uvicorn[standard]` is in `requirements.txt`)
3. **Pipeline outputs** (for data endpoints): `outputs/forecasts.csv`, `outputs/recommendations.csv`, etc. Health works without them; summary/recs return 404 until the pipeline has run.

If outputs are missing, run the **run-pipeline** skill first.

## Workflow checklist

Copy and track progress:

```
Backend start:
- [ ] cd retail_forecasting_optimization
- [ ] Confirm .venv exists (install requirements if needed)
- [ ] Check terminals — reuse uvicorn if already on :8000
- [ ] Start uvicorn --reload on port 8000 (background)
- [ ] Verify GET /api/health returns {"status":"ok"}
```

## Start command

**Check existing terminals first.** If uvicorn is already listening on 8000, reuse it.

```bash
# macOS / Linux
./.venv/bin/uvicorn src.dashboard_api:app --reload --port 8000

# Windows / PowerShell
.\.venv\Scripts\python.exe -m uvicorn src.dashboard_api:app --reload --port 8000
```

Run in the **background** (`block_until_ms: 0`). Ready when logs show Uvicorn listening on port 8000.

Optional: OpenAPI docs at http://127.0.0.1:8000/docs

## Key endpoints

| Method | Path | Notes |
|--------|------|-------|
| GET | `/api/health` | Liveness; no CSVs required |
| GET | `/api/summary` | Executive KPIs |
| GET | `/api/action-breakdown` | Actions by department |
| GET | `/api/sku-forecasts?top_skus=8` | Forward curves |
| GET | `/api/recommendations` | Optional `risk`, `action`, `department` filters |
| GET | `/api/metrics/department` | WAPE by department |

## Success criteria

- Process running in background
- `curl -s http://127.0.0.1:8000/api/health` → `{"status":"ok"}`
- Tell the user the base URL: **http://127.0.0.1:8000**

## Failure handling

| Symptom | Action |
|---------|--------|
| Port 8000 in use | Reuse existing API, or stop the occupant and restart |
| `uvicorn` / import errors | `./.venv/bin/pip install -r requirements.txt` |
| `/api/health` ok but data 404 | Run pipeline (`run-pipeline`), then retry data endpoints (`--reload` picks up new files on next request) |
| Wrong working directory | Must run from `retail_forecasting_optimization/` so `src.dashboard_api` and config paths resolve |

## Do not

- Block the agent session on the long-running server — always background it
- Start a second copy if one is already healthy
- Commit regenerated `outputs/` unless the user asks
