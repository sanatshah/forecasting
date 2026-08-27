"""FastAPI dashboard API over pipeline CSV outputs."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .dashboard import aggregations
from .utils import DEFAULT_CONFIG_PATH, load_config

app = FastAPI(title="Retail Forecasting Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_config: Optional[Dict[str, Any]] = None


def get_config() -> Dict[str, Any]:
    global _config
    if _config is None:
        _config = load_config(str(DEFAULT_CONFIG_PATH))
    return _config


def _dataset_error(exc: Exception) -> HTTPException:
    detail = str(exc)
    if isinstance(exc, FileNotFoundError):
        return HTTPException(
            status_code=404,
            detail={
                "message": "Pipeline outputs not found. Run the pipeline first.",
                "hint": "cd retail_forecasting_optimization && ./.venv/bin/python main.py --quick",
                "error": detail,
            },
        )
    return HTTPException(status_code=500, detail={"message": "Failed to load dataset", "error": detail})


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/summary")
def summary() -> Dict[str, Any]:
    try:
        return aggregations.executive_summary(get_config())
    except (FileNotFoundError, KeyError) as exc:
        raise _dataset_error(exc) from exc


@app.get("/api/action-breakdown")
def action_breakdown() -> Dict[str, Any]:
    try:
        return aggregations.action_breakdown(get_config())
    except (FileNotFoundError, KeyError) as exc:
        raise _dataset_error(exc) from exc


@app.get("/api/sku-forecasts")
def sku_forecasts(top_skus: int = Query(default=8, ge=1, le=50)) -> Dict[str, Any]:
    try:
        return aggregations.sku_forecasts(get_config(), top_skus)
    except (FileNotFoundError, KeyError) as exc:
        raise _dataset_error(exc) from exc


@app.get("/api/holdout-forecasts")
def holdout_forecasts(sku_id: str = Query(..., min_length=1)) -> Dict[str, Any]:
    try:
        return aggregations.holdout_forecasts(get_config(), sku_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Holdout predictions file is missing",
                "hint": "cd retail_forecasting_optimization && ./.venv/bin/python main.py --quick",
                "error": str(exc),
            },
        ) from exc
    except KeyError as exc:
        message = exc.args[0] if exc.args else str(exc)
        raise HTTPException(
            status_code=404,
            detail={"message": message, "error": str(exc)},
        ) from exc


@app.get("/api/recommendations")
def recommendations(
    risk: Optional[str] = None,
    action: Optional[str] = None,
    department: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        return aggregations.recommendations_list(
            get_config(),
            risk=risk,
            action=action,
            department=department,
        )
    except (FileNotFoundError, KeyError) as exc:
        raise _dataset_error(exc) from exc


@app.get("/api/metrics/department")
def metrics_department() -> Dict[str, Any]:
    try:
        return aggregations.department_metrics(get_config())
    except (FileNotFoundError, KeyError) as exc:
        raise _dataset_error(exc) from exc
