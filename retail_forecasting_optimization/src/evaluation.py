"""Retail forecast accuracy metrics and multi-level evaluation.

Metrics implemented
--------------------
* **WAPE**  - Weighted Absolute Percentage Error = sum|A-F| / sum|A|.
              Robust to zeros; the primary retail accuracy metric.
* **MAPE**  - Mean Absolute Percentage Error with safe zero handling (rows
              where actual == 0 are excluded from the mean).
* **MAE**   - Mean Absolute Error.
* **RMSE**  - Root Mean Squared Error.
* **Bias**  - Mean (Forecast - Actual); positive = over-forecast.
* **Forecast accuracy %** = (1 - WAPE) clipped to [0, 1].

Evaluation can be sliced by any set of dimensions (department, channel,
location, sku, promo vs non-promo, lifecycle status).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .utils import get_logger, safe_divide

logger = get_logger(__name__)


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Weighted Absolute Percentage Error. Returns NaN if total actual is 0."""
    actual = np.asarray(actual, dtype="float64")
    forecast = np.asarray(forecast, dtype="float64")
    denom = np.abs(actual).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(actual - forecast).sum() / denom)


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean Absolute Percentage Error, excluding rows where actual == 0."""
    actual = np.asarray(actual, dtype="float64")
    forecast = np.asarray(forecast, dtype="float64")
    mask = actual != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])))


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean Absolute Error."""
    actual = np.asarray(actual, dtype="float64")
    forecast = np.asarray(forecast, dtype="float64")
    return float(np.mean(np.abs(actual - forecast)))


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Root Mean Squared Error."""
    actual = np.asarray(actual, dtype="float64")
    forecast = np.asarray(forecast, dtype="float64")
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean forecast bias (Forecast - Actual). Positive = over-forecast."""
    actual = np.asarray(actual, dtype="float64")
    forecast = np.asarray(forecast, dtype="float64")
    return float(np.mean(forecast - actual))


def compute_metrics(actual: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
    """Return the full metric dictionary for one aligned actual/forecast pair."""
    w = wape(actual, forecast)
    return {
        "wape": w,
        "mape": mape(actual, forecast),
        "mae": mae(actual, forecast),
        "rmse": rmse(actual, forecast),
        "bias": bias(actual, forecast),
        "forecast_accuracy": float(np.clip(1 - w, 0, 1)) if not np.isnan(w) else float("nan"),
        "n": int(len(actual)),
    }


def evaluate_by(
    df: pd.DataFrame,
    actual_col: str = "actual",
    forecast_col: str = "forecast",
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute metrics overall and (optionally) within each group.

    Parameters
    ----------
    df:
        Frame containing aligned actuals and forecasts (one row per prediction).
    group_cols:
        If provided, metrics are computed per unique combination *in addition*
        to an ``__overall__`` row.
    """
    rows: List[Dict[str, object]] = []

    overall = compute_metrics(df[actual_col].to_numpy(), df[forecast_col].to_numpy())
    overall_row = {"level": "overall", "group": "__overall__", **overall}
    rows.append(overall_row)

    if group_cols:
        for col in group_cols:
            if col not in df.columns:
                continue
            for key, g in df.groupby(col):
                m = compute_metrics(g[actual_col].to_numpy(), g[forecast_col].to_numpy())
                rows.append({"level": col, "group": str(key), **m})

    return pd.DataFrame(rows)


def evaluate_promo_split(
    df: pd.DataFrame,
    actual_col: str = "actual",
    forecast_col: str = "forecast",
    promo_col: str = "promo_flag",
) -> pd.DataFrame:
    """Compute metrics separately for promo vs non-promo periods."""
    rows: List[Dict[str, object]] = []
    if promo_col not in df.columns:
        return pd.DataFrame(rows)
    for label, sub in [
        ("promo", df[df[promo_col] == 1]),
        ("non_promo", df[df[promo_col] != 1]),
    ]:
        if len(sub) == 0:
            continue
        m = compute_metrics(sub[actual_col].to_numpy(), sub[forecast_col].to_numpy())
        rows.append({"level": "promo_period", "group": label, **m})
    return pd.DataFrame(rows)


def full_evaluation(
    df: pd.DataFrame,
    actual_col: str = "actual",
    forecast_col: str = "forecast",
) -> pd.DataFrame:
    """Run the complete multi-level evaluation used in the pipeline.

    Slices: overall, department, channel, location, sku, lifecycle status,
    and promo vs non-promo.
    """
    group_cols = [
        c
        for c in [
            "department",
            "channel",
            "location_id",
            "sku_id",
            "product_lifecycle_status",
        ]
        if c in df.columns
    ]
    base = evaluate_by(df, actual_col, forecast_col, group_cols)
    promo = evaluate_promo_split(df, actual_col, forecast_col)
    return pd.concat([base, promo], ignore_index=True)
