"""Forward forecasting: build future covariates and generate forecasts.

After model selection has picked a winner, this module produces genuine
forward-looking forecasts for the configured horizons (default 7/14/28 days).

Future known covariates
------------------------
Real future price/promo calendars are usually planned in advance. For the
sample we do not have them, so we build a conservative future frame by carrying
forward each series' last observed price/inventory/hierarchy values and
deriving calendar fields from the future dates. Promo/holiday flags default to
0 for future dates. All of these assumptions are documented in the README and
are easy to replace with a real future calendar.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .model_baseline import get_baseline_models
from .model_ml import MLForecaster
from .utils import get_logger, resolve_path

logger = get_logger(__name__)


# Covariate columns carried forward from the last observed row of each series.
_CARRY_FORWARD = [
    "sku_id",
    "product_id",
    "location_id",
    "department",
    "class",
    "subclass",
    "channel",
    "regular_price",
    "selling_price",
    "markdown_pct",
    "inventory_on_hand",
    "inventory_in_transit",
    "season",
    "product_lifecycle_status",
]


def build_future_frame(
    cleaned_df: pd.DataFrame, config: Dict[str, Any], horizon: int
) -> pd.DataFrame:
    """Create ``horizon`` future rows per series with known covariates filled.

    Calendar fields are computed from the future date; price/inventory/hierarchy
    are carried from the last observed row; promo/holiday default to 0.
    """
    date_col = config["data"]["date_col"]
    rows: List[Dict[str, Any]] = []

    for sid, g in cleaned_df.sort_values(date_col).groupby("series_id"):
        last = g.iloc[-1]
        last_date = last[date_col]
        for step in range(1, horizon + 1):
            fdate = last_date + pd.Timedelta(days=step)
            row: Dict[str, Any] = {"series_id": sid, date_col: fdate}
            for col in _CARRY_FORWARD:
                if col in g.columns:
                    row[col] = last[col]
            # Calendar-derived / planned covariates.
            row["promo_flag"] = 0
            row["promo_event_name"] = None
            row["holiday_flag"] = 0
            row["stockout_flag"] = 0
            row["derived_stockout_flag"] = 0
            row["fiscal_week"] = int(fdate.isocalendar().week)
            row["fiscal_month"] = int(fdate.month)
            row["fiscal_quarter"] = int((fdate.month - 1) // 3 + 1)
            rows.append(row)

    future = pd.DataFrame(rows)
    logger.info(
        "Built future frame: %d rows (%d series x %d days)",
        len(future),
        cleaned_df["series_id"].nunique(),
        horizon,
    )
    return future


def _forecast_with_best(
    best_model_name: str,
    ml_model: MLForecaster,
    cleaned_df: pd.DataFrame,
    future_df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Produce daily forward forecasts using the selected model.

    Baselines are re-fit on the *full* cleaned history; the ML model reuses the
    already-fitted estimator but is re-seeded with full history for recursion.
    """
    date_col = config["data"]["date_col"]
    dims = [
        c
        for c in ["sku_id", "product_id", "location_id", "channel", "department",
                  "class", "subclass", "product_lifecycle_status"]
        if c in future_df.columns
    ]
    frames: List[pd.DataFrame] = []

    if best_model_name == MLForecaster.name:
        ml_model.set_history(cleaned_df)
        for sid, g in future_df.sort_values(date_col).groupby("series_id"):
            preds = ml_model.forecast_recursive(sid, g)
            frame = g[[date_col] + dims].copy()
            frame["series_id"] = sid
            frame["forecast_units"] = np.round(preds, 2)
            frames.append(frame)
    else:
        models = get_baseline_models(config)
        model = models[best_model_name]
        model.fit(cleaned_df)
        for sid, g in future_df.sort_values(date_col).groupby("series_id"):
            h = len(g)
            preds = model.predict(sid, h)
            frame = g[[date_col] + dims].copy()
            frame["series_id"] = sid
            frame["forecast_units"] = np.round(np.asarray(preds[:h]), 2)
            frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    out["model"] = best_model_name
    # Per-series day index (1..horizon) for horizon rollups.
    out["horizon_day"] = out.groupby("series_id")[date_col].rank(method="first").astype(int)
    return out.sort_values(["series_id", date_col]).reset_index(drop=True)


def generate_forecasts(
    comparison: Dict[str, Any],
    cleaned_df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Generate and persist daily forward forecasts for the primary horizon.

    Returns the daily forecast DataFrame and writes ``outputs/forecasts.csv``.
    """
    horizon = max(config["forecast"]["horizons"])
    future_df = build_future_frame(cleaned_df, config, horizon)
    daily = _forecast_with_best(
        comparison["best_model_name"],
        comparison["ml_model"],
        cleaned_df,
        future_df,
        config,
    )

    path = resolve_path(config["paths"]["forecasts_csv"])
    path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(path, index=False)
    logger.info("Saved %d daily forecasts to %s", len(daily), path)
    return daily


def horizon_rollups(
    daily_forecast: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    """Aggregate daily forecasts into cumulative sums for each horizon.

    Returns one row per series per horizon with ``forecast_units`` = total
    predicted demand over the first ``h`` days.
    """
    horizons = config["forecast"]["horizons"]
    dims = [
        c
        for c in ["sku_id", "product_id", "location_id", "channel", "department",
                  "class", "subclass", "product_lifecycle_status"]
        if c in daily_forecast.columns
    ]
    rows: List[pd.DataFrame] = []
    for h in horizons:
        sub = daily_forecast[daily_forecast["horizon_day"] <= h]
        agg = (
            sub.groupby(["series_id"] + dims, as_index=False)["forecast_units"]
            .sum()
        )
        agg["forecast_horizon"] = h
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)
