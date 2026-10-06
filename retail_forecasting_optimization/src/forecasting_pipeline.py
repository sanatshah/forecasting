"""Forward forecasting: build future covariates, forecast each target, derive KPIs.

After model selection has picked a winner per target, this module produces
forward-looking forecasts for the configured horizons (default 7/14/28 days)
and combines them into the subscriber KPIs:

* ``net_adds = gross_adds - churned_subs``
* ``paid_subs = opening base + cumulative net adds``
* ``hours_per_paid_sub = hours_watched / paid_subs``

Future known covariates
------------------------
The content calendar (tentpoles, holidays) is known in advance and is joined
from ``content_calendar.csv`` - the main structural advantage over retail,
where future promos were unknown. Price, segment attributes and the opening
subscriber base are carried forward from the last observed row (planned price
changes are evaluated as scenarios, not baked into the baseline). Promo flags
default to 0 for future dates.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .model_baseline import get_baseline_models
from .model_chronos import ChronosForecaster
from .model_ml import MLForecaster
from .utils import get_logger, resolve_path, safe_divide

logger = get_logger(__name__)


# Covariate columns carried forward from the last observed row of each series.
_CARRY_FORWARD = [
    "tier",
    "acquisition_channel",
    "distribution_partner",
    "list_price",
    "effective_price",
    "season",
]

_CALENDAR_COLUMNS = [
    "tentpole_flag",
    "tentpole_name",
    "tentpole_type",
    "tentpole_intensity",
    "holiday_flag",
]

SEGMENT_DIMS = ["tier", "acquisition_channel", "distribution_partner"]


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def build_future_frame(
    cleaned_df: pd.DataFrame,
    config: Dict[str, Any],
    horizon: int,
    calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create ``horizon`` future rows per series with known covariates filled.

    Calendar fields come from the future date and the content calendar when
    provided; price and segment attributes are carried from the last observed
    row; ``paid_subs_bod`` is the last observed ``paid_subs_eod``.
    """
    date_col = config["data"]["date_col"]
    rows: List[Dict[str, Any]] = []

    for sid, g in cleaned_df.sort_values(date_col).groupby("series_id"):
        last = g.iloc[-1]
        last_date = last[date_col]
        opening = last.get("paid_subs_eod", last.get("paid_subs_bod", 0))
        for step in range(1, horizon + 1):
            fdate = last_date + pd.Timedelta(days=step)
            row: Dict[str, Any] = {"series_id": sid, date_col: fdate}
            for col in _CARRY_FORWARD:
                if col in g.columns:
                    row[col] = last[col]
            row["season"] = _season(fdate.month)
            row["paid_subs_bod"] = opening
            row["discount_pct"] = 0.0
            row["promo_flag"] = 0
            row["promo_name"] = None
            row["price_increase_flag"] = 0
            row["tentpole_flag"] = 0
            row["tentpole_name"] = None
            row["tentpole_type"] = "none"
            row["tentpole_intensity"] = 1.0
            row["holiday_flag"] = 0
            row["fiscal_week"] = int(fdate.isocalendar().week)
            row["fiscal_month"] = int(fdate.month)
            row["fiscal_quarter"] = int((fdate.month - 1) // 3 + 1)
            if "list_price" in row:
                row["effective_price"] = row["list_price"]
            rows.append(row)

    future = pd.DataFrame(rows)
    if calendar is not None and not calendar.empty and not future.empty:
        cal_cols = [c for c in _CALENDAR_COLUMNS if c in calendar.columns]
        cal = calendar[[date_col] + cal_cols]
        future = future.drop(columns=cal_cols).merge(cal, on=date_col, how="left")
        future["tentpole_flag"] = future["tentpole_flag"].fillna(0).astype(int)
        future["tentpole_type"] = future["tentpole_type"].fillna("none")
        future["tentpole_intensity"] = future["tentpole_intensity"].fillna(1.0)
        future["holiday_flag"] = future["holiday_flag"].fillna(0).astype(int)

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
    """Produce daily forward forecasts of the configured target.

    Baselines are re-fit on the *full* cleaned history; the ML model reuses the
    already-fitted estimator but is re-seeded with full history for recursion.
    """
    date_col = config["data"]["date_col"]
    dims = [c for c in SEGMENT_DIMS + ["tentpole_flag", "tentpole_name"] if c in future_df.columns]
    frames: List[pd.DataFrame] = []

    if best_model_name == MLForecaster.name:
        ml_model.set_history(cleaned_df)
        for sid, g in future_df.sort_values(date_col).groupby("series_id"):
            preds = ml_model.forecast_recursive(sid, g)
            frame = g[[date_col] + dims].copy()
            frame["series_id"] = sid
            frame["forecast_value"] = np.round(preds, 2)
            frames.append(frame)
    elif best_model_name == ChronosForecaster.name:
        model = ChronosForecaster(config)
        if not model.is_available():
            raise RuntimeError(
                "Chronos was selected as best model but is no longer available."
            )
        model.fit(cleaned_df)
        for sid, g in future_df.sort_values(date_col).groupby("series_id"):
            h = len(g)
            preds = model.predict(sid, h)
            frame = g[[date_col] + dims].copy()
            frame["series_id"] = sid
            frame["forecast_value"] = np.round(np.asarray(preds[:h]), 2)
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
            frame["forecast_value"] = np.round(np.asarray(preds[:h]), 2)
            frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    out["model"] = best_model_name
    out["horizon_day"] = out.groupby("series_id")[date_col].rank(method="first").astype(int)
    return out.sort_values(["series_id", date_col]).reset_index(drop=True)


def forecast_target(
    comparison: Dict[str, Any],
    cleaned_df: pd.DataFrame,
    config: Dict[str, Any],
    calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Daily forward forecast of ``config["data"]["target_col"]`` (long format)."""
    horizon = max(config["forecast"]["horizons"])
    future_df = build_future_frame(cleaned_df, config, horizon, calendar)
    return _forecast_with_best(
        comparison["best_model_name"],
        comparison["ml_model"],
        cleaned_df,
        future_df,
        config,
    )


def combine_target_forecasts(
    per_target: Dict[str, pd.DataFrame],
    cleaned_df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Merge per-target daily forecasts into one wide frame and derive KPIs.

    Output columns per series/day: ``forecast_<target>`` and ``model_<target>``
    for each target, plus ``opening_paid_subs``, ``forecast_net_adds``,
    ``forecast_paid_subs`` (end of day) and ``forecast_hours_per_paid_sub``.
    """
    date_col = config["data"]["date_col"]
    keys = ["series_id", date_col]
    wide: pd.DataFrame | None = None
    for target, daily in per_target.items():
        cols = daily.rename(
            columns={"forecast_value": f"forecast_{target}", "model": f"model_{target}"}
        )
        if wide is None:
            wide = cols
        else:
            wide = wide.merge(
                cols[keys + [f"forecast_{target}", f"model_{target}"]], on=keys, how="outer"
            )
    assert wide is not None, "combine_target_forecasts needs at least one target"
    wide = wide.sort_values(keys).reset_index(drop=True)
    wide["horizon_day"] = wide.groupby("series_id")[date_col].rank(method="first").astype(int)

    opening = (
        cleaned_df.sort_values(date_col).groupby("series_id").tail(1).set_index("series_id")
    )
    open_col = "paid_subs_eod" if "paid_subs_eod" in opening.columns else "paid_subs_bod"
    wide["opening_paid_subs"] = wide["series_id"].map(opening[open_col]).fillna(0).astype(float)

    adds = wide.get("forecast_gross_adds", pd.Series(0.0, index=wide.index)).fillna(0)
    churn = wide.get("forecast_churned_subs", pd.Series(0.0, index=wide.index)).fillna(0)
    wide["forecast_net_adds"] = np.round(adds - churn, 2)
    wide["forecast_paid_subs"] = np.round(
        wide["opening_paid_subs"] + wide.groupby("series_id")["forecast_net_adds"].cumsum(), 2
    )
    if "forecast_hours_watched" in wide.columns:
        wide["forecast_hours_per_paid_sub"] = np.round(
            safe_divide(wide["forecast_hours_watched"], wide["forecast_paid_subs"]), 4
        )
    return wide


def save_forecasts(daily: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Persist the wide daily forecast to ``outputs/forecasts.csv``."""
    path = resolve_path(config["paths"]["forecasts_csv"])
    path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(path, index=False)
    logger.info("Saved %d daily forecasts to %s", len(daily), path)


def horizon_rollups(
    daily_forecast: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    """Aggregate the wide daily forecast into one row per series per horizon.

    Flows (adds, churn, net adds, hours) are summed over the first ``h`` days;
    ``ending_paid_subs`` is the base at day ``h``; ``avg_paid_subs`` is the mean
    end-of-day base; ``hours_per_paid_sub_month`` normalizes usage to 30 days.
    """
    horizons = config["forecast"]["horizons"]
    dims = [c for c in SEGMENT_DIMS if c in daily_forecast.columns]
    flow_cols = [
        c
        for c in ["forecast_gross_adds", "forecast_churned_subs", "forecast_net_adds",
                  "forecast_hours_watched"]
        if c in daily_forecast.columns
    ]
    rows: List[pd.DataFrame] = []
    for h in horizons:
        sub = daily_forecast[daily_forecast["horizon_day"] <= h].sort_values(
            ["series_id", "horizon_day"]
        )
        grouped = sub.groupby(["series_id"] + dims, as_index=False)
        agg = grouped[flow_cols].sum()
        agg["opening_paid_subs"] = grouped["opening_paid_subs"].first()["opening_paid_subs"]
        agg["ending_paid_subs"] = grouped["forecast_paid_subs"].last()["forecast_paid_subs"]
        agg["avg_paid_subs"] = grouped["forecast_paid_subs"].mean()["forecast_paid_subs"]
        agg["tentpole_days"] = (
            grouped["tentpole_flag"].sum()["tentpole_flag"] if "tentpole_flag" in sub.columns else 0
        )
        if "forecast_hours_watched" in agg.columns:
            agg["hours_per_paid_sub_month"] = safe_divide(
                agg["forecast_hours_watched"], agg["avg_paid_subs"]
            ) * (30.0 / h)
        agg["forecast_horizon"] = h
        agg["forecast_start"] = sub[config["data"]["date_col"]].min()
        agg["forecast_end"] = sub[config["data"]["date_col"]].max()
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)
