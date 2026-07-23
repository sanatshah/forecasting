"""Retail-specific, leakage-safe time-series feature engineering.

Design principles
-----------------
* **No leakage**: lag/rolling features are computed per series and reference
  only *past* observations (``shift(1)`` before rolling). Price/promo/calendar
  features are "known covariates" - values known in advance for future dates -
  so they may use the current row.
* **Reusable for train and inference**: the same :func:`build_features`
  function is applied to history and to future rows; the recursive forecaster
  in :mod:`src.model_ml` calls it repeatedly as it rolls forward.
* **Config-driven windows**: lag and rolling windows come from ``config.yaml``.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .utils import get_logger, safe_divide

logger = get_logger(__name__)


# Categorical hierarchy columns used by the ML model (encoded downstream).
HIERARCHY_COLUMNS = [
    "department",
    "class",
    "subclass",
    "location_id",
    "channel",
    "product_lifecycle_status",
    "season",
]


def _add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add day-of-week, week-of-year, month, quarter and weekend flag.

    Fiscal columns from the source data are passed through untouched by the
    caller; here we add the *civil* calendar decomposition.
    """
    d = df[date_col].dt
    df["dow"] = d.dayofweek
    df["week_of_year"] = d.isocalendar().week.astype(int)
    df["month"] = d.month
    df["quarter"] = d.quarter
    df["day_of_month"] = d.day
    df["weekend_flag"] = (d.dayofweek >= 5).astype(int)
    return df


def _add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add discount depth, price gap, and price index vs series history.

    ``price_index_vs_hist`` compares the current selling price to the
    *expanding* historical mean selling price of the series (shifted by 1 so
    the current day is excluded), which avoids leakage.
    """
    if {"regular_price", "selling_price"}.issubset(df.columns):
        df["price_gap"] = df["regular_price"] - df["selling_price"]
        df["discount_depth"] = safe_divide(df["price_gap"], df["regular_price"])
    if "markdown_pct" not in df.columns and "discount_depth" in df.columns:
        df["markdown_pct"] = df["discount_depth"]

    if "selling_price" in df.columns:
        hist_mean = (
            df.groupby("series_id")["selling_price"]
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=0, drop=True)
        )
        # Fall back to the current price for the first observation.
        hist_mean = hist_mean.fillna(df["selling_price"])
        df["price_index_vs_hist"] = safe_divide(df["selling_price"], hist_mean, fill=1.0)
    return df


def _add_promo_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add days-since-last-promo and days-until-next-promo per series.

    ``days_until_next_promo`` uses future promo *calendar* information, which is
    legitimately known in advance (promotions are planned), so it is treated as
    a known covariate rather than leakage.
    """
    if "promo_flag" not in df.columns:
        return df

    def _per_series(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(date_col)
        promo = g["promo_flag"].fillna(0).astype(int).to_numpy()
        n = len(promo)

        # Days since last promo (walk forward).
        since = np.full(n, 9999, dtype="int64")
        last = -1
        for i in range(n):
            if promo[i] == 1:
                last = i
            since[i] = (i - last) if last >= 0 else 9999
        # Days until next promo (walk backward).
        until = np.full(n, 9999, dtype="int64")
        nxt = -1
        for i in range(n - 1, -1, -1):
            if promo[i] == 1:
                nxt = i
            until[i] = (nxt - i) if nxt >= 0 else 9999
        g["days_since_last_promo"] = since
        g["days_until_next_promo"] = until
        return g

    df = (
        df.groupby("series_id", group_keys=False)[df.columns.tolist()]
        .apply(_per_series)
    )
    return df


def _add_lag_features(
    df: pd.DataFrame, target: str, lags: List[int]
) -> pd.DataFrame:
    """Add shifted target lags per series (strictly past values)."""
    grp = df.groupby("series_id")[target]
    for lag in lags:
        df[f"lag_{lag}"] = grp.shift(lag)
    return df


def _add_rolling_features(
    df: pd.DataFrame, target: str, windows: List[int]
) -> pd.DataFrame:
    """Add rolling mean/std of the target per series.

    We ``shift(1)`` before rolling so window statistics exclude the current
    day - this is what prevents target leakage.
    """
    grp = df.groupby("series_id")[target]
    shifted = grp.shift(1)
    for w in windows:
        df[f"roll_mean_{w}"] = (
            shifted.groupby(df["series_id"]).rolling(w, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
        df[f"roll_std_{w}"] = (
            shifted.groupby(df["series_id"]).rolling(w, min_periods=1).std()
            .reset_index(level=0, drop=True)
        )
    return df


def _add_inventory_features(
    df: pd.DataFrame, feat_cfg: Dict[str, Any]
) -> pd.DataFrame:
    """Add weeks-of-supply and a constrained-demand flag.

    Weeks of supply = inventory_on_hand / (avg daily demand * 7), using a
    trailing demand rate (shifted, so no leakage). ``constrained_demand_flag``
    marks days where a stockout occurred or supply is critically low, because
    observed sales on those days understate true demand.
    """
    target = "units_sold"
    win = feat_cfg["wos_demand_window"]

    if "units_sold" in df.columns:
        demand_rate = (
            df.groupby("series_id")[target]
            .apply(lambda s: s.shift(1).rolling(win, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        df["avg_daily_demand"] = demand_rate.fillna(0)
    else:
        df["avg_daily_demand"] = 0.0

    if "inventory_on_hand" in df.columns:
        weekly_demand = df["avg_daily_demand"] * 7.0
        df["weeks_of_supply"] = safe_divide(
            df["inventory_on_hand"], weekly_demand, fill=99.0
        )
    else:
        df["weeks_of_supply"] = 99.0

    thr_days = feat_cfg["constrained_wos_threshold_days"]
    low_supply = df["inventory_on_hand"] <= (df["avg_daily_demand"] * thr_days) if (
        "inventory_on_hand" in df.columns
    ) else pd.Series(False, index=df.index)
    stockout = df.get("derived_stockout_flag", df.get("stockout_flag", 0))
    df["constrained_demand_flag"] = (
        (pd.Series(stockout, index=df.index).fillna(0).astype(int) == 1) | low_supply
    ).astype(int)
    return df


def _add_lifecycle_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add product age in days (days since the series' first observed date)."""
    first_seen = df.groupby("series_id")[date_col].transform("min")
    df["product_age_days"] = (df[date_col] - first_seen).dt.days
    return df


def build_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Build the full retail feature set on an already-cleaned DataFrame.

    Expects ``series_id`` to be present (see
    :func:`src.data_loader.add_series_id`). Returns a new DataFrame sorted by
    series and date with all engineered columns added.
    """
    data_cfg = config["data"]
    feat_cfg = config["features"]
    date_col = data_cfg["date_col"]
    target = data_cfg["target_col"]

    if "series_id" not in df.columns:
        raise ValueError("build_features requires a 'series_id' column.")

    df = df.sort_values(["series_id", date_col]).reset_index(drop=True).copy()

    df = _add_calendar_features(df, date_col)
    df = _add_price_features(df)
    df = _add_promo_features(df, date_col)
    df = _add_lag_features(df, target, feat_cfg["lags"])
    df = _add_rolling_features(df, target, feat_cfg["rolling_windows"])
    df = _add_inventory_features(df, feat_cfg)
    df = _add_lifecycle_features(df, date_col)

    df = df.sort_values(["series_id", date_col]).reset_index(drop=True)
    logger.info("Feature engineering produced %d columns", df.shape[1])
    return df


def get_feature_columns(df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    """Return the ordered list of model input columns present in ``df``.

    Combines engineered numeric features, known covariates, and hierarchy
    categoricals. Only columns actually present are returned so the same call
    works on partial frames during inference.
    """
    feat_cfg = config["features"]
    numeric: List[str] = []
    numeric += [f"lag_{l}" for l in feat_cfg["lags"]]
    for w in feat_cfg["rolling_windows"]:
        numeric += [f"roll_mean_{w}", f"roll_std_{w}"]
    numeric += [
        "discount_depth",
        "price_gap",
        "price_index_vs_hist",
        "markdown_pct",
        "regular_price",
        "selling_price",
        "promo_flag",
        "days_since_last_promo",
        "days_until_next_promo",
        "holiday_flag",
        "dow",
        "week_of_year",
        "month",
        "quarter",
        "day_of_month",
        "weekend_flag",
        "fiscal_week",
        "fiscal_month",
        "fiscal_quarter",
        "inventory_on_hand",
        "inventory_in_transit",
        "weeks_of_supply",
        "avg_daily_demand",
        "constrained_demand_flag",
        "product_age_days",
    ]
    categorical = [c for c in HIERARCHY_COLUMNS if c in df.columns]
    cols = [c for c in (numeric + categorical) if c in df.columns]
    # De-duplicate preserving order.
    seen: set = set()
    ordered = [c for c in cols if not (c in seen or seen.add(c))]
    return ordered
