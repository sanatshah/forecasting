"""Subscriber-forecasting, leakage-safe time-series feature engineering.

Design principles
-----------------
* **No leakage**: lag/rolling features are computed per series and reference
  only *past* observations (``shift(1)`` before rolling). Price, promo,
  content-calendar and calendar features are "known covariates" - values known
  in advance for future dates - so they may use the current row.
* **Target-agnostic**: the same functions serve every forecast target
  (``gross_adds``, ``churned_subs``, ``hours_watched``); the target column is
  read from ``config["data"]["target_col"]``.
* **Reusable for train and inference**: :func:`build_features` is applied to
  history and to future rows; the recursive forecaster in :mod:`src.model_ml`
  calls it repeatedly as it rolls forward.
* **Config-driven windows**: lag, rolling and event windows come from
  ``config.yaml``.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .utils import get_logger, safe_divide

logger = get_logger(__name__)


# Categorical segment columns used by the ML model (encoded downstream).
HIERARCHY_COLUMNS = [
    "tier",
    "acquisition_channel",
    "distribution_partner",
    "season",
    "tentpole_type",
]

_NO_EVENT = 9999


def _add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add day-of-week, week-of-year, month, quarter and weekend flag."""
    d = df[date_col].dt
    df["dow"] = d.dayofweek
    df["week_of_year"] = d.isocalendar().week.astype(int)
    df["month"] = d.month
    df["quarter"] = d.quarter
    df["day_of_month"] = d.day
    df["weekend_flag"] = (d.dayofweek >= 5).astype(int)
    return df


def _days_since_flag(flags: np.ndarray) -> np.ndarray:
    """Days since the most recent flagged day (inclusive); sentinel if none."""
    n = len(flags)
    since = np.full(n, _NO_EVENT, dtype="int64")
    last = -1
    for i in range(n):
        if flags[i] == 1:
            last = i
        since[i] = (i - last) if last >= 0 else _NO_EVENT
    return since


def _days_until_flag(flags: np.ndarray) -> np.ndarray:
    """Days until the next flagged day (inclusive); sentinel if none."""
    n = len(flags)
    until = np.full(n, _NO_EVENT, dtype="int64")
    nxt = -1
    for i in range(n - 1, -1, -1):
        if flags[i] == 1:
            nxt = i
        until[i] = (nxt - i) if nxt >= 0 else _NO_EVENT
    return until


def _add_price_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add discount depth, price gap, list-price index and price-change timing.

    ``price_index_vs_hist`` compares the current list price to the *expanding*
    historical mean list price of the series (shifted by 1 so the current day
    is excluded). ``days_since_price_increase`` uses the planned price-change
    calendar, which is known in advance.
    """
    if {"list_price", "effective_price"}.issubset(df.columns):
        df["price_gap"] = df["list_price"] - df["effective_price"]
        df["discount_depth"] = safe_divide(df["price_gap"], df["list_price"])

    if "list_price" in df.columns:
        hist_mean = (
            df.groupby("series_id")["list_price"]
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=0, drop=True)
        )
        hist_mean = hist_mean.fillna(df["list_price"])
        df["price_index_vs_hist"] = safe_divide(df["list_price"], hist_mean, fill=1.0)

    if "price_increase_flag" in df.columns:
        df = df.sort_values(["series_id", date_col])
        since = df.groupby("series_id")["price_increase_flag"].transform(
            lambda s: pd.Series(
                _days_since_flag(s.fillna(0).astype(int).to_numpy()), index=s.index
            )
        )
        df["days_since_price_increase"] = since.astype("int64")
    return df


def _add_event_features(
    df: pd.DataFrame,
    date_col: str,
    feat_cfg: Dict[str, Any],
    target: str,
    flag_col: str = "promo_flag",
    prefix: str = "promo",
) -> pd.DataFrame:
    """Add event timing and historical event-response features per series.

    Used for both promo offers (``promo_flag``) and content tentpoles
    (``tentpole_flag``). Produces ``days_since_last_<prefix>``,
    ``days_until_next_<prefix>`` and ``<prefix>_uplift_ratio``.

    ``days_until_next_*`` uses the future event calendar, which is planned in
    advance, so it is a known covariate rather than leakage.

    ``<prefix>_uplift_ratio`` compares mean target on prior event days to prior
    non-event days using ``shift(1)`` target so the current day's value never
    enters the ratio. The trailing window falls back to an expanding mean when
    it holds no event (or no non-event) days.
    """
    if flag_col not in df.columns:
        return df

    win = feat_cfg["event_window"]
    has_target = target in df.columns

    def _per_series(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(date_col)
        flags = g[flag_col].fillna(0).astype(int).to_numpy()
        g[f"days_since_last_{prefix}"] = _days_since_flag(flags)
        g[f"days_until_next_{prefix}"] = _days_until_flag(flags)

        if has_target:
            value = g[target].shift(1)
            past_flag = g[flag_col].fillna(0).astype(int).shift(1)
            event_value = value.where(past_flag == 1)
            base_value = value.where(past_flag != 1)

            event_trail = event_value.rolling(win, min_periods=1).mean()
            base_trail = base_value.rolling(win, min_periods=1).mean()
            event_exp = event_value.expanding(min_periods=1).mean()
            base_exp = base_value.expanding(min_periods=1).mean()

            event_mean = event_trail.where(event_trail.notna(), event_exp)
            base_mean = base_trail.where(base_trail.notna(), base_exp)
            ratio = safe_divide(event_mean.to_numpy(), base_mean.to_numpy(), fill=1.0)
            undefined = (event_mean.isna() | base_mean.isna()).to_numpy()
            ratio = np.where(undefined, 1.0, ratio)
            g[f"{prefix}_uplift_ratio"] = ratio

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


def _trailing_rate(
    df: pd.DataFrame, numerator: str, denominator: str, win: int
) -> pd.Series:
    """Per-series trailing ``mean(numerator) / mean(denominator)`` on past rows.

    The denominator is masked wherever the numerator is unknown so that future
    rows (numerator NaN during recursive forecasting) do not dilute the rate.
    """
    num = df.groupby("series_id")[numerator].shift(1)
    den = df.groupby("series_id")[denominator].shift(1).where(num.notna())
    num_mean = num.groupby(df["series_id"]).rolling(win, min_periods=1).mean().reset_index(
        level=0, drop=True
    )
    den_mean = den.groupby(df["series_id"]).rolling(win, min_periods=1).mean().reset_index(
        level=0, drop=True
    )
    rate = safe_divide(num_mean.fillna(0).to_numpy(), den_mean.fillna(0).to_numpy(), fill=0.0)
    return pd.Series(rate, index=df.index)


def _add_base_features(df: pd.DataFrame, feat_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Add subscriber-base features: trailing churn rate and hours per sub.

    ``paid_subs_bod`` (the base at the start of the day) is itself a model
    input - churn and usage scale with it the way retail sales scale with
    inventory on hand. The trailing rates use only prior days.
    """
    if "paid_subs_bod" not in df.columns:
        return df
    win = feat_cfg["event_window"]
    if "churned_subs" in df.columns:
        df["trailing_churn_rate"] = _trailing_rate(df, "churned_subs", "paid_subs_bod", win)
    if "hours_watched" in df.columns:
        df["trailing_hours_per_sub"] = _trailing_rate(df, "hours_watched", "paid_subs_bod", win)
    return df


def _add_age_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add segment age in days (days since the series' first observed date)."""
    first_seen = df.groupby("series_id")[date_col].transform("min")
    df["series_age_days"] = (df[date_col] - first_seen).dt.days
    return df


def build_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Build the full feature set on an already-cleaned DataFrame.

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
    df = _add_price_features(df, date_col)
    df = _add_event_features(df, date_col, feat_cfg, target, "promo_flag", "promo")
    df = _add_event_features(df, date_col, feat_cfg, target, "tentpole_flag", "tentpole")
    df = df.sort_values(["series_id", date_col]).reset_index(drop=True)
    df = _add_lag_features(df, target, feat_cfg["lags"])
    df = _add_rolling_features(df, target, feat_cfg["rolling_windows"])
    df = _add_base_features(df, feat_cfg)
    df = _add_age_features(df, date_col)

    df = df.sort_values(["series_id", date_col]).reset_index(drop=True)
    logger.debug("Feature engineering produced %d columns", df.shape[1])
    return df


def get_feature_columns(df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    """Return the ordered list of model input columns present in ``df``.

    Combines engineered numeric features, known covariates, and segment
    categoricals. Only columns actually present are returned so the same call
    works on partial frames during inference. Same-day outcomes
    (``paid_subs_eod``, ``daily_active_subs``, other targets) are excluded.
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
        "discount_pct",
        "list_price",
        "effective_price",
        "price_increase_flag",
        "days_since_price_increase",
        "promo_flag",
        "days_since_last_promo",
        "days_until_next_promo",
        "promo_uplift_ratio",
        "tentpole_flag",
        "tentpole_intensity",
        "days_since_last_tentpole",
        "days_until_next_tentpole",
        "tentpole_uplift_ratio",
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
        "paid_subs_bod",
        "trailing_churn_rate",
        "trailing_hours_per_sub",
        "series_age_days",
    ]
    categorical = [c for c in HIERARCHY_COLUMNS if c in df.columns]
    cols = [c for c in (numeric + categorical) if c in df.columns]
    seen: set = set()
    ordered = [c for c in cols if not (c in seen or seen.add(c))]
    return ordered
