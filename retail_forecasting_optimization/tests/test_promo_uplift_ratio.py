"""Isolation tests for promo_uplift_ratio in _add_promo_features.

These tests call _add_promo_features directly with minimal frames so failures
pinpoint promo uplift logic rather than validation, lags, or rolling features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import _add_promo_features


def _promo_frame(
    promo_flags: list[int],
    units_sold: list[int] | None = None,
    *,
    series_id: str = "S1",
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Minimal frame with only columns _add_promo_features needs."""
    n = len(promo_flags)
    dates = pd.date_range(start, periods=n, freq="D")
    data: dict = {
        "series_id": series_id,
        "date": dates,
        "promo_flag": promo_flags,
    }
    if units_sold is not None:
        data["units_sold"] = units_sold
    return pd.DataFrame(data)


def _apply_promo_features(df: pd.DataFrame, config, date_col: str = "date") -> pd.DataFrame:
    return _add_promo_features(df.copy(), date_col, config["features"])


def _expected_promo_uplift(
    promo_flags: list[int],
    units_sold: list[int],
    win: int = 28,
) -> np.ndarray:
    """Reference implementation mirroring _add_promo_features uplift logic."""
    demand = pd.Series(units_sold, dtype="float64").shift(1)
    past_promo = pd.Series(promo_flags).fillna(0).astype(int).shift(1)
    promo_demand = demand.where(past_promo == 1)
    non_promo_demand = demand.where(past_promo != 1)

    promo_trail = promo_demand.rolling(win, min_periods=1).mean()
    non_promo_trail = non_promo_demand.rolling(win, min_periods=1).mean()
    promo_exp = promo_demand.expanding(min_periods=1).mean()
    non_promo_exp = non_promo_demand.expanding(min_periods=1).mean()

    promo_mean = promo_trail.where(promo_trail.notna(), promo_exp)
    non_promo_mean = non_promo_trail.where(non_promo_trail.notna(), non_promo_exp)

    ratio = np.full(len(units_sold), 1.0, dtype="float64")
    for i in range(len(units_sold)):
        p = promo_mean.iloc[i]
        n = non_promo_mean.iloc[i]
        if pd.isna(p):
            ratio[i] = 1.0
        elif pd.isna(n) or n == 0:
            ratio[i] = 1.0
        else:
            ratio[i] = p / n
    return ratio


@pytest.fixture
def feat_cfg(config):
    return config["features"]


def test_skips_uplift_when_promo_flag_missing(feat_cfg):
    """Without promo_flag the helper should be a no-op."""
    df = _promo_frame([0, 0], units_sold=[10, 10]).drop(columns=["promo_flag"])
    out = _apply_promo_features(df, {"features": feat_cfg})
    assert "promo_uplift_ratio" not in out.columns
    assert "days_since_last_promo" not in out.columns


def test_skips_uplift_when_units_sold_missing(feat_cfg):
    """Timing features should still be added when demand is absent."""
    df = _promo_frame([0, 1, 0])
    out = _apply_promo_features(df, {"features": feat_cfg})
    assert "days_since_last_promo" in out.columns
    assert "days_until_next_promo" in out.columns
    assert "promo_uplift_ratio" not in out.columns


def test_first_row_defaults_to_one(feat_cfg):
    """No shifted history on the first row -> ratio 1.0."""
    df = _promo_frame([1, 0, 0], units_sold=[30, 10, 10])
    out = _apply_promo_features(df, {"features": feat_cfg}).sort_values("date")
    assert out.iloc[0]["promo_uplift_ratio"] == 1.0


def test_defaults_to_one_without_promo_history(feat_cfg):
    """All-non-promo series should stay at the neutral default."""
    df = _promo_frame([0, 0, 0, 0], units_sold=[5, 6, 7, 8])
    out = _apply_promo_features(df, {"features": feat_cfg}).sort_values("date")
    assert np.allclose(out["promo_uplift_ratio"].to_numpy(), 1.0)


def test_defaults_to_one_when_non_promo_baseline_is_zero(feat_cfg):
    """Zero non-promo demand should not produce inf or NaN."""
    df = _promo_frame([0, 0, 1, 0], units_sold=[0, 0, 20, 10])
    out = _apply_promo_features(df, {"features": feat_cfg}).sort_values("date")
    assert out.iloc[3]["promo_uplift_ratio"] == 1.0


def test_excludes_current_day_demand(feat_cfg):
    """Mutating today's units_sold must not change today's ratio."""
    df = _promo_frame([0, 0, 1, 0, 0], units_sold=[10, 10, 20, 10, 10])
    out = _apply_promo_features(df, {"features": feat_cfg}).sort_values("date")
    baseline = out.iloc[4]["promo_uplift_ratio"]

    mutated = df.copy()
    mutated.loc[4, "units_sold"] = 999
    mutated_out = _apply_promo_features(mutated, {"features": feat_cfg}).sort_values("date")
    assert mutated_out.iloc[4]["promo_uplift_ratio"] == baseline


def test_matches_reference_implementation(feat_cfg):
    """Full-series output should match an independent reference calculator."""
    promo = [0, 0, 1, 0, 0, 1, 0]
    units = [10, 10, 20, 10, 10, 30, 10]
    df = _promo_frame(promo, units_sold=units)
    out = _apply_promo_features(df, {"features": feat_cfg}).sort_values("date")
    expected = _expected_promo_uplift(promo, units, win=feat_cfg["wos_demand_window"])
    assert np.allclose(out["promo_uplift_ratio"].to_numpy(), expected)


def test_known_ratio_on_prior_promo_day(feat_cfg):
    """Hand-checked row: promo demand 20 vs non-promo mean 10 -> ratio 2.0."""
    df = _promo_frame([0, 0, 1, 0, 0], units_sold=[10, 10, 20, 10, 10])
    out = _apply_promo_features(df, {"features": feat_cfg}).sort_values("date")
    assert np.isclose(out.iloc[3]["promo_uplift_ratio"], 2.0)


def test_falls_back_to_expanding_when_trailing_window_is_sparse(feat_cfg):
    """Promo outside the trailing window should still contribute via expanding."""
    win = feat_cfg["wos_demand_window"]
    n = win + 5
    promo = [1] + [0] * (n - 1)
    units = [30] + [10] * (n - 1)
    df = _promo_frame(promo, units_sold=units)
    out = _apply_promo_features(df, {"features": feat_cfg}).sort_values("date")

    # Trailing window no longer sees the lone promo day; expanding still does.
    assert np.isclose(out.iloc[-1]["promo_uplift_ratio"], 3.0)


def test_computed_independently_per_series(feat_cfg):
    """Each series should use only its own shifted history."""
    df_a = _promo_frame([0, 0, 1, 0], units_sold=[10, 10, 20, 10], series_id="A")
    df_b = _promo_frame([0, 0, 0, 0], units_sold=[5, 5, 5, 5], series_id="B")
    df = pd.concat([df_a, df_b], ignore_index=True)
    out = _apply_promo_features(df, {"features": feat_cfg})

    series_a = out[out["series_id"] == "A"].sort_values("date")
    series_b = out[out["series_id"] == "B"].sort_values("date")
    assert np.isclose(series_a.iloc[3]["promo_uplift_ratio"], 2.0)
    assert np.allclose(series_b["promo_uplift_ratio"].to_numpy(), 1.0)
