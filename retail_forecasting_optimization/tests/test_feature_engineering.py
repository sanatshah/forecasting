"""Tests for feature engineering (correctness and leakage safety)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_validation import validate_and_clean
from src.feature_engineering import build_features, get_feature_columns


def _featured(sample_df, config):
    cleaned, _ = validate_and_clean(sample_df, config)
    return build_features(cleaned, config)


def test_lag_and_rolling_columns_created(sample_df, config):
    """All configured lag/rolling feature columns should exist."""
    feats = _featured(sample_df, config)
    for lag in config["features"]["lags"]:
        assert f"lag_{lag}" in feats.columns
    for w in config["features"]["rolling_windows"]:
        assert f"roll_mean_{w}" in feats.columns
        assert f"roll_std_{w}" in feats.columns


def test_lag_is_past_value_no_leakage(sample_df, config):
    """lag_1 at row t must equal the target at row t-1 within a series."""
    feats = _featured(sample_df, config).sort_values(["series_id", "date"])
    for _, g in feats.groupby("series_id"):
        g = g.reset_index(drop=True)
        # First row lag is NaN; subsequent rows equal the previous actual.
        expected = g["units_sold"].shift(1)
        got = g["lag_1"]
        mask = expected.notna()
        assert np.allclose(got[mask].to_numpy(), expected[mask].to_numpy())


def test_rolling_excludes_current_day(sample_df, config):
    """roll_mean_7 at row t must not include the current day's target.

    We verify by confirming the rolling mean equals the mean of the prior
    up-to-7 actuals (shifted), i.e. it is computed on shifted history.
    """
    feats = _featured(sample_df, config).sort_values(["series_id", "date"])
    for _, g in feats.groupby("series_id"):
        g = g.reset_index(drop=True)
        manual = g["units_sold"].shift(1).rolling(7, min_periods=1).mean()
        got = g["roll_mean_7"]
        mask = manual.notna()
        assert np.allclose(got[mask].to_numpy(), manual[mask].to_numpy())


def test_promo_uplift_ratio_uses_only_prior_demand(config):
    """Promo uplift at row t must not change when row t demand changes."""
    frame = pd.DataFrame(
        {
            "series_id": ["series_a"] * 5,
            "date": pd.date_range("2025-01-01", periods=5, freq="D"),
            "units_sold": [10.0, 20.0, 12.0, 30.0, 999.0],
            "promo_flag": [0, 1, 0, 1, 0],
        }
    )

    feats = build_features(frame, config)
    changed = frame.copy()
    changed.loc[4, "units_sold"] = 1.0
    changed_feats = build_features(changed, config)

    assert np.isclose(feats.loc[2, "promo_uplift_ratio"], 2.0)
    assert np.isclose(feats.loc[4, "promo_uplift_ratio"], 25.0 / 11.0)
    assert np.isclose(
        feats.loc[4, "promo_uplift_ratio"],
        changed_feats.loc[4, "promo_uplift_ratio"],
    )


def test_promo_uplift_ratio_defaults_to_one(config):
    """Missing promo history and a zero baseline should produce neutral lift."""
    frame = pd.DataFrame(
        {
            "series_id": ["no_promo"] * 3 + ["zero_baseline"] * 3,
            "date": list(pd.date_range("2025-01-01", periods=3, freq="D")) * 2,
            "units_sold": [5.0, 6.0, 7.0, 0.0, 10.0, 5.0],
            "promo_flag": [0, 0, 0, 0, 1, 0],
        }
    )

    feats = build_features(frame, config)

    assert np.allclose(
        feats.loc[feats["series_id"] == "no_promo", "promo_uplift_ratio"],
        1.0,
    )
    zero_baseline_last = feats.loc[
        feats["series_id"] == "zero_baseline", "promo_uplift_ratio"
    ].iloc[-1]
    assert zero_baseline_last == 1.0
    assert np.isfinite(feats["promo_uplift_ratio"]).all()


def test_price_and_calendar_features(sample_df, config):
    """Price and calendar derived features should be present and sane."""
    feats = _featured(sample_df, config)
    assert "discount_depth" in feats.columns
    assert "weekend_flag" in feats.columns
    assert feats["weekend_flag"].isin([0, 1]).all()
    # No markdown in sample -> discount depth should be ~0.
    assert np.allclose(feats["discount_depth"].fillna(0), 0.0)


def test_weeks_of_supply_present(sample_df, config):
    """Inventory feature weeks_of_supply should be computed and non-negative."""
    feats = _featured(sample_df, config)
    assert "weeks_of_supply" in feats.columns
    assert (feats["weeks_of_supply"] >= 0).all()


def test_feature_columns_subset_of_frame(sample_df, config):
    """get_feature_columns must only return columns present in the frame."""
    feats = _featured(sample_df, config)
    cols = get_feature_columns(feats, config)
    assert len(cols) > 0
    assert "promo_uplift_ratio" in cols
    assert set(cols).issubset(set(feats.columns))
