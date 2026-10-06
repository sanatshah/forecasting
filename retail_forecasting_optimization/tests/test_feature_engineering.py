"""Tests for feature engineering (correctness and leakage safety)."""
from __future__ import annotations

import numpy as np

from src.data_validation import validate_and_clean
from src.feature_engineering import build_features, get_feature_columns
from src.utils import config_for_target


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
        expected = g["gross_adds"].shift(1)
        got = g["lag_1"]
        mask = expected.notna()
        assert np.allclose(got[mask].to_numpy(), expected[mask].to_numpy())


def test_rolling_excludes_current_day(sample_df, config):
    """roll_mean_7 at row t must be the mean of the prior up-to-7 actuals."""
    feats = _featured(sample_df, config).sort_values(["series_id", "date"])
    for _, g in feats.groupby("series_id"):
        g = g.reset_index(drop=True)
        manual = g["gross_adds"].shift(1).rolling(7, min_periods=1).mean()
        got = g["roll_mean_7"]
        mask = manual.notna()
        assert np.allclose(got[mask].to_numpy(), manual[mask].to_numpy())


def test_lags_follow_configured_target(sample_df, config):
    """Switching the target re-points lag features at that target."""
    cleaned, _ = validate_and_clean(sample_df, config)
    feats = build_features(cleaned, config_for_target(config, "churned_subs"))
    g = feats[feats["series_id"] == feats["series_id"].iloc[0]].reset_index(drop=True)
    expected = g["churned_subs"].shift(1)
    mask = expected.notna()
    assert np.allclose(g["lag_1"][mask].to_numpy(), expected[mask].to_numpy())


def test_price_and_calendar_features(sample_df, config):
    """Price and calendar derived features should be present and sane."""
    feats = _featured(sample_df, config)
    assert "discount_depth" in feats.columns
    assert feats["weekend_flag"].isin([0, 1]).all()
    promo = feats["promo_flag"] == 1
    assert np.allclose(feats.loc[~promo, "discount_depth"], 0.0)
    assert (feats.loc[promo, "discount_depth"] > 0).all()


def test_base_features_present(sample_df, config):
    """Trailing churn rate and hours per sub are computed and non-negative."""
    feats = _featured(sample_df, config)
    for col in ["trailing_churn_rate", "trailing_hours_per_sub", "series_age_days"]:
        assert col in feats.columns
        assert (feats[col] >= 0).all()


def test_feature_columns_subset_of_frame(sample_df, config):
    """get_feature_columns must only return columns present in the frame."""
    feats = _featured(sample_df, config)
    cols = get_feature_columns(feats, config)
    assert set(cols).issubset(set(feats.columns))
    for col in [
        "promo_uplift_ratio",
        "tentpole_uplift_ratio",
        "days_until_next_tentpole",
        "trailing_churn_rate",
        "paid_subs_bod",
    ]:
        assert col in cols


def test_same_day_outcomes_excluded_from_features(sample_df, config):
    """Same-day outcomes and sibling targets must never be model inputs."""
    feats = _featured(sample_df, config)
    cols = get_feature_columns(feats, config)
    for col in ["paid_subs_eod", "daily_active_subs", *config["data"]["targets"]]:
        assert col not in cols
