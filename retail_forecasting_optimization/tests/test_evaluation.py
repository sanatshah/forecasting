"""Unit tests for evaluation metrics, holdout split, and eval protocol guards."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval_baseline import (
    COVARIATE_PROTOCOL_ACTUAL_HOLDOUT,
    COVARIATE_PROTOCOL_PRODUCTION,
    FROZEN_BASELINE_PATH,
    assess_baseline_comparison_validity,
    compare_to_frozen_baseline,
    load_frozen_baseline,
)
from src.evaluation import (
    bias,
    compute_metrics,
    full_evaluation,
    mae,
    mape,
    rmse,
    wape,
)
from src.forecasting_pipeline import build_future_frame
from src.model_ml import MLForecaster
from src.model_selection import _ml_predictions, time_based_cutoff


# ---------------------------------------------------------------------------
# Metric formula tests
# ---------------------------------------------------------------------------


def test_wape_formula():
    actual = np.array([10.0, 20.0, 30.0])
    forecast = np.array([12.0, 18.0, 33.0])
    # sum|A-F| = 2 + 2 + 3 = 7; sum|A| = 60
    assert wape(actual, forecast) == pytest.approx(7.0 / 60.0)


def test_wape_nan_when_sum_actual_zero():
    actual = np.array([0.0, 0.0])
    forecast = np.array([1.0, 2.0])
    assert np.isnan(wape(actual, forecast))


def test_mape_excludes_zero_actual():
    actual = np.array([0.0, 10.0, 20.0])
    forecast = np.array([99.0, 11.0, 18.0])
    # Only rows with actual != 0: |1/10| + |2/20| over 2 rows = 0.1
    assert mape(actual, forecast) == pytest.approx(0.1)


def test_mape_nan_when_all_actual_zero():
    actual = np.array([0.0, 0.0])
    forecast = np.array([1.0, 2.0])
    assert np.isnan(mape(actual, forecast))


def test_mae_formula():
    actual = np.array([10.0, 20.0])
    forecast = np.array([13.0, 16.0])
    assert mae(actual, forecast) == pytest.approx(3.5)


def test_rmse_formula():
    actual = np.array([10.0, 20.0])
    forecast = np.array([13.0, 16.0])
    assert rmse(actual, forecast) == pytest.approx(np.sqrt((9 + 16) / 2))


def test_bias_formula_positive_over_forecast():
    actual = np.array([10.0, 20.0])
    forecast = np.array([12.0, 22.0])
    assert bias(actual, forecast) == pytest.approx(2.0)


def test_compute_metrics_bundle():
    actual = np.array([10.0, 20.0, 30.0])
    forecast = np.array([12.0, 18.0, 33.0])
    m = compute_metrics(actual, forecast)
    assert m["wape"] == pytest.approx(7.0 / 60.0)
    assert m["n"] == 3
    assert m["forecast_accuracy"] == pytest.approx(1 - 7.0 / 60.0)


# ---------------------------------------------------------------------------
# Holdout split tests
# ---------------------------------------------------------------------------


def test_time_based_cutoff_holdout_days_28(sample_df, config):
    date_col = config["data"]["date_col"]
    holdout_days = config["forecast"]["holdout_days"]
    assert holdout_days == 28

    cutoff = time_based_cutoff(sample_df, date_col, holdout_days)
    max_date = sample_df[date_col].max()
    assert cutoff == max_date - pd.Timedelta(days=28)

    train = sample_df[sample_df[date_col] <= cutoff]
    holdout = sample_df[sample_df[date_col] > cutoff]
    assert len(train) + len(holdout) == len(sample_df)
    assert train[date_col].max() <= cutoff
    assert holdout[date_col].min() > cutoff


# ---------------------------------------------------------------------------
# full_evaluation slice presence
# ---------------------------------------------------------------------------


def _prediction_frame(config) -> pd.DataFrame:
    """Minimal holdout-style frame with all slice dimensions."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-03-01", periods=4, freq="D"),
            "department": ["Womens Apparel", "Home", "Womens Apparel", "Home"],
            "channel": ["store", "online", "store", "online"],
            "location_id": ["LOC01", "LOC02", "LOC01", "LOC02"],
            "sku_id": ["SKU9001", "SKU9002", "SKU9001", "SKU9002"],
            "product_lifecycle_status": ["Core", "New", "Core", "New"],
            "promo_flag": [0, 1, 0, 1],
            "actual": [10.0, 20.0, 15.0, 25.0],
            "forecast": [11.0, 19.0, 14.0, 26.0],
        }
    )


def test_full_evaluation_includes_required_slices(config):
    ev = full_evaluation(_prediction_frame(config))
    levels = set(ev["level"])
    assert "overall" in levels
    assert "department" in levels
    assert "channel" in levels
    assert "location_id" in levels
    assert "sku_id" in levels
    assert "product_lifecycle_status" in levels
    assert "promo_period" in levels

    promo_groups = set(ev[ev["level"] == "promo_period"]["group"])
    assert promo_groups == {"promo", "non_promo"}


# ---------------------------------------------------------------------------
# Holdout ML vs production covariate protocol (CI guard)
# ---------------------------------------------------------------------------


def test_holdout_ml_uses_actual_holdout_covariates_not_production_defaults(
    sample_df, config
):
    """Holdout ML must use realized holdout covariates; production zeros promo/holiday.

    A silent switch to production-like covariates during holdout backtest must fail CI.
    """
    cleaned = sample_df.copy()
    date_col = config["data"]["date_col"]
    target = config["data"]["target_col"]

    cutoff = time_based_cutoff(cleaned, date_col, config["forecast"]["holdout_days"])
    holdout = cleaned[cleaned[date_col] > cutoff].copy()

    # Force distinct holdout covariates so the mismatch is observable.
    holdout.loc[holdout.index[0], "promo_flag"] = 1
    holdout.loc[holdout.index[0], "holiday_flag"] = 1
    holdout.loc[holdout.index[0], "selling_price"] = 99.0
    holdout.loc[holdout.index[0], "inventory_on_hand"] = 42

    sid = holdout.iloc[0]["series_id"]
    holdout_row = holdout[holdout["series_id"] == sid].sort_values(date_col).iloc[0]

    # Holdout ML path passes raw holdout rows (minus target) into forecast_recursive.
    future_df = holdout[holdout["series_id"] == sid].drop(columns=[target])
    assert int(future_df.iloc[0]["promo_flag"]) == 1
    assert int(future_df.iloc[0]["holiday_flag"]) == 1
    assert float(future_df.iloc[0]["selling_price"]) == 99.0
    assert float(future_df.iloc[0]["inventory_on_hand"]) == 42.0

    # Production forward path zeros promo/holiday and carry-forwards last observed values.
    prod_future = build_future_frame(cleaned, config, horizon=1)
    prod_row = prod_future[prod_future["series_id"] == sid].iloc[0]
    last = cleaned[cleaned["series_id"] == sid].sort_values(date_col).iloc[-1]

    assert int(prod_row["promo_flag"]) == 0
    assert int(prod_row["holiday_flag"]) == 0
    assert float(prod_row["selling_price"]) == float(last["selling_price"])
    assert float(prod_row["inventory_on_hand"]) == float(last["inventory_on_hand"])

    # Document the protocol labels enforced by the harness.
    assert holdout_row["promo_flag"] != prod_row["promo_flag"] or holdout_row[
        "holiday_flag"
    ] != prod_row["holiday_flag"]


def test_ml_predictions_covariate_protocol_constant(sample_df, config, monkeypatch):
    """_ml_predictions must remain on the actual-holdout-covariate protocol."""
    from src.data_validation import validate_and_clean
    from src.feature_engineering import build_features

    cleaned, _ = validate_and_clean(sample_df, config)
    features = build_features(cleaned, config)
    date_col = config["data"]["date_col"]
    cutoff = time_based_cutoff(cleaned, date_col, config["forecast"]["holdout_days"])
    holdout = cleaned[cleaned[date_col] > cutoff].copy()

    captured: dict = {}

    def _spy_forecast_recursive(self, series_id, future_df):
        captured.setdefault("promo_by_series", {})[series_id] = future_df[
            "promo_flag"
        ].tolist()
        return np.full(len(future_df), 10.0, dtype="float64")

    monkeypatch.setattr(
        "src.model_selection.MLForecaster.forecast_recursive",
        _spy_forecast_recursive,
    )
    _ml_predictions(cleaned, features, holdout, config)

    assert "promo_by_series" in captured
    for sid, flags in captured["promo_by_series"].items():
        expected = (
            holdout[holdout["series_id"] == sid]
            .sort_values(date_col)["promo_flag"]
            .tolist()
        )
        assert flags == expected


# ---------------------------------------------------------------------------
# Frozen baseline record and comparison guards
# ---------------------------------------------------------------------------


def test_frozen_baseline_record_exists_and_has_required_fields():
    assert FROZEN_BASELINE_PATH.is_file()
    baseline = load_frozen_baseline()
    assert baseline["model"] == MLForecaster.name
    assert baseline["metrics"]["wape"] == pytest.approx(0.2722)
    assert baseline["protocol"]["holdout_days"] == 28
    assert baseline["protocol"]["covariate_protocol"] == COVARIATE_PROTOCOL_ACTUAL_HOLDOUT
    assert baseline["protocol"]["chronos_present"] is False


def test_baseline_comparison_invalid_in_quick_mode(config):
    comparison = {
        "summary": pd.DataFrame(
            [{"model": MLForecaster.name, "wape": 0.25, "mae": 7.0, "rmse": 11.0, "bias": 1.0}]
        ),
        "holdout_covariate_protocol": COVARIATE_PROTOCOL_ACTUAL_HOLDOUT,
        "optimistic_eval": True,
        "n_series": 60,
    }
    valid, reasons = assess_baseline_comparison_validity(
        config, comparison, quick=True
    )
    assert not valid
    assert any("quick" in r.lower() for r in reasons)


def test_baseline_comparison_valid_on_matching_protocol(config):
    comparison = {
        "summary": pd.DataFrame(
            [{"model": MLForecaster.name, "wape": 0.25, "mae": 7.0, "rmse": 11.0, "bias": 1.0}]
        ),
        "holdout_covariate_protocol": COVARIATE_PROTOCOL_ACTUAL_HOLDOUT,
        "optimistic_eval": True,
        "n_series": 60,
    }
    valid, reasons = assess_baseline_comparison_validity(
        config, comparison, quick=False
    )
    assert valid, reasons


def test_baseline_beat_when_wape_lower(config):
    comparison = {
        "summary": pd.DataFrame(
            [{"model": MLForecaster.name, "wape": 0.20, "mae": 7.0, "rmse": 11.0, "bias": 1.0}]
        ),
        "holdout_covariate_protocol": COVARIATE_PROTOCOL_ACTUAL_HOLDOUT,
        "optimistic_eval": True,
        "n_series": 60,
    }
    result = compare_to_frozen_baseline(comparison, config, quick=False)
    assert result["comparison_valid"]
    assert result["beat_baseline"]


def test_baseline_invalid_when_protocol_switches_to_production(config):
    comparison = {
        "summary": pd.DataFrame(
            [{"model": MLForecaster.name, "wape": 0.20, "mae": 7.0, "rmse": 11.0, "bias": 1.0}]
        ),
        "holdout_covariate_protocol": COVARIATE_PROTOCOL_PRODUCTION,
        "optimistic_eval": False,
        "n_series": 60,
    }
    result = compare_to_frozen_baseline(comparison, config, quick=False)
    assert not result["comparison_valid"]
    assert not result["beat_baseline"]
