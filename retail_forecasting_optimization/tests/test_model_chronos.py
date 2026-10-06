"""Tests for the Chronos adapter and its model-selection wiring."""
from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src.data_validation import validate_and_clean
from src.feature_engineering import build_features
from src.forecasting_pipeline import _forecast_with_best, build_future_frame
from src.model_chronos import ChronosForecaster
from src.model_selection import _advanced_predictions, compare_models


class FakeChronos:
    """Deterministic stand-in that avoids chronos/torch dependencies."""

    name = "chronos"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.date_col = config["data"]["date_col"]
        self.target = config["data"]["target_col"]
        self._history: Dict[str, np.ndarray] = {}

    def is_available(self) -> bool:
        return True

    def fit(self, history_df: pd.DataFrame) -> "FakeChronos":
        self._history = {}
        for sid, g in history_df.sort_values(self.date_col).groupby("series_id"):
            self._history[sid] = g[self.target].to_numpy(dtype="float64")
        return self

    def predict(self, series_id: str, horizon: int) -> np.ndarray:
        hist = self._history[series_id]
        # Repeat last value — good enough to exercise the wiring path.
        return np.full(horizon, float(hist[-1]), dtype="float64")


def test_disabled_chronos_is_unavailable(config):
    """enabled=false must make is_available() return False."""
    cfg = {**config, "models": {**config["models"], "chronos": {"enabled": False}}}
    model = ChronosForecaster(cfg)
    assert model.is_available() is False


def test_advanced_predictions_skips_when_unavailable(sample_df, config, monkeypatch):
    """_advanced_predictions returns None when Chronos is not available."""
    cleaned, _ = validate_and_clean(sample_df, config)
    date_col = config["data"]["date_col"]
    cutoff = cleaned[date_col].max() - pd.Timedelta(days=config["forecast"]["holdout_days"])
    holdout = cleaned[cleaned[date_col] > cutoff].copy()

    class UnavailableChronos(ChronosForecaster):
        def is_available(self) -> bool:
            return False

    monkeypatch.setattr(
        "src.model_selection.ChronosForecaster", UnavailableChronos
    )
    assert _advanced_predictions(cleaned, holdout, config) is None


def test_compare_models_includes_fake_chronos(sample_df, config, monkeypatch):
    """A fake Chronos adapter must appear in predictions and the WAPE summary."""
    monkeypatch.setattr("src.model_selection.ChronosForecaster", FakeChronos)
    cleaned, _ = validate_and_clean(sample_df, config)
    features = build_features(cleaned, config)
    result = compare_models(cleaned, features, config)

    assert FakeChronos.name in result["predictions"]
    assert FakeChronos.name in set(result["summary"]["model"])
    pred = result["predictions"][FakeChronos.name]
    assert {"actual", "forecast", "series_id"}.issubset(pred.columns)
    assert len(pred) > 0


def test_forecast_with_best_routes_to_chronos(sample_df, config, monkeypatch):
    """When chronos wins, _forecast_with_best must use the Chronos branch."""
    monkeypatch.setattr(
        "src.forecasting_pipeline.ChronosForecaster", FakeChronos
    )
    cleaned, _ = validate_and_clean(sample_df, config)
    future = build_future_frame(cleaned, config, horizon=7)
    # Minimal ml_model stub; unused when chronos wins.
    daily = _forecast_with_best(
        FakeChronos.name,
        ml_model=None,  # type: ignore[arg-type]
        cleaned_df=cleaned,
        future_df=future,
        config=config,
    )
    assert (daily["model"] == FakeChronos.name).all()
    assert len(daily) == cleaned["series_id"].nunique() * 7
    assert (daily["forecast_value"] >= 0).all()


@pytest.mark.skipif(
    os.environ.get("RUN_CHRONOS_SMOKE") != "1",
    reason="Set RUN_CHRONOS_SMOKE=1 to run the live Chronos smoke test.",
)
def test_live_chronos_smoke(sample_df, config):
    """Optional live smoke: fit + predict with real chronos/torch if installed."""
    pytest.importorskip("chronos")
    pytest.importorskip("torch")

    cleaned, _ = validate_and_clean(sample_df, config)
    model = ChronosForecaster(config)
    assert model.is_available()
    model.fit(cleaned)
    sid = cleaned["series_id"].iloc[0]
    preds = model.predict(sid, horizon=7)
    assert preds.shape == (7,)
    assert np.all(preds >= 0)
    assert np.all(np.isfinite(preds))
