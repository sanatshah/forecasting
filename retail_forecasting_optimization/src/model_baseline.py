"""Naive and statistical baseline forecasters.

All forecasters share a minimal interface:

    fit(history_df) -> self
    predict(series_id, horizon) -> np.ndarray of length ``horizon``

``history_df`` is the cleaned, feature-free (raw target) data containing at
least ``series_id``, the date column, and the target column. Baselines only
need the target history, which makes them fast and dependency-light.

Baselines provided
-------------------
* ``NaiveLast7Average``     - repeat the mean of the last 7 observed days.
* ``SameWeekdayLastWeek``   - repeat value from 7 days ago (weekly naive), tiled.
* ``SeasonalNaive``         - repeat the last full weekly (7-day) pattern.
* ``MovingAverage``         - repeat a configurable-window trailing mean.
* ``SarimaxForecaster``     - SARIMAX per series (guarded, optional).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)


class BaseForecaster:
    """Common storage of per-series target history."""

    name = "base"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.date_col = config["data"]["date_col"]
        self.target = config["data"]["target_col"]
        self._history: Dict[str, np.ndarray] = {}

    def fit(self, history_df: pd.DataFrame) -> "BaseForecaster":
        """Cache each series' ordered target history."""
        for sid, g in history_df.sort_values(self.date_col).groupby("series_id"):
            self._history[sid] = g[self.target].to_numpy(dtype="float64")
        return self

    def predict(self, series_id: str, horizon: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class NaiveLast7Average(BaseForecaster):
    """Forecast every future day as the mean of the last 7 observed days."""

    name = "naive_last7_avg"

    def predict(self, series_id: str, horizon: int) -> np.ndarray:
        hist = self._history.get(series_id, np.array([0.0]))
        val = float(np.mean(hist[-7:])) if len(hist) else 0.0
        return np.full(horizon, val)


class SameWeekdayLastWeek(BaseForecaster):
    """Weekly naive: future day d maps to the observed value 7 days earlier.

    We tile the last 7 observed values across the horizon so that each future
    weekday reuses the matching weekday from the most recent week.
    """

    name = "same_weekday_last_week"

    def predict(self, series_id: str, horizon: int) -> np.ndarray:
        hist = self._history.get(series_id, np.array([0.0]))
        last_week = hist[-7:] if len(hist) >= 7 else np.resize(hist, 7)
        reps = int(np.ceil(horizon / 7))
        return np.tile(last_week, reps)[:horizon]


class SeasonalNaive(SameWeekdayLastWeek):
    """Alias emphasising the weekly seasonal-naive interpretation."""

    name = "seasonal_naive"


class MovingAverage(BaseForecaster):
    """Repeat a trailing moving average of the configured window."""

    name = "moving_average"

    def __init__(self, config: Dict[str, Any], window: int = 14):
        super().__init__(config)
        self.window = window

    def predict(self, series_id: str, horizon: int) -> np.ndarray:
        hist = self._history.get(series_id, np.array([0.0]))
        val = float(np.mean(hist[-self.window:])) if len(hist) else 0.0
        return np.full(horizon, val)


class SarimaxForecaster(BaseForecaster):
    """SARIMAX per series (optional/guarded).

    SARIMAX is computationally heavy, so:
      * it is only fitted for up to ``models.sarimax.max_series`` series;
      * fitting failures fall back to a last-7 average for that series;
      * if statsmodels is unavailable the class degrades to a naive baseline.

    For series beyond the cap (or on any error) we store a fallback value.
    """

    name = "sarimax"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        s = config["models"]["sarimax"]
        self.enabled = bool(s.get("enabled", True))
        self.max_series = int(s.get("max_series", 5))
        self.order = tuple(s.get("order", [1, 1, 1]))
        self.seasonal_order = tuple(s.get("seasonal_order", [1, 1, 1, 7]))
        self._fitted: Dict[str, Any] = {}
        self._fallback: Dict[str, float] = {}

    def fit(self, history_df: pd.DataFrame) -> "SarimaxForecaster":
        super().fit(history_df)
        if not self.enabled:
            return self
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: WPS433
        except Exception as exc:  # pragma: no cover
            logger.warning("statsmodels unavailable (%s); SARIMAX -> naive.", exc)
            return self

        # Fit on the series with the most volume first (most informative).
        order_by_vol = sorted(
            self._history.items(), key=lambda kv: -float(np.sum(kv[1]))
        )
        for sid, hist in order_by_vol[: self.max_series]:
            self._fallback[sid] = float(np.mean(hist[-7:])) if len(hist) else 0.0
            try:
                model = SARIMAX(
                    hist,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                self._fitted[sid] = model.fit(disp=False)
            except Exception as exc:  # pragma: no cover
                logger.warning("SARIMAX fit failed for %s (%s); using fallback.", sid, exc)
        return self

    def predict(self, series_id: str, horizon: int) -> np.ndarray:
        if series_id in self._fitted:
            try:
                fc = self._fitted[series_id].forecast(steps=horizon)
                return np.clip(np.asarray(fc, dtype="float64"), 0, None)
            except Exception:  # pragma: no cover
                pass
        # Fallback: last-7 average.
        hist = self._history.get(series_id, np.array([0.0]))
        val = self._fallback.get(
            series_id, float(np.mean(hist[-7:])) if len(hist) else 0.0
        )
        return np.full(horizon, max(val, 0.0))


def get_baseline_models(config: Dict[str, Any]) -> Dict[str, BaseForecaster]:
    """Instantiate the standard set of baseline forecasters."""
    models: Dict[str, BaseForecaster] = {
        NaiveLast7Average.name: NaiveLast7Average(config),
        SameWeekdayLastWeek.name: SameWeekdayLastWeek(config),
        SeasonalNaive.name: SeasonalNaive(config),
        MovingAverage.name: MovingAverage(config, window=14),
    }
    if config["models"]["sarimax"].get("enabled", True):
        models[SarimaxForecaster.name] = SarimaxForecaster(config)
    return models
