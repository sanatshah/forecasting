"""Global machine-learning forecaster with recursive multi-step prediction.

Why a *global* model?
---------------------
Instead of fitting one model per series, we train a single gradient-boosting
regressor across all series using engineered features plus categorical
segment columns. This shares statistical strength across similar tiers /
channels / partners and scales to many segments - the intended production path.

Backend selection
------------------
The backend is chosen from ``config.models.ml_backend_preference`` using the
first library that is importable, in order. ``sklearn_hist``
(``HistGradientBoostingRegressor``) is always available as a fallback, so the
system runs even without xgboost/lightgbm.

Recursive forecasting
---------------------
Multi-step forecasts are produced recursively: we predict day t+1, append it to
history, rebuild lag/rolling features, then predict t+2, and so on. Future
known covariates (price, promo, content calendar) are supplied by the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .feature_engineering import build_features, get_feature_columns
from .utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Advanced-model interface (placeholder for foundation/deep-learning models)
# ---------------------------------------------------------------------------
class AdvancedForecasterInterface:
    """Abstract hook for future foundation models (Chronos, TimesFM, PatchTST).

    Implementations should accept the same ``history_df`` / ``future_df``
    contract as :class:`MLForecaster` so they can be dropped into
    :mod:`src.model_selection` without pipeline changes. This class is a no-op
    placeholder; it deliberately does not require heavy dependencies to import.
    """

    name = "advanced_placeholder"
    available = False

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def is_available(self) -> bool:
        """Return whether the backing model/dependencies are installed."""
        return self.available

    def fit(self, history_df: pd.DataFrame):  # pragma: no cover - placeholder
        raise NotImplementedError(
            "Advanced model not installed. Implement this adapter to plug in "
            "Chronos/TimesFM/PatchTST. See README 'Future enhancements'."
        )

    def predict(self, *args, **kwargs):  # pragma: no cover - placeholder
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Gradient-boosting backend factory
# ---------------------------------------------------------------------------
@dataclass
class _Backend:
    """Wrapper describing the selected regressor backend."""

    name: str
    estimator: Any
    supports_categorical: bool = False
    categorical_features: List[str] = field(default_factory=list)


def _build_backend(config: Dict[str, Any]) -> _Backend:
    """Instantiate the first available gradient-boosting backend."""
    prefs = config["models"]["ml_backend_preference"]
    p = config["models"]["ml_params"]
    seed = config["project"]["random_seed"]

    for pref in prefs:
        if pref == "lightgbm":
            try:
                from lightgbm import LGBMRegressor

                est = LGBMRegressor(
                    n_estimators=p["n_estimators"],
                    learning_rate=p["learning_rate"],
                    max_depth=p["max_depth"],
                    subsample=p["subsample"],
                    colsample_bytree=p["colsample_bytree"],
                    min_child_samples=p["min_child_samples"],
                    random_state=seed,
                    verbosity=-1,
                )
                logger.info("ML backend: lightgbm")
                return _Backend("lightgbm", est, supports_categorical=True)
            except Exception as exc:
                logger.warning("lightgbm unavailable (%s).", exc)
        elif pref == "xgboost":
            try:
                from xgboost import XGBRegressor

                est = XGBRegressor(
                    n_estimators=p["n_estimators"],
                    learning_rate=p["learning_rate"],
                    max_depth=p["max_depth"],
                    subsample=p["subsample"],
                    colsample_bytree=p["colsample_bytree"],
                    random_state=seed,
                    tree_method="hist",
                    enable_categorical=True,
                )
                logger.info("ML backend: xgboost")
                return _Backend("xgboost", est, supports_categorical=True)
            except Exception as exc:
                logger.warning("xgboost unavailable (%s).", exc)
        elif pref == "sklearn_hist":
            from sklearn.ensemble import HistGradientBoostingRegressor

            est = HistGradientBoostingRegressor(
                max_iter=p["n_estimators"],
                learning_rate=p["learning_rate"],
                max_depth=p["max_depth"],
                random_state=seed,
            )
            logger.info("ML backend: sklearn HistGradientBoostingRegressor")
            return _Backend("sklearn_hist", est, supports_categorical=False)

    # Absolute fallback.
    from sklearn.ensemble import HistGradientBoostingRegressor

    logger.info("ML backend: sklearn HistGradientBoostingRegressor (fallback)")
    return _Backend("sklearn_hist", HistGradientBoostingRegressor(random_state=seed))


class MLForecaster:
    """Global gradient-boosting forecaster over engineered subscriber features."""

    name = "ml_gradient_boosting"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.date_col = config["data"]["date_col"]
        self.target = config["data"]["target_col"]
        self.backend = _build_backend(config)
        self.feature_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self._category_maps: Dict[str, Dict[Any, int]] = {}
        self._history_df: Optional[pd.DataFrame] = None
        self._fitted = False

    # -- encoding helpers ---------------------------------------------------
    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Integer-encode hierarchy categoricals (stable across train/infer)."""
        df = df.copy()
        for col in self.categorical_cols:
            if fit:
                cats = {v: i for i, v in enumerate(sorted(df[col].astype(str).unique()))}
                self._category_maps[col] = cats
            cats = self._category_maps.get(col, {})
            df[col] = df[col].astype(str).map(cats).fillna(-1).astype("int64")
        return df

    def _prepare_matrix(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Select feature columns and encode categoricals into a numeric X."""
        X = df[self.feature_cols].copy()
        for col in self.categorical_cols:
            if col in X.columns:
                pass  # already numeric-encoded upstream
        # Fill remaining NaNs (early lags/rolling) with 0.
        X = X.fillna(0.0)
        return X

    # -- training -----------------------------------------------------------
    def fit(self, features_df: pd.DataFrame) -> "MLForecaster":
        """Fit on an already-feature-engineered training frame."""
        from .feature_engineering import HIERARCHY_COLUMNS

        self.feature_cols = get_feature_columns(features_df, self.config)
        self.categorical_cols = [c for c in HIERARCHY_COLUMNS if c in self.feature_cols]

        df = self._encode_categoricals(features_df, fit=True)
        X = self._prepare_matrix(df, fit=True)
        y = df[self.target].to_numpy(dtype="float64")

        logger.info(
            "Training %s on %d rows x %d features", self.backend.name, len(X), X.shape[1]
        )
        self.backend.estimator.fit(X, y)
        self._fitted = True
        return self

    def set_history(self, history_df: pd.DataFrame) -> None:
        """Store cleaned history (with series_id) for recursive forecasting."""
        self._history_df = history_df.copy()

    # -- prediction ---------------------------------------------------------
    def _predict_matrix(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict directly on a fully-featured frame (single shot)."""
        df = self._encode_categoricals(features_df, fit=False)
        X = self._prepare_matrix(df, fit=False)
        preds = self.backend.estimator.predict(X)
        return np.clip(np.asarray(preds, dtype="float64"), 0, None)

    def predict_on_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Public single-shot prediction (used for holdout evaluation)."""
        if not self._fitted:
            raise RuntimeError("MLForecaster must be fitted before predict.")
        return self._predict_matrix(features_df)

    def forecast_recursive(
        self, series_id: str, future_df: pd.DataFrame
    ) -> np.ndarray:
        """Recursively forecast the target for future rows of one series.

        Parameters
        ----------
        future_df:
            Rows for the horizon with known covariates (date, price, promo,
            content calendar, segment, subscriber base) but no target.

        Returns the predicted target for each future row.
        """
        if self._history_df is None:
            raise RuntimeError("Call set_history() before forecast_recursive().")

        hist = self._history_df[self._history_df["series_id"] == series_id].copy()
        future_sorted = future_df.sort_values(self.date_col).reset_index(drop=True)
        preds: List[float] = []

        for step, future_row in future_sorted.iterrows():
            # Append all remaining future rows (target unknown) so calendar
            # look-ahead features such as days_until_next_tentpole match training.
            remaining = future_sorted.iloc[step:].copy()
            remaining[self.target] = np.nan
            working = pd.concat([hist, remaining], ignore_index=True)
            feat = build_features(working, self.config)
            current = feat[feat[self.date_col] == future_row[self.date_col]]
            yhat = float(self._predict_matrix(current.iloc[[0]])[0])
            preds.append(yhat)

            # Commit the prediction into history so subsequent lags see it.
            committed = future_row.to_dict()
            committed[self.target] = yhat
            hist = pd.concat([hist, pd.DataFrame([committed])], ignore_index=True)

        return np.asarray(preds, dtype="float64")

    # -- explainability -----------------------------------------------------
    def feature_importance(self) -> pd.DataFrame:
        """Return a sorted feature-importance table if the backend exposes it."""
        est = self.backend.estimator
        importances = getattr(est, "feature_importances_", None)
        if importances is None:
            return pd.DataFrame(columns=["feature", "importance"])
        imp = pd.DataFrame(
            {"feature": self.feature_cols, "importance": importances}
        ).sort_values("importance", ascending=False)
        return imp.reset_index(drop=True)
