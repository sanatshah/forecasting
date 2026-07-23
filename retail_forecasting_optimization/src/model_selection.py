"""Model comparison and selection.

Runs a time-based backtest: for each series the trailing ``holdout_days`` are
held out. Baselines and the global ML model each forecast that window, and we
score them with the retail metrics in :mod:`src.evaluation`. The model with the
lowest overall WAPE is selected.

Leakage control
---------------
* The ML model is trained only on rows with ``date <= cutoff``. Because lag and
  rolling features reference earlier actuals, training rows never see holdout
  values.
* Holdout predictions for the ML model use **recursive** forecasting seeded
  with train-only history, so multi-step forecasts do not peek at holdout
  actuals.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .evaluation import full_evaluation, wape
from .feature_engineering import build_features
from .model_baseline import get_baseline_models
from .model_ml import MLForecaster
from .utils import get_logger, resolve_path

logger = get_logger(__name__)


def time_based_cutoff(df: pd.DataFrame, date_col: str, holdout_days: int) -> pd.Timestamp:
    """Return the last training date (max date minus holdout window)."""
    max_date = df[date_col].max()
    return max_date - pd.Timedelta(days=holdout_days)


def _dimension_columns(df: pd.DataFrame) -> List[str]:
    """Dimension columns carried into the prediction frame for slicing."""
    return [
        c
        for c in [
            "sku_id",
            "product_id",
            "location_id",
            "channel",
            "department",
            "class",
            "subclass",
            "product_lifecycle_status",
            "promo_flag",
        ]
        if c in df.columns
    ]


def _baseline_predictions(
    cleaned_df: pd.DataFrame,
    holdout: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """Fit each baseline on train history and predict the holdout window."""
    date_col = config["data"]["date_col"]
    target = config["data"]["target_col"]
    cutoff = time_based_cutoff(cleaned_df, date_col, config["forecast"]["holdout_days"])
    train = cleaned_df[cleaned_df[date_col] <= cutoff]

    models = get_baseline_models(config)
    for m in models.values():
        m.fit(train)

    dims = _dimension_columns(holdout)
    out: Dict[str, pd.DataFrame] = {}
    for name, model in models.items():
        frames: List[pd.DataFrame] = []
        for sid, g in holdout.sort_values(date_col).groupby("series_id"):
            h = len(g)
            preds = model.predict(sid, h)
            frame = g[[date_col] + dims].copy()
            frame["series_id"] = sid
            frame["actual"] = g[target].to_numpy()
            frame["forecast"] = np.asarray(preds[:h], dtype="float64")
            frames.append(frame)
        out[name] = pd.concat(frames, ignore_index=True)
    return out


def _ml_predictions(
    cleaned_df: pd.DataFrame,
    features_df: pd.DataFrame,
    holdout: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, MLForecaster]:
    """Train the ML model on train rows and recursively forecast the holdout."""
    date_col = config["data"]["date_col"]
    target = config["data"]["target_col"]
    cutoff = time_based_cutoff(cleaned_df, date_col, config["forecast"]["holdout_days"])

    train_feats = features_df[features_df[date_col] <= cutoff].copy()
    train_hist = cleaned_df[cleaned_df[date_col] <= cutoff].copy()

    model = MLForecaster(config)
    model.fit(train_feats)
    model.set_history(train_hist)

    dims = _dimension_columns(holdout)
    frames: List[pd.DataFrame] = []
    for sid, g in holdout.sort_values(date_col).groupby("series_id"):
        future_df = g.drop(columns=[target])
        preds = model.forecast_recursive(sid, future_df)
        frame = g[[date_col] + dims].copy()
        frame["series_id"] = sid
        frame["actual"] = g[target].to_numpy()
        frame["forecast"] = preds
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), model


def compare_models(
    cleaned_df: pd.DataFrame,
    features_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Backtest all models and pick the best by overall WAPE.

    Returns a dict with:
        metrics_table    - long DataFrame of per-model, per-slice metrics
        summary          - one row per model (overall WAPE and friends)
        best_model_name  - name of the WAPE-winning model
        predictions      - {model_name: prediction DataFrame}
        ml_model         - the fitted MLForecaster (for reuse/explainability)
    """
    date_col = config["data"]["date_col"]
    cutoff = time_based_cutoff(cleaned_df, date_col, config["forecast"]["holdout_days"])
    holdout = cleaned_df[cleaned_df[date_col] > cutoff].copy()
    logger.info(
        "Backtest: train<=%s, holdout=%d rows across %d series",
        cutoff.date(),
        len(holdout),
        holdout["series_id"].nunique(),
    )

    predictions = _baseline_predictions(cleaned_df, holdout, config)
    ml_pred, ml_model = _ml_predictions(cleaned_df, features_df, holdout, config)
    predictions[MLForecaster.name] = ml_pred

    # Per-model, per-slice metrics + overall summary.
    metric_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []
    for name, pred in predictions.items():
        ev = full_evaluation(pred)
        ev.insert(0, "model", name)
        metric_frames.append(ev)
        overall = ev[ev["level"] == "overall"].iloc[0]
        summary_rows.append(
            {
                "model": name,
                "wape": overall["wape"],
                "mape": overall["mape"],
                "mae": overall["mae"],
                "rmse": overall["rmse"],
                "bias": overall["bias"],
                "forecast_accuracy": overall["forecast_accuracy"],
            }
        )

    metrics_table = pd.concat(metric_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values("wape").reset_index(drop=True)
    best_model_name = summary.iloc[0]["model"]
    logger.info(
        "Best model by WAPE: %s (WAPE=%.4f)",
        best_model_name,
        summary.iloc[0]["wape"],
    )

    return {
        "metrics_table": metrics_table,
        "summary": summary,
        "best_model_name": best_model_name,
        "predictions": predictions,
        "ml_model": ml_model,
        "cutoff": cutoff,
    }


def save_metrics(result: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Persist the full metric table to ``outputs/model_metrics.csv``."""
    path = resolve_path(config["paths"]["metrics_csv"])
    path.parent.mkdir(parents=True, exist_ok=True)
    result["metrics_table"].to_csv(path, index=False)
    logger.info("Saved model metrics to %s", path)
