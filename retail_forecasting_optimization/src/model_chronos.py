"""Amazon Chronos-Bolt adapter behind :class:`AdvancedForecasterInterface`.

Chronos is used zero-shot: ``fit`` caches per-series target history and
``predict`` calls ``BaseChronosPipeline.predict_quantiles`` for the requested
horizon. Heavy dependencies (``chronos-forecasting``, ``torch``) are imported
lazily so the rest of the pipeline stays lightweight; when they are missing,
:meth:`ChronosForecaster.is_available` returns ``False`` and model selection
skips this adapter.

OpenMP note
-----------
LightGBM/XGBoost and PyTorch both ship OpenMP runtimes. Loading Chronos after
a GBDT train can deadlock the process. We pin torch/OMP to one thread and set
``KMP_DUPLICATE_LIB_OK`` before importing torch; model selection also runs
Chronos *before* the ML holdout pass so torch initializes first.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .model_ml import AdvancedForecasterInterface
from .utils import get_logger

logger = get_logger(__name__)


def _configure_torch_runtime() -> None:
    """Mitigate OpenMP conflicts with LightGBM/XGBoost before importing torch."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


class ChronosForecaster(AdvancedForecasterInterface):
    """Zero-shot Chronos-Bolt forecaster with a baseline-style predict API."""

    name = "chronos"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.date_col = config["data"]["date_col"]
        self.target = config["data"]["target_col"]
        c = config.get("models", {}).get("chronos", {})
        self.enabled = c.get("enabled", True)
        self.model_id = c.get("model_id", "amazon/chronos-bolt-small")
        self.device = c.get("device", "cpu")
        self.torch_dtype_name = c.get("torch_dtype", "float32")
        self.context_length = c.get("context_length")
        self.quantile_level = float(c.get("quantile_level", 0.5))
        self._history: Dict[str, np.ndarray] = {}
        self._pipeline = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """Return whether Chronos deps are installed and the adapter is enabled."""
        if not self.enabled:
            return False
        if self._available is None:
            try:
                _configure_torch_runtime()
                import torch  # noqa: F401
                from chronos import BaseChronosPipeline  # noqa: F401

                self._available = True
            except Exception as exc:
                logger.info("Chronos unavailable (%s); skipping.", exc)
                self._available = False
        return self._available

    def _resolve_torch_dtype(self):
        import torch

        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return mapping.get(str(self.torch_dtype_name).lower(), torch.float32)

    def _load(self) -> None:
        """Lazy-load the Chronos pipeline (downloads weights on first use)."""
        if self._pipeline is not None:
            return
        _configure_torch_runtime()
        import torch
        from chronos import BaseChronosPipeline

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Already initialized in this process; single-thread still applies.
            pass

        logger.info("Loading Chronos model %s on %s", self.model_id, self.device)
        dtype = self._resolve_torch_dtype()
        try:
            self._pipeline = BaseChronosPipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
                dtype=dtype,
            )
        except TypeError:
            # chronos-forecasting 1.5.x / older transformers accept torch_dtype.
            self._pipeline = BaseChronosPipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
                torch_dtype=dtype,
            )

    def fit(self, history_df: pd.DataFrame) -> "ChronosForecaster":
        """Cache each series' ordered target history (zero-shot; no training)."""
        self._history = {}
        for sid, g in history_df.sort_values(self.date_col).groupby("series_id"):
            self._history[sid] = g[self.target].to_numpy(dtype="float64")
        return self

    def predict(self, series_id: str, horizon: int) -> np.ndarray:
        """Forecast ``horizon`` steps for one series from cached history."""
        import torch

        if series_id not in self._history:
            raise KeyError(f"Series {series_id!r} was not seen in fit().")
        self._load()
        hist = self._history[series_id]
        if self.context_length:
            hist = hist[-int(self.context_length) :]
        _, mean = self._pipeline.predict_quantiles(
            context=torch.tensor(hist, dtype=torch.float32),
            prediction_length=int(horizon),
            quantile_levels=[self.quantile_level],
        )
        preds = mean[0].detach().cpu().numpy()[:horizon]
        return np.clip(np.asarray(preds, dtype="float64"), 0, None)
