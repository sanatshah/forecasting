"""Visualization: render standard pipeline plots via the declarative plot engine.

Builtin chart definitions live in ``plot_specs/*.json``. This module remains the
pipeline-facing entry point so ``main.py`` does not need to know about specs.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from .plotting.renderer import render_builtin_specs
from .utils import get_logger

logger = get_logger(__name__)


def generate_all_plots(
    config: Dict[str, Any],
    holdout_predictions: pd.DataFrame,
    metrics_table: pd.DataFrame,
    daily_forecast: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> List[str]:
    """Generate every builtin plot_spec and return the list of file paths.

    In-memory frames from the pipeline are preferred when available so plots
    work even before CSVs are flushed; the CLI path loads from disk instead.
    """
    data_by_dataset = {
        "holdout_predictions": holdout_predictions,
        "metrics": metrics_table,
        "forecasts": daily_forecast,
        "recommendations": recommendations,
    }
    paths = render_builtin_specs(config, data_by_dataset=data_by_dataset)
    logger.info("Generated %d plots", len(paths))
    return paths
