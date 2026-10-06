"""Explainability: feature importance, top forecast drivers, and narratives.

Two layers of explainability:

1. **Model-level** - global feature importance from the fitted ML model, plus a
   short list of the strongest drivers.
2. **Recommendation-level** - the scenario engine already writes a plain
   English ``explanation`` per segment; helpers here summarize those for
   reporting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .model_ml import MLForecaster  # noqa: E402
from .utils import ensure_dir, get_logger  # noqa: E402

logger = get_logger(__name__)


def feature_importance_table(ml_model: MLForecaster) -> pd.DataFrame:
    """Return the ML model's feature-importance table (may be empty)."""
    return ml_model.feature_importance()


def top_drivers(ml_model: MLForecaster, top_n: int = 10) -> List[str]:
    """Return a human-readable list of the top-N forecast drivers."""
    imp = ml_model.feature_importance()
    if imp.empty:
        return []
    return [
        f"{r.feature} ({r.importance:.3f})"
        for r in imp.head(top_n).itertuples(index=False)
    ]


def plot_feature_importance(
    ml_model: MLForecaster, plots_dir: Path, top_n: int = 15
) -> Optional[str]:
    """Save a horizontal bar chart of the top-N feature importances."""
    imp = ml_model.feature_importance()
    if imp.empty:
        logger.info("No feature importance available for backend %s", ml_model.backend.name)
        return None
    imp = imp.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp["feature"], imp["importance"])
    ax.set_title(f"Top {top_n} Forecast Drivers ({ml_model.backend.name})")
    ax.set_xlabel("Importance")
    path = plots_dir / "feature_importance.png"
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature importance plot to %s", path)
    return str(path)


def recommendation_narratives(recs: pd.DataFrame, top_n: int = 5) -> List[str]:
    """Return the most material segment explanations (by net-adds magnitude)."""
    if recs.empty or "explanation" not in recs.columns:
        return []
    ranked = recs.reindex(
        recs["forecast_net_adds"].abs().sort_values(ascending=False).index
    )
    return ranked["explanation"].head(top_n).tolist()


def generate_explainability(
    config: Dict[str, Any],
    ml_model: MLForecaster,
    recommendations: pd.DataFrame,
) -> Dict[str, Any]:
    """Bundle model-level and recommendation-level explainability artifacts."""
    plots_dir = ensure_dir(config["paths"]["plots_dir"])
    return {
        "feature_importance": feature_importance_table(ml_model),
        "top_drivers": top_drivers(ml_model),
        "importance_plot": plot_feature_importance(ml_model, plots_dir),
        "narratives": recommendation_narratives(recommendations),
    }
