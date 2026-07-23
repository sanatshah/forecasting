"""Visualization: save the standard set of retail forecasting plots.

All figures are written as PNGs into ``outputs/plots/``. Matplotlib is used
with a non-interactive backend so the module works in headless runs (CI, cron).
Seaborn is used where it simplifies styling but is imported defensively.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .utils import ensure_dir, get_logger  # noqa: E402

logger = get_logger(__name__)

try:  # seaborn is optional; fall back to matplotlib styling if missing
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    _HAS_SNS = True
except Exception:  # pragma: no cover
    _HAS_SNS = False


def _save(fig: plt.Figure, plots_dir: Path, name: str) -> str:
    """Save and close a figure; return its path as a string."""
    path = plots_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot %s", path)
    return str(path)


def plot_actual_vs_forecast(
    predictions: pd.DataFrame, plots_dir: Path, date_col: str = "date"
) -> str:
    """Aggregate actual vs forecast across all series over the holdout window."""
    agg = predictions.groupby(date_col)[["actual", "forecast"]].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(agg[date_col], agg["actual"], label="Actual", marker="o", ms=3)
    ax.plot(agg[date_col], agg["forecast"], label="Forecast", marker="x", ms=3)
    ax.set_title("Actual vs Forecast Demand (holdout, all series)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units")
    ax.legend()
    fig.autofmt_xdate()
    return _save(fig, plots_dir, "actual_vs_forecast.png")


def plot_forecast_by_sku_location(
    daily_forecast: pd.DataFrame, plots_dir: Path, date_col: str = "date", top_n: int = 6
) -> str:
    """Plot forward forecast curves for the top-N series by total volume."""
    totals = daily_forecast.groupby("series_id")["forecast_units"].sum().sort_values(ascending=False)
    top_series = list(totals.head(top_n).index)
    fig, ax = plt.subplots(figsize=(10, 5))
    for sid in top_series:
        s = daily_forecast[daily_forecast["series_id"] == sid]
        ax.plot(s[date_col], s["forecast_units"], label=sid, marker=".", ms=3)
    ax.set_title(f"Forward Forecast by SKU/Location (top {len(top_series)} series)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Forecast units")
    ax.legend(fontsize=7)
    fig.autofmt_xdate()
    return _save(fig, plots_dir, "forecast_by_sku_location.png")


def plot_error_by_department(metrics_table: pd.DataFrame, plots_dir: Path) -> str:
    """Bar chart of MAE by department for the best model."""
    dept = metrics_table[(metrics_table["level"] == "department")]
    if dept.empty:
        dept = pd.DataFrame({"group": ["n/a"], "mae": [0]})
    dept = dept.groupby("group", as_index=False)["mae"].mean().sort_values("mae")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(dept["group"], dept["mae"])
    ax.set_title("Forecast Error (MAE) by Department")
    ax.set_xlabel("MAE (units)")
    return _save(fig, plots_dir, "error_by_department.png")


def plot_wape_by_dimension(
    metrics_table: pd.DataFrame, plots_dir: Path
) -> str:
    """Grouped bar chart of WAPE by department and by channel."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, level, title in [
        (axes[0], "department", "WAPE by Department"),
        (axes[1], "channel", "WAPE by Channel"),
    ]:
        sub = metrics_table[metrics_table["level"] == level]
        if sub.empty:
            ax.set_visible(False)
            continue
        sub = sub.groupby("group", as_index=False)["wape"].mean().sort_values("wape")
        ax.barh(sub["group"], sub["wape"])
        ax.set_title(title)
        ax.set_xlabel("WAPE")
    return _save(fig, plots_dir, "wape_by_dimension.png")


def plot_inventory_risk_heatmap(recs: pd.DataFrame, plots_dir: Path) -> str:
    """Heatmap of risk-flag counts by department x horizon."""
    primary = recs
    pivot = (
        primary.pivot_table(
            index="department", columns="risk_flag", values="forecast_units",
            aggfunc="count", fill_value=0,
        )
        if "department" in recs.columns
        else pd.DataFrame()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    if pivot.empty:
        ax.text(0.5, 0.5, "No data", ha="center")
    elif _HAS_SNS:
        sns.heatmap(pivot, annot=True, fmt="d", cmap="Reds", ax=ax)
    else:  # pragma: no cover
        im = ax.imshow(pivot.values, aspect="auto", cmap="Reds")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        fig.colorbar(im, ax=ax)
    ax.set_title("Inventory Risk Heatmap (recommendation counts)")
    return _save(fig, plots_dir, "inventory_risk_heatmap.png")


def plot_forecast_distribution(daily_forecast: pd.DataFrame, plots_dir: Path,
                               date_col: str = "date") -> str:
    """Show total forecast demand distribution over the forward horizon."""
    agg = daily_forecast.groupby(date_col)["forecast_units"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(agg[date_col], agg["forecast_units"], alpha=0.4)
    ax.plot(agg[date_col], agg["forecast_units"], marker=".", ms=3)
    ax.set_title("Forward Forecast Demand Over Time (all series)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total forecast units")
    fig.autofmt_xdate()
    return _save(fig, plots_dir, "forecast_distribution.png")


def plot_recommendation_summary(recs: pd.DataFrame, plots_dir: Path) -> str:
    """Bar chart of recommendation counts by reason code."""
    counts = recs["reason_code"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("Recommendation Summary by Reason Code")
    ax.set_ylabel("Count")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    return _save(fig, plots_dir, "recommendation_summary.png")


def generate_all_plots(
    config: Dict[str, Any],
    holdout_predictions: pd.DataFrame,
    metrics_table: pd.DataFrame,
    daily_forecast: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> List[str]:
    """Generate every standard plot and return the list of file paths."""
    plots_dir = ensure_dir(config["paths"]["plots_dir"])
    date_col = config["data"]["date_col"]
    paths: List[str] = []
    # Each plot is guarded so one failure doesn't abort the rest.
    generators = [
        lambda: plot_actual_vs_forecast(holdout_predictions, plots_dir, date_col),
        lambda: plot_forecast_by_sku_location(daily_forecast, plots_dir, date_col),
        lambda: plot_error_by_department(metrics_table, plots_dir),
        lambda: plot_wape_by_dimension(metrics_table, plots_dir),
        lambda: plot_inventory_risk_heatmap(recommendations, plots_dir),
        lambda: plot_forecast_distribution(daily_forecast, plots_dir, date_col),
        lambda: plot_recommendation_summary(recommendations, plots_dir),
    ]
    for gen in generators:
        try:
            paths.append(gen())
        except Exception as exc:  # pragma: no cover
            logger.warning("Plot generation failed: %s", exc)
    return paths
