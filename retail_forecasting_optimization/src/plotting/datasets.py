"""Named dataset registry over pipeline CSV artifacts.

Each entry maps a short name (e.g. ``forecasts``) to a config path key, optional
date columns to parse, and a one-line description for agent discovery.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..utils import get_logger, resolve_path

logger = get_logger(__name__)

_MISSING_HINT = (
    "Run the pipeline first to create artifacts: "
    "`python main.py --quick` (from retail_forecasting_optimization/). "
    "See the run-pipeline skill for details."
)


@dataclass(frozen=True)
class DatasetDef:
    """Metadata for one named dataset in the registry."""

    name: str
    path_key: str
    description: str
    date_columns: tuple[str, ...] = ()
    # When path_key points at a directory, the file lives at processed_dir/filename.
    filename: Optional[str] = None


DATASETS: Dict[str, DatasetDef] = {
    "forecasts": DatasetDef(
        name="forecasts",
        path_key="forecasts_csv",
        description=(
            "Daily forward forecasts per segment (tier/channel): gross adds, churn, "
            "hours, plus derived net adds, paid subs and hours per paid sub."
        ),
        date_columns=("date",),
    ),
    "recommendations": DatasetDef(
        name="recommendations",
        path_key="recommendations_csv",
        description="Segment scenarios per horizon: risk flags, actions, price-change deltas.",
        date_columns=("date", "forecast_start", "forecast_end"),
    ),
    "okr_summary": DatasetDef(
        name="okr_summary",
        path_key="okr_summary_csv",
        description="Growth OKR rollup per horizon: high-value net adds, usage per paid sub, baseline vs price change.",
        date_columns=("forecast_start", "forecast_end"),
    ),
    "metrics": DatasetDef(
        name="metrics",
        path_key="metrics_csv",
        description="Per-model, per-target evaluation metrics by level (overall, tier, channel, segment).",
    ),
    "holdout_predictions": DatasetDef(
        name="holdout_predictions",
        path_key="holdout_predictions_csv",
        description="Holdout-window actual vs forecast for the best model, per target.",
        date_columns=("date",),
    ),
    "cleaned": DatasetDef(
        name="cleaned",
        path_key="processed_dir",
        description="Validated cleaned history used for training.",
        date_columns=("date",),
        filename="cleaned.csv",
    ),
}

DATASET_NAMES: List[str] = list(DATASETS.keys())


def _resolve_csv_path(config: Dict[str, Any], dset: DatasetDef) -> Path:
    """Resolve the on-disk CSV path for a dataset definition."""
    base = resolve_path(config["paths"][dset.path_key])
    if dset.filename is not None:
        return base / dset.filename
    return base


def dataset_path(config: Dict[str, Any], name: str) -> Path:
    """Return the resolved CSV path for ``name``; raise if unknown."""
    if name not in DATASETS:
        raise KeyError(
            f"Unknown dataset {name!r}. Valid datasets: {', '.join(DATASET_NAMES)}"
        )
    return _resolve_csv_path(config, DATASETS[name])


def list_datasets(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return registry rows with path, existence, and optional row count."""
    rows: List[Dict[str, Any]] = []
    for name, dset in DATASETS.items():
        path = _resolve_csv_path(config, dset)
        exists = path.is_file()
        n_rows: Optional[int] = None
        if exists:
            try:
                n_rows = int(sum(1 for _ in open(path, "r", encoding="utf-8")) - 1)
            except OSError:
                n_rows = None
        rows.append(
            {
                "name": name,
                "description": dset.description,
                "path": str(path),
                "exists": exists,
                "n_rows": n_rows,
            }
        )
    return rows


def load_dataset(config: Dict[str, Any], name: str) -> pd.DataFrame:
    """Load a named dataset CSV; raise a clear error if the file is missing."""
    if name not in DATASETS:
        raise KeyError(
            f"Unknown dataset {name!r}. Valid datasets: {', '.join(DATASET_NAMES)}"
        )
    dset = DATASETS[name]
    path = _resolve_csv_path(config, dset)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset {name!r} not found at {path}. {_MISSING_HINT}"
        )
    df = pd.read_csv(path)
    for col in dset.date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    logger.info("Loaded dataset %s (%d rows) from %s", name, len(df), path)
    return df


def describe_dataset(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Return columns, dtypes, null counts, and sample values for agents."""
    df = load_dataset(config, name)
    columns: List[Dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        sample = (
            series.dropna().astype(str).unique()[:5].tolist()
            if series.dtype == object or str(series.dtype).startswith("string")
            else series.dropna().head(5).tolist()
        )
        # JSON-serialize timestamps / numpy scalars.
        sample_out = []
        for v in sample:
            if hasattr(v, "isoformat"):
                sample_out.append(v.isoformat())
            elif hasattr(v, "item"):
                sample_out.append(v.item())
            else:
                sample_out.append(v)
        columns.append(
            {
                "name": col,
                "dtype": str(series.dtype),
                "nulls": int(series.isna().sum()),
                "n_unique": int(series.nunique(dropna=True)),
                "sample": sample_out,
            }
        )
    return {
        "name": name,
        "description": DATASETS[name].description,
        "path": str(dataset_path(config, name)),
        "n_rows": len(df),
        "columns": columns,
    }
