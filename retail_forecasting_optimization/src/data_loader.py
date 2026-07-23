"""Data ingestion: load the raw retail CSV into a typed, sorted DataFrame.

The loader is deliberately conservative: it parses dates, coerces numeric
columns, and sorts by series + date so that every downstream module (feature
engineering, splitting, forecasting) can rely on a stable ordering.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .utils import get_logger, resolve_path

logger = get_logger(__name__)

# Columns we expect to be numeric. Coerced with errors="coerce" so bad values
# surface as NaN for the validation layer rather than crashing the load.
_NUMERIC_COLUMNS = [
    "units_sold",
    "sales_revenue",
    "regular_price",
    "selling_price",
    "markdown_pct",
    "promo_flag",
    "inventory_on_hand",
    "inventory_in_transit",
    "stockout_flag",
    "holiday_flag",
    "fiscal_week",
    "fiscal_month",
    "fiscal_quarter",
]


def load_raw_data(config: Dict[str, Any], path: str | Path | None = None) -> pd.DataFrame:
    """Load the input CSV described by ``config`` (or an explicit ``path``).

    Steps:
        1. Read CSV.
        2. Parse the date column to ``datetime64``.
        3. Coerce known numeric columns.
        4. Sort by series keys + date and reset the index.

    Returns a DataFrame; validation and cleaning happen in
    :mod:`src.data_validation`.
    """
    data_cfg = config["data"]
    csv_path = resolve_path(path or config["paths"]["input_csv"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    logger.info("Loading raw data from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows x %d columns", df.shape[0], df.shape[1])

    date_col = data_cfg["date_col"]
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    for col in _NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sort_keys = [k for k in data_cfg["series_keys"] if k in df.columns]
    sort_keys = sort_keys + [date_col] if date_col in df.columns else sort_keys
    if sort_keys:
        df = df.sort_values(sort_keys).reset_index(drop=True)

    return df


def add_series_id(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Add a single ``series_id`` column joining the configured series keys.

    Many downstream operations are simpler with one grouping key than with a
    multi-column groupby, so we materialize it once here.
    """
    keys = config["data"]["series_keys"]
    df = df.copy()
    df["series_id"] = df[keys].astype(str).agg("|".join, axis=1)
    return df
