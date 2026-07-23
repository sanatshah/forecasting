"""Shared pytest fixtures.

Provides a loaded config and a small synthetic retail dataset so tests run fast
and deterministically without depending on the full sample CSV.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is importable when tests run from any cwd.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data_loader import add_series_id  # noqa: E402
from src.utils import load_config  # noqa: E402


@pytest.fixture(scope="session")
def config():
    """Load the project configuration once per test session."""
    return load_config()


@pytest.fixture()
def sample_df(config):
    """Build a small, clean two-series dataset (~60 days each)."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    frames = []
    for sku, loc, dept, base in [
        ("SKU9001", "LOC01", "Womens Apparel", 20),
        ("SKU9002", "LOC02", "Home", 8),
    ]:
        units = np.clip(base + rng.normal(0, 3, len(dates)), 0, None).round()
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "sku_id": sku,
                    "product_id": "P" + sku,
                    "location_id": loc,
                    "department": dept,
                    "class": "ClassA",
                    "subclass": "SubA",
                    "channel": "store",
                    "units_sold": units.astype(int),
                    "sales_revenue": units * 10.0,
                    "regular_price": 10.0,
                    "selling_price": 10.0,
                    "markdown_pct": 0.0,
                    "promo_flag": 0,
                    "promo_event_name": None,
                    "inventory_on_hand": 200,
                    "inventory_in_transit": 0,
                    "stockout_flag": 0,
                    "holiday_flag": 0,
                    "fiscal_week": dates.isocalendar().week.astype(int),
                    "fiscal_month": dates.month,
                    "fiscal_quarter": dates.quarter,
                    "season": "Winter",
                    "product_lifecycle_status": "Core",
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    return add_series_id(df, config)
