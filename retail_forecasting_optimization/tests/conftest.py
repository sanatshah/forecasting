"""Shared pytest fixtures.

Provides a loaded config and a small synthetic subscriber dataset so tests run
fast and deterministically without depending on the full sample CSV.
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
    """Build a small, clean two-segment dataset (60 days each).

    The subscriber ledger is consistent: ``paid_subs_eod == paid_subs_bod +
    gross_adds - churned_subs`` and each day's bod is the prior day's eod.
    Sundays are tentpoles (Sunday Night Football) and days 20-24 are a promo.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    sunday = (dates.dayofweek == 6).astype(int)
    promo = np.zeros(len(dates), dtype=int)
    promo[20:25] = 1

    frames = []
    for tier, channel, partner, price, start_subs, adds_base in [
        ("Premium", "direct", "Peacock", 7.99, 50_000, 120),
        ("Ad Tier", "app_store", "Apple/Google", 4.99, 30_000, 80),
    ]:
        adds = np.clip(
            adds_base * (1 + 0.5 * sunday + 0.3 * promo) + rng.normal(0, 8, len(dates)), 0, None
        ).round().astype(int)
        churn = np.clip(adds_base * 0.8 + rng.normal(0, 6, len(dates)), 0, None).round().astype(int)
        bod = np.empty(len(dates), dtype=int)
        eod = np.empty(len(dates), dtype=int)
        base = start_subs
        for i in range(len(dates)):
            bod[i] = base
            base = base + adds[i] - churn[i]
            eod[i] = base
        hours = (bod * (0.9 + 0.4 * sunday) + rng.normal(0, 200, len(dates))).clip(0).round(1)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "tier": tier,
                    "acquisition_channel": channel,
                    "distribution_partner": partner,
                    "gross_adds": adds,
                    "churned_subs": churn,
                    "paid_subs_bod": bod,
                    "paid_subs_eod": eod,
                    "hours_watched": hours,
                    "daily_active_subs": (bod * 0.35).round().astype(int),
                    "list_price": price,
                    "effective_price": np.where(promo == 1, round(price * 0.6, 2), price),
                    "discount_pct": np.where(promo == 1, 0.4, 0.0),
                    "promo_flag": promo,
                    "promo_name": np.where(promo == 1, "Winter Sale", None),
                    "tentpole_flag": sunday,
                    "tentpole_name": np.where(sunday == 1, "Sunday Night Football", None),
                    "tentpole_type": np.where(sunday == 1, "sports", "none"),
                    "tentpole_intensity": np.where(sunday == 1, 1.45, 1.0),
                    "price_increase_flag": 0,
                    "holiday_flag": 0,
                    "fiscal_week": dates.isocalendar().week.astype(int).to_numpy(),
                    "fiscal_month": dates.month,
                    "fiscal_quarter": dates.quarter,
                    "season": "Winter",
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    return add_series_id(df, config)
