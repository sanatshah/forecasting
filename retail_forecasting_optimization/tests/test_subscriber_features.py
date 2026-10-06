"""Isolation tests for subscriber feature helpers in ``src.feature_engineering``.

Covers ``_add_event_features`` (promo and tentpole timing / uplift),
``_add_base_features`` (trailing churn rate, hours per sub) and the
price-increase timing in ``_add_price_features``.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    _NO_EVENT,
    _add_base_features,
    _add_event_features,
    _add_price_features,
)


def _event_frame(flags: List[int], values: List[float], series_id: str = "A") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": series_id,
            "date": pd.date_range("2024-01-01", periods=len(flags), freq="D"),
            "tentpole_flag": flags,
            "gross_adds": values,
        }
    )


def _run_event(df: pd.DataFrame, window: int = 28) -> pd.DataFrame:
    out = _add_event_features(
        df, "date", {"event_window": window}, "gross_adds", "tentpole_flag", "tentpole"
    )
    return out.sort_values(["series_id", "date"]).reset_index(drop=True)


def _expected_uplift(flags: List[int], values: List[float], window: int) -> List[float]:
    """Reference implementation: prior event-day mean / prior non-event-day mean."""
    out = []
    for t in range(len(flags)):
        lo = max(0, t - window)
        prior = range(lo, t)
        ev = [values[i] for i in prior if flags[i] == 1]
        base = [values[i] for i in prior if flags[i] != 1]
        if not ev:
            ev = [values[i] for i in range(t) if flags[i] == 1]
        if not base:
            base = [values[i] for i in range(t) if flags[i] != 1]
        if not ev:
            out.append(1.0)
        elif not base or np.mean(base) == 0:
            out.append(1.0)
        else:
            out.append(float(np.mean(ev) / np.mean(base)))
    return out


# ---------------------------------------------------------------------------
# _add_event_features
# ---------------------------------------------------------------------------
def test_event_columns_created():
    out = _run_event(_event_frame([0, 1, 0], [10, 30, 10]))
    for col in ["days_since_last_tentpole", "days_until_next_tentpole", "tentpole_uplift_ratio"]:
        assert col in out.columns


def test_event_skipped_when_flag_missing():
    df = _event_frame([0, 1, 0], [10, 30, 10]).drop(columns="tentpole_flag")
    out = _add_event_features(df, "date", {"event_window": 28}, "gross_adds", "tentpole_flag", "tentpole")
    assert list(out.columns) == list(df.columns)


def test_event_timing_without_target_skips_uplift():
    df = _event_frame([0, 1, 0], [10, 30, 10]).drop(columns="gross_adds")
    out = _run_event(df)
    assert "days_until_next_tentpole" in out.columns
    assert "tentpole_uplift_ratio" not in out.columns


def test_event_hand_checked_values():
    out = _run_event(_event_frame([0, 1, 0, 1, 0], [10, 30, 10, 40, 10]))
    assert out["days_since_last_tentpole"].tolist() == [_NO_EVENT, 0, 1, 0, 1]
    assert out["days_until_next_tentpole"].tolist() == [1, 0, 1, 0, _NO_EVENT]
    assert out["tentpole_uplift_ratio"].tolist() == pytest.approx([1.0, 1.0, 3.0, 3.0, 3.5])


def test_event_matches_reference_on_longer_series():
    rng = np.random.default_rng(1)
    flags = (rng.random(40) < 0.25).astype(int).tolist()
    values = rng.integers(5, 50, 40).astype(float).tolist()
    out = _run_event(_event_frame(flags, values), window=7)
    assert out["tentpole_uplift_ratio"].tolist() == pytest.approx(_expected_uplift(flags, values, 7))


def test_event_uplift_defaults_to_one_without_prior_events():
    out = _run_event(_event_frame([0, 0, 0, 1], [10, 12, 11, 50]))
    assert (out["tentpole_uplift_ratio"] == 1.0).all()


def test_event_uplift_no_leakage_from_today():
    flags, values = [0, 1, 0, 1, 0], [10.0, 30.0, 10.0, 40.0, 10.0]
    base = _run_event(_event_frame(flags, values))
    mutated_values = list(values)
    mutated_values[3] = 9_999.0
    mutated = _run_event(_event_frame(flags, mutated_values))
    assert mutated.loc[3, "tentpole_uplift_ratio"] == base.loc[3, "tentpole_uplift_ratio"]
    assert mutated.loc[4, "tentpole_uplift_ratio"] != base.loc[4, "tentpole_uplift_ratio"]


def test_event_window_falls_back_to_expanding():
    out = _run_event(_event_frame([1, 0, 0, 0, 0], [50, 10, 10, 10, 10]), window=2)
    assert out.loc[4, "tentpole_uplift_ratio"] == pytest.approx(5.0)


def test_event_uplift_defaults_to_one_without_prior_baseline_days():
    out = _run_event(_event_frame([1, 1, 0], [50, 60, 10]))
    assert out["tentpole_uplift_ratio"].tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_event_per_series_independence():
    a = _event_frame([0, 1, 0, 1, 0], [10, 30, 10, 40, 10], "A")
    b = _event_frame([1, 0, 0, 0, 1], [80, 20, 20, 20, 90], "B")
    combined = _run_event(pd.concat([a, b], ignore_index=True))
    alone = _run_event(b)
    got = combined[combined["series_id"] == "B"].reset_index(drop=True)
    for col in ["days_since_last_tentpole", "days_until_next_tentpole", "tentpole_uplift_ratio"]:
        assert got[col].tolist() == pytest.approx(alone[col].tolist())


def test_event_promo_prefix():
    df = _event_frame([0, 1, 0], [10, 30, 10]).rename(columns={"tentpole_flag": "promo_flag"})
    out = _add_event_features(df, "date", {"event_window": 28}, "gross_adds")
    assert {"days_since_last_promo", "days_until_next_promo", "promo_uplift_ratio"}.issubset(out.columns)


# ---------------------------------------------------------------------------
# _add_base_features
# ---------------------------------------------------------------------------
def _base_frame(churn: List[float], bod: List[float], series_id: str = "A") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": series_id,
            "date": pd.date_range("2024-01-01", periods=len(churn), freq="D"),
            "churned_subs": churn,
            "paid_subs_bod": bod,
            "hours_watched": [b * 1.5 for b in bod],
        }
    )


def test_base_hand_checked_rate():
    out = _add_base_features(_base_frame([10, 20, 30], [1000, 1000, 1000]), {"event_window": 28})
    assert out["trailing_churn_rate"].tolist() == pytest.approx([0.0, 0.01, 0.015])
    assert out["trailing_hours_per_sub"].tolist() == pytest.approx([0.0, 1.5, 1.5])


def test_base_skipped_without_paid_subs_bod():
    df = _base_frame([10, 20], [1000, 1000]).drop(columns="paid_subs_bod")
    out = _add_base_features(df, {"event_window": 28})
    assert "trailing_churn_rate" not in out.columns


def test_base_no_leakage_from_today():
    base = _add_base_features(_base_frame([10, 20, 30], [1000] * 3), {"event_window": 28})
    mutated = _add_base_features(_base_frame([10, 20, 9_999], [1000] * 3), {"event_window": 28})
    assert mutated.loc[2, "trailing_churn_rate"] == base.loc[2, "trailing_churn_rate"]


def test_base_future_rows_do_not_dilute_rate():
    df = _base_frame([10, 20, np.nan, np.nan], [1000, 1000, 5000, 5000])
    out = _add_base_features(df, {"event_window": 28})
    assert out.loc[3, "trailing_churn_rate"] == pytest.approx(0.015)


def test_base_per_series_independence():
    a = _base_frame([10, 20, 30], [1000] * 3, "A")
    b = _base_frame([100, 100, 100], [2000] * 3, "B")
    combined = _add_base_features(pd.concat([a, b], ignore_index=True), {"event_window": 28})
    got = combined[combined["series_id"] == "B"]["trailing_churn_rate"].tolist()
    assert got == pytest.approx([0.0, 0.05, 0.05])


# ---------------------------------------------------------------------------
# _add_price_features: price-increase timing
# ---------------------------------------------------------------------------
def test_days_since_price_increase():
    df = pd.DataFrame(
        {
            "series_id": "A",
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "list_price": [7.99, 7.99, 10.99, 10.99, 10.99],
            "effective_price": [7.99, 7.99, 10.99, 10.99, 10.99],
            "price_increase_flag": [0, 0, 1, 0, 0],
        }
    )
    out = _add_price_features(df, "date").reset_index(drop=True)
    assert out["days_since_price_increase"].tolist() == [_NO_EVENT, _NO_EVENT, 0, 1, 2]
    assert out.loc[2, "price_index_vs_hist"] == pytest.approx(10.99 / 7.99)
