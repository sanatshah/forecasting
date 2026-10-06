"""Tests for the subscriber scenario engine, multi-target combination and OKRs."""
from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.data_validation import validate_and_clean
from src.forecasting_pipeline import combine_target_forecasts, horizon_rollups
from src.scenario_engine import (
    ACTION_RETENTION_OFFER,
    PRICE_HOLD,
    PRICE_NONE,
    PRICE_PROCEED,
    RISK_CHURN_SPIKE,
    RISK_NEGATIVE_NET_ADDS,
    RISK_OK,
    RISK_TENTPOLE_CLIFF,
    RISK_USAGE_DECLINE,
    _classify,
    active_fraction,
    build_okr_summary,
    channel_scale,
    detect_tentpole_cliff,
    generate_recommendations,
    get_elasticity,
    objective_score,
    price_response,
)

PREMIUM = "Premium|direct"
AD_TIER = "Ad Tier|app_store"
HORIZON = 28


# ---------------------------------------------------------------------------
# Fixtures: synthetic per-target forward forecasts on top of sample_df
# ---------------------------------------------------------------------------
@pytest.fixture()
def cleaned(sample_df, config):
    df, _ = validate_and_clean(sample_df, config)
    return df


def _trailing_hours_per_sub(cleaned: pd.DataFrame, sid: str, config) -> float:
    g = cleaned[cleaned["series_id"] == sid].tail(config["scenarios"]["risk"]["trailing_usage_days"])
    return float(g["hours_watched"].sum() / g["paid_subs_bod"].sum())


def _per_target(cleaned: pd.DataFrame, config, daily: Dict[str, Dict[str, float]]) -> Dict[str, pd.DataFrame]:
    """Constant daily forecasts per series: ``daily[series_id][target] = value``."""
    last = cleaned.sort_values("date").groupby("series_id").tail(1).set_index("series_id")
    out: Dict[str, pd.DataFrame] = {}
    for target in config["data"]["targets"]:
        frames = []
        for sid, values in daily.items():
            row = last.loc[sid]
            dates = pd.date_range(row["date"] + pd.Timedelta(days=1), periods=HORIZON, freq="D")
            frames.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "tier": row["tier"],
                        "acquisition_channel": row["acquisition_channel"],
                        "distribution_partner": row["distribution_partner"],
                        "tentpole_flag": (dates.dayofweek == 6).astype(int),
                        "series_id": sid,
                        "forecast_value": values[target],
                        "model": "seasonal_naive",
                        "horizon_day": np.arange(1, HORIZON + 1),
                    }
                )
            )
        out[target] = pd.concat(frames, ignore_index=True)
    return out


@pytest.fixture()
def scenario_config(config):
    cfg = copy.deepcopy(config)
    cfg["scenarios"]["price_changes"] = [
        {"tier": "Premium", "effective_date": "2024-03-15", "new_price": 9.99}
    ]
    return cfg


@pytest.fixture()
def wide(cleaned, scenario_config):
    ad_opening = float(cleaned[cleaned["series_id"] == AD_TIER]["paid_subs_eod"].iloc[-1])
    pr_opening = float(cleaned[cleaned["series_id"] == PREMIUM]["paid_subs_eod"].iloc[-1])
    daily = {
        PREMIUM: {
            "gross_adds": 100.0,
            "churned_subs": 150.0,
            "hours_watched": _trailing_hours_per_sub(cleaned, PREMIUM, scenario_config) * pr_opening,
        },
        AD_TIER: {
            "gross_adds": 70.0,
            "churned_subs": 50.0,
            "hours_watched": _trailing_hours_per_sub(cleaned, AD_TIER, scenario_config) * ad_opening,
        },
    }
    return combine_target_forecasts(_per_target(cleaned, scenario_config, daily), cleaned, scenario_config)


@pytest.fixture()
def recs(wide, cleaned, scenario_config):
    return generate_recommendations(horizon_rollups(wide, scenario_config), cleaned, scenario_config)


# ---------------------------------------------------------------------------
# Elasticity and price math
# ---------------------------------------------------------------------------
def test_elasticity_lookup_by_tier_and_default(config):
    assert get_elasticity(config, "churn", "Premium Plus") == pytest.approx(1.4)
    assert get_elasticity(config, "acquisition", "Ad Tier") == pytest.approx(-1.3)
    assert get_elasticity(config, "churn", "Unknown Tier") == pytest.approx(
        config["scenarios"]["elasticity"]["churn"]["default"]
    )


def test_channel_scale(config):
    assert channel_scale(config, "mvpd_partner") == pytest.approx(0.3)
    assert channel_scale(config, "unknown_channel") == pytest.approx(1.0)


@pytest.mark.parametrize(
    "ratio, elasticity, frac, expected",
    [
        (1.5, 2.0, 1.0, 2.25),
        (1.5, 2.0, 0.5, 1.625),
        (1.5, 2.0, 0.0, 1.0),
        (1.25, -1.0, 1.0, 0.8),
        (0.0, 2.0, 1.0, 1.0),
    ],
)
def test_price_response(ratio, elasticity, frac, expected):
    assert price_response(ratio, elasticity, frac) == pytest.approx(expected)


def test_active_fraction():
    start, end = pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-28")
    assert active_fraction(start, end, pd.Timestamp("2026-10-15")) == pytest.approx(0.5)
    assert active_fraction(start, end, pd.Timestamp("2026-11-01")) == 0.0
    assert active_fraction(start, end, pd.Timestamp("2026-09-01")) == 1.0


def test_objective_score(config):
    assert objective_score(1000.0, 10.0, 10.0, config) == pytest.approx(700.0)


# ---------------------------------------------------------------------------
# Risk classification and tentpole cliffs
# ---------------------------------------------------------------------------
def test_cliff_after_recent_tentpole_with_nothing_in_window():
    big = pd.Series(pd.to_datetime(["2026-02-08"]))
    assert detect_tentpole_cliff(big, pd.Timestamp("2026-02-15"), pd.Timestamp("2026-03-14"), 30)
    assert not detect_tentpole_cliff(big, pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-28"), 30)


def test_cliff_when_tentpole_ends_early_in_window():
    big = pd.Series(pd.to_datetime(["2026-02-08"]))
    start, end = pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-28")
    assert detect_tentpole_cliff(big, start, end, 30)
    late = pd.Series(pd.to_datetime(["2026-02-08", "2026-02-25"]))
    assert not detect_tentpole_cliff(late, start, end, 30)
    assert not detect_tentpole_cliff(pd.Series([], dtype="datetime64[ns]"), start, end, 30)


def test_classify_priority(config):
    assert _classify(-1, 0.01, 0.001, -0.5, True, True, config) == RISK_NEGATIVE_NET_ADDS
    assert _classify(10, 0.01, 0.001, -0.5, True, True, config) == RISK_CHURN_SPIKE
    assert _classify(10, 0.001, 0.001, -0.5, True, True, config) == RISK_TENTPOLE_CLIFF
    assert _classify(10, 0.001, 0.001, -0.5, False, True, config) == RISK_USAGE_DECLINE
    assert _classify(10, 0.001, 0.001, 0.0, False, False, config) == RISK_OK


# ---------------------------------------------------------------------------
# Multi-target combination and horizon rollups
# ---------------------------------------------------------------------------
def test_combine_derives_net_adds_and_paid_subs(wide, cleaned):
    g = wide[wide["series_id"] == PREMIUM].reset_index(drop=True)
    opening = float(cleaned[cleaned["series_id"] == PREMIUM]["paid_subs_eod"].iloc[-1])
    assert (g["forecast_net_adds"] == -50.0).all()
    assert g["opening_paid_subs"].iloc[0] == pytest.approx(opening)
    assert g["forecast_paid_subs"].iloc[-1] == pytest.approx(opening - 50.0 * HORIZON)
    assert g["forecast_hours_per_paid_sub"].iloc[0] == pytest.approx(
        g["forecast_hours_watched"].iloc[0] / (opening - 50.0), rel=1e-3
    )
    for target in ["gross_adds", "churned_subs", "hours_watched"]:
        assert f"forecast_{target}" in wide.columns
        assert f"model_{target}" in wide.columns


def test_horizon_rollups_sum_flows(wide, scenario_config):
    rollups = horizon_rollups(wide, scenario_config)
    assert len(rollups) == 2 * len(scenario_config["forecast"]["horizons"])
    row = rollups[(rollups["series_id"] == AD_TIER) & (rollups["forecast_horizon"] == 7)].iloc[0]
    assert row["forecast_gross_adds"] == pytest.approx(70.0 * 7)
    assert row["forecast_net_adds"] == pytest.approx(20.0 * 7)
    assert row["ending_paid_subs"] == pytest.approx(row["opening_paid_subs"] + 20.0 * 7)
    assert row["tentpole_days"] == 1


# ---------------------------------------------------------------------------
# Recommendations and price scenario
# ---------------------------------------------------------------------------
def test_negative_net_adds_flagged_for_retention(recs):
    row = recs[(recs["series_id"] == PREMIUM) & (recs["forecast_horizon"] == HORIZON)].iloc[0]
    assert row["risk_flag"] == RISK_NEGATIVE_NET_ADDS
    assert row["recommended_action"] == ACTION_RETENTION_OFFER


def test_stable_segment_without_price_change_is_ok(recs):
    row = recs[(recs["series_id"] == AD_TIER) & (recs["forecast_horizon"] == HORIZON)].iloc[0]
    assert row["risk_flag"] == RISK_OK
    assert row["price_decision"] == PRICE_NONE
    assert row["scenario_net_adds"] == pytest.approx(row["forecast_net_adds"])


def test_price_change_only_applies_after_effective_date(recs):
    short = recs[(recs["series_id"] == PREMIUM) & (recs["forecast_horizon"] == 7)].iloc[0]
    assert short["price_decision"] == PRICE_NONE
    assert short["net_adds_delta"] == pytest.approx(0.0)

    full = recs[(recs["series_id"] == PREMIUM) & (recs["forecast_horizon"] == HORIZON)].iloc[0]
    assert full["scenario_new_price"] == pytest.approx(9.99)
    assert full["scenario_churned_subs"] > full["forecast_churned_subs"]
    assert full["scenario_gross_adds"] < full["forecast_gross_adds"]
    assert full["net_adds_delta"] < 0


def test_price_decision_follows_objective(recs, scenario_config):
    row = recs[(recs["series_id"] == PREMIUM) & (recs["forecast_horizon"] == HORIZON)].iloc[0]
    realized = 0.5 * row["list_price"] + 0.5 * row["scenario_new_price"]
    base = objective_score(row["baseline_revenue"], row["forecast_churned_subs"], row["list_price"], scenario_config)
    scen = objective_score(row["scenario_revenue"], row["scenario_churned_subs"], realized, scenario_config)
    expected = PRICE_PROCEED if scen >= base else PRICE_HOLD
    assert row["price_decision"] == expected
    assert row["explanation"]


# ---------------------------------------------------------------------------
# Growth OKR rollup
# ---------------------------------------------------------------------------
def test_okr_summary_rows_and_high_value_scope(recs, scenario_config):
    okr = build_okr_summary(recs, scenario_config)
    assert len(okr) == 2 * len(scenario_config["forecast"]["horizons"])
    assert set(okr["scenario"]) == {"baseline", "price_change"}

    base = okr[(okr["forecast_horizon"] == HORIZON) & (okr["scenario"] == "baseline")].iloc[0]
    assert base["high_value_net_adds"] == pytest.approx(-50.0 * HORIZON)
    assert base["total_net_adds"] == pytest.approx((-50.0 + 20.0) * HORIZON)
    assert base["high_value_hours_per_paid_sub_month"] > 0

    scen = okr[(okr["forecast_horizon"] == HORIZON) & (okr["scenario"] == "price_change")].iloc[0]
    assert scen["high_value_net_adds"] < base["high_value_net_adds"]


def test_okr_summary_baseline_only_without_price_changes(recs, scenario_config):
    cfg = copy.deepcopy(scenario_config)
    cfg["scenarios"]["price_changes"] = []
    okr = build_okr_summary(recs, cfg)
    assert set(okr["scenario"]) == {"baseline"}
