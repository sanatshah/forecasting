"""Tests for the optimization engine logic and guardrails."""
from __future__ import annotations

import pandas as pd

from src.optimization_engine import (
    ACTION_REDUCE,
    ACTION_REPLENISH,
    REASON_HIGH_STOCK_LOW_DEMAND,
    REASON_STOCKOUT_RISK,
    generate_recommendations,
    get_elasticity,
    objective_score,
    unit_cost,
    weeks_of_supply,
)


def test_weeks_of_supply_math():
    """70 units over 7 days = 10/day = 70/week; 140 on hand -> 2 weeks."""
    assert weeks_of_supply(140, 70, 7) == 2.0


def test_weeks_of_supply_no_demand_is_high():
    """Zero forecast demand should yield an effectively infinite supply."""
    assert weeks_of_supply(100, 0, 28) >= 99.0


def test_elasticity_lookup_precedence(config):
    """Department elasticity should override the global default."""
    default = config["optimization"]["elasticity"]["default"]
    dept_val = get_elasticity(config, "Womens Apparel", "AnyClass")
    assert dept_val == config["optimization"]["elasticity"]["by_department"]["Womens Apparel"]
    assert get_elasticity(config, "Unknown Dept", "Unknown") == default


def test_unit_cost_from_margin(config):
    """Unit cost should equal price * (1 - assumed gross margin)."""
    gm = config["optimization"]["assumed_gross_margin"]
    assert unit_cost(100.0, config) == 100.0 * (1 - gm)


def test_objective_penalizes_overstock(config):
    """Higher weeks-of-supply beyond the healthy band should lower the score."""
    healthy = objective_score(100.0, config["optimization"]["wos_healthy_high"], config)
    overstocked = objective_score(100.0, config["optimization"]["wos_overstock_threshold"] + 5, config)
    assert overstocked < healthy


def _forecast_row(units, inv, dept="Womens Apparel", horizon=7):
    return {
        "series_id": "SKU1|LOC1|store",
        "sku_id": "SKU1",
        "location_id": "LOC1",
        "channel": "store",
        "department": dept,
        "class": "ClassA",
        "forecast_units": units,
        "forecast_horizon": horizon,
    }


def test_stockout_recommendation(config):
    """Very low inventory vs strong demand should flag stockout + replenish."""
    forecasts = pd.DataFrame([_forecast_row(units=140, inv=5)])
    state = pd.DataFrame(
        [{
            "series_id": "SKU1|LOC1|store",
            "inventory_on_hand": 5,
            "inventory_in_transit": 0,
            "regular_price": 20.0,
            "selling_price": 20.0,
            "date": pd.Timestamp("2024-03-01"),
        }]
    )
    recs = generate_recommendations(forecasts, state, config)
    row = recs.iloc[0]
    assert row["risk_flag"] == "STOCKOUT"
    assert row["recommended_action"] == ACTION_REPLENISH
    assert row["reason_code"] == REASON_STOCKOUT_RISK


def test_overstock_recommendation(config):
    """High inventory vs weak demand should flag overstock + reduce exposure."""
    forecasts = pd.DataFrame([_forecast_row(units=7, inv=1000)])
    state = pd.DataFrame(
        [{
            "series_id": "SKU1|LOC1|store",
            "inventory_on_hand": 1000,
            "inventory_in_transit": 0,
            "regular_price": 20.0,
            "selling_price": 20.0,
            "date": pd.Timestamp("2024-03-01"),
        }]
    )
    recs = generate_recommendations(forecasts, state, config)
    row = recs.iloc[0]
    assert row["risk_flag"] == "OVERSTOCK"
    assert row["recommended_action"] == ACTION_REDUCE
    assert row["reason_code"] == REASON_HIGH_STOCK_LOW_DEMAND


def test_markdown_respects_guardrails(config):
    """Recommended markdown must never exceed the configured maximum depth."""
    forecasts = pd.DataFrame([_forecast_row(units=7, inv=1000)])
    state = pd.DataFrame(
        [{
            "series_id": "SKU1|LOC1|store",
            "inventory_on_hand": 1000,
            "inventory_in_transit": 0,
            "regular_price": 20.0,
            "selling_price": 20.0,
            "date": pd.Timestamp("2024-03-01"),
        }]
    )
    recs = generate_recommendations(forecasts, state, config)
    assert recs["recommended_markdown_pct"].max() <= config["optimization"]["max_markdown_pct"]
    # Recommended new price implied by markdown should keep margin over cost.
    assert recs["expected_margin"].iloc[0] >= 0
