"""Optimization engine: turn forecasts into retail business recommendations.

Given horizon-level demand forecasts plus current inventory and price, this
module produces, per series and horizon:

* an **inventory action** (replenish / hold / transfer / reduce exposure) with
  stockout- and overstock-risk flags derived from weeks-of-supply;
* a **markdown / pricing recommendation** using configurable price elasticity
  and hard guardrails (min margin, max markdown depth, no negative price,
  selling <= regular);
* a **reason code** and plain-English explanation;
* expected sales and expected margin under the recommended action.

The scoring uses a simple, transparent objective
(:func:`objective_score`) that rewards margin and penalizes overstock and
stockout risk. It is intentionally structured so it can later be swapped for
``scipy.optimize`` / linear programming without changing the interface.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .utils import get_logger, resolve_path, safe_divide

logger = get_logger(__name__)


# Reason codes (stable identifiers used across outputs and tests).
REASON_HIGH_STOCK_LOW_DEMAND = "HIGH_STOCK_LOW_DEMAND"
REASON_LOW_STOCK_HIGH_DEMAND = "LOW_STOCK_HIGH_DEMAND"
REASON_PROMO_RESPONSE_STRONG = "PROMO_RESPONSE_STRONG"
REASON_PROMO_RESPONSE_WEAK = "PROMO_RESPONSE_WEAK"
REASON_STOCKOUT_RISK = "STOCKOUT_RISK"
REASON_OVERSTOCK_RISK = "OVERSTOCK_RISK"
REASON_NORMAL_DEMAND = "NORMAL_DEMAND"

# Inventory actions.
ACTION_REPLENISH = "REPLENISH"
ACTION_HOLD = "HOLD"
ACTION_TRANSFER = "TRANSFER"
ACTION_REDUCE = "REDUCE_EXPOSURE"


def get_elasticity(config: Dict[str, Any], department: str, klass: str) -> float:
    """Resolve price elasticity for a department/class from config.

    Lookup order: class override -> department -> global default.
    """
    el = config["optimization"]["elasticity"]
    by_class = el.get("by_class", {}) or {}
    by_dept = el.get("by_department", {}) or {}
    if klass in by_class:
        return float(by_class[klass])
    if department in by_dept:
        return float(by_dept[department])
    return float(el["default"])


def unit_cost(regular_price: float, config: Dict[str, Any]) -> float:
    """Estimate unit cost from regular price and an assumed gross margin."""
    gm = config["optimization"]["assumed_gross_margin"]
    return float(regular_price) * (1.0 - gm)


def weeks_of_supply(inventory_on_hand: float, forecast_units: float, horizon_days: int) -> float:
    """Compute weeks of supply from horizon demand and current inventory."""
    daily = safe_divide(forecast_units, float(horizon_days), fill=0.0)
    weekly = float(daily) * 7.0
    if weekly <= 0:
        return 99.0  # effectively infinite supply if no demand expected
    return float(inventory_on_hand) / weekly


def _demand_at_markdown(
    base_units: float,
    base_price: float,
    new_price: float,
    elasticity: float,
) -> float:
    """Project demand at a new price using constant-elasticity assumption.

    demand_new = demand_base * (price_new / price_base) ** elasticity
    With elasticity < 0, lowering price increases demand.
    """
    if base_price <= 0 or new_price <= 0:
        return base_units
    ratio = new_price / base_price
    return float(base_units) * (ratio ** elasticity)


def objective_score(
    expected_margin: float,
    weeks_supply: float,
    config: Dict[str, Any],
) -> float:
    """Transparent objective balancing margin against inventory risk.

    score = w_margin * margin
            - w_over * overstock_excess
            - w_stock * stockout_shortfall

    Overstock/stockout terms are measured in weeks-of-supply distance from the
    healthy band. Higher is better. This is the hook to replace with an LP.
    """
    w = config["optimization"]["objective_weights"]
    low = config["optimization"]["wos_healthy_low"]
    high = config["optimization"]["wos_healthy_high"]
    overstock_excess = max(0.0, weeks_supply - high)
    stockout_shortfall = max(0.0, low - weeks_supply)
    return (
        w["margin"] * expected_margin
        - w["overstock_penalty"] * overstock_excess
        - w["stockout_penalty"] * stockout_shortfall
    )


def _choose_markdown(
    row: pd.Series,
    wos: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Search candidate markdown depths and pick the objective-maximizing one.

    Respects guardrails: markdown within [current, max], resulting selling price
    >= cost * (1 + min_margin) and never negative, selling <= regular.
    """
    opt = config["optimization"]
    regular = float(row.get("regular_price", 0.0) or 0.0)
    current_selling = float(row.get("selling_price", regular) or regular)
    base_units = float(row["forecast_units"])
    horizon_days = int(row["forecast_horizon"])
    department = str(row.get("department", ""))
    klass = str(row.get("class", ""))
    elasticity = get_elasticity(config, department, klass)
    cost = unit_cost(regular, config)

    # Guardrail: minimum selling price that preserves min margin over cost.
    min_price = max(cost * (1.0 + opt["min_margin_pct"]), 0.0)

    current_md = 0.0 if regular <= 0 else max(0.0, 1.0 - current_selling / regular)
    max_md = opt["max_markdown_pct"]
    step = opt["markdown_step"]

    candidates = np.arange(current_md, max_md + 1e-9, step)
    best = None
    for md in candidates:
        new_price = regular * (1.0 - md)
        if new_price < min_price:
            continue  # violates margin guardrail
        if new_price > regular:
            continue  # never above regular
        exp_units = _demand_at_markdown(base_units, current_selling, new_price, elasticity)
        # Cap expected sales by available inventory (can't sell what we don't have).
        inv = float(row.get("inventory_on_hand", exp_units) or 0.0)
        exp_sales_units = min(exp_units, inv) if inv > 0 else exp_units
        exp_margin = (new_price - cost) * exp_sales_units
        new_wos = weeks_of_supply(inv, exp_units, horizon_days)
        score = objective_score(exp_margin, new_wos, config)
        cand = {
            "markdown_pct": float(md),
            "new_price": float(new_price),
            "expected_units": float(exp_units),
            "expected_sales_units": float(exp_sales_units),
            "expected_margin": float(exp_margin),
            "score": float(score),
        }
        if best is None or cand["score"] > best["score"]:
            best = cand

    if best is None:  # all violated guardrails; hold current price
        exp_margin = (current_selling - cost) * base_units
        best = {
            "markdown_pct": float(current_md),
            "new_price": float(current_selling),
            "expected_units": float(base_units),
            "expected_sales_units": float(base_units),
            "expected_margin": float(exp_margin),
            "score": objective_score(exp_margin, wos, config),
        }
    return best


def _classify(row: pd.Series, wos: float, config: Dict[str, Any]) -> Dict[str, str]:
    """Assign inventory action, risk flag and reason code from weeks of supply."""
    opt = config["optimization"]
    stock_thr = opt["wos_stockout_threshold"]
    over_thr = opt["wos_overstock_threshold"]

    if wos <= stock_thr:
        return {
            "risk_flag": "STOCKOUT",
            "recommended_action": ACTION_REPLENISH,
            "reason_code": REASON_STOCKOUT_RISK,
        }
    if wos >= over_thr:
        return {
            "risk_flag": "OVERSTOCK",
            "recommended_action": ACTION_REDUCE,
            "reason_code": REASON_HIGH_STOCK_LOW_DEMAND,
        }
    # In-band: shade toward the nearer edge.
    if wos < opt["wos_healthy_low"]:
        return {
            "risk_flag": "WATCH_LOW",
            "recommended_action": ACTION_HOLD,
            "reason_code": REASON_LOW_STOCK_HIGH_DEMAND,
        }
    if wos > opt["wos_healthy_high"]:
        return {
            "risk_flag": "WATCH_HIGH",
            "recommended_action": ACTION_TRANSFER,
            "reason_code": REASON_OVERSTOCK_RISK,
        }
    return {
        "risk_flag": "OK",
        "recommended_action": ACTION_HOLD,
        "reason_code": REASON_NORMAL_DEMAND,
    }


def _explain(row: pd.Series, wos: float, decision: Dict[str, Any]) -> str:
    """Generate a plain-English explanation for the recommendation."""
    return (
        f"Forecast {row['forecast_units']:.0f} units over {int(row['forecast_horizon'])} "
        f"days with {float(row.get('inventory_on_hand', 0)):.0f} on hand "
        f"(~{wos:.1f} weeks of supply). "
        f"Recommended to {decision['recommended_action'].replace('_', ' ').lower()} "
        f"and set markdown to {decision['recommended_markdown_pct']*100:.0f}% "
        f"(reason: {decision['reason_code']}). "
        f"Expected sales {decision['expected_sales']:.0f} units, "
        f"expected margin {decision['expected_margin']:.0f}."
    )


def generate_recommendations(
    horizon_forecasts: pd.DataFrame,
    latest_state: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Produce the full recommendation table from horizon forecasts.

    Parameters
    ----------
    horizon_forecasts:
        Output of :func:`src.forecasting_pipeline.horizon_rollups` - one row per
        series per horizon with ``forecast_units``.
    latest_state:
        The most recent observed row per series (inventory, prices, hierarchy).
    """
    date_col = config["data"]["date_col"]

    # Merge current inventory/price state onto each forecast row.
    state_cols = [
        "series_id",
        "inventory_on_hand",
        "inventory_in_transit",
        "regular_price",
        "selling_price",
        date_col,
    ]
    state = latest_state[[c for c in state_cols if c in latest_state.columns]].copy()
    merged = horizon_forecasts.merge(state, on="series_id", how="left", suffixes=("", "_state"))

    records: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        inv = float(row.get("inventory_on_hand", 0.0) or 0.0)
        wos = weeks_of_supply(inv, float(row["forecast_units"]), int(row["forecast_horizon"]))
        cls = _classify(row, wos, config)
        md = _choose_markdown(row, wos, config)

        decision = {
            "recommended_action": cls["recommended_action"],
            "reason_code": cls["reason_code"],
            "recommended_markdown_pct": md["markdown_pct"],
            "expected_sales": md["expected_sales_units"],
            "expected_margin": md["expected_margin"],
        }
        explanation = _explain(row, wos, decision)

        records.append(
            {
                "date": row.get(date_col),
                "sku_id": row.get("sku_id"),
                "location_id": row.get("location_id"),
                "channel": row.get("channel"),
                "department": row.get("department"),
                "forecast_horizon": int(row["forecast_horizon"]),
                "forecast_units": round(float(row["forecast_units"]), 2),
                "inventory_on_hand": inv,
                "weeks_of_supply": round(wos, 2),
                "risk_flag": cls["risk_flag"],
                "recommended_action": cls["recommended_action"],
                "recommended_markdown_pct": round(md["markdown_pct"], 3),
                "expected_sales": round(md["expected_sales_units"], 2),
                "expected_margin": round(md["expected_margin"], 2),
                "objective_score": round(md["score"], 2),
                "reason_code": cls["reason_code"],
                "explanation": explanation,
            }
        )

    recs = pd.DataFrame.from_records(records)
    logger.info("Generated %d recommendations", len(recs))
    return recs


def save_recommendations(recs: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Persist the recommendation table to ``outputs/recommendations.csv``."""
    path = resolve_path(config["paths"]["recommendations_csv"])
    path.parent.mkdir(parents=True, exist_ok=True)
    recs.to_csv(path, index=False)
    logger.info("Saved recommendations to %s", path)
