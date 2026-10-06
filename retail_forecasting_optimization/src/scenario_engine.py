"""Scenario engine: turn subscriber forecasts into segment decisions and OKRs.

Replaces the retail inventory / markdown optimizer. For each segment
(tier x acquisition channel) and horizon it produces:

* the **baseline** outlook - gross adds, churn, net adds, ending paid subs,
  hours per paid sub, and revenue;
* a **price-change scenario** for any planned list-price change on the tier,
  using constant-elasticity responses for churn and acquisition;
* a **risk flag**, **recommended action**, **price decision**, reason code and
  plain-English explanation;
* a transparent ``objective_score`` (revenue minus a churn penalty).

:func:`build_okr_summary` rolls the segment rows up into the two Growth OKRs:
high-value subscriber growth and usage per paid sub.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from .utils import get_logger, resolve_path, safe_divide

logger = get_logger(__name__)

# Risk flags (in priority order: the first that applies wins).
RISK_NEGATIVE_NET_ADDS = "NEGATIVE_NET_ADDS"
RISK_CHURN_SPIKE = "CHURN_SPIKE"
RISK_TENTPOLE_CLIFF = "TENTPOLE_CLIFF"
RISK_USAGE_DECLINE = "USAGE_DECLINE"
RISK_PRICE_SENSITIVE = "PRICE_SENSITIVE"
RISK_OK = "OK"

# Recommended actions.
ACTION_RETENTION_OFFER = "RETENTION_OFFER"
ACTION_ENGAGEMENT_PUSH = "ENGAGEMENT_PUSH"
ACTION_ANNUAL_PLAN_UPSELL = "ANNUAL_PLAN_UPSELL"
ACTION_PROCEED_PRICE_CHANGE = "PROCEED_PRICE_CHANGE"
ACTION_HOLD_PRICE = "HOLD_PRICE"
ACTION_MONITOR = "MONITOR"

# Price decisions (independent of the risk-driven action).
PRICE_PROCEED = "PROCEED_PRICE_CHANGE"
PRICE_HOLD = "HOLD_PRICE"
PRICE_NONE = "NO_CHANGE_PLANNED"

# Reason codes.
REASON_ADDS_BELOW_CHURN = "ADDS_BELOW_CHURN"
REASON_CHURN_ABOVE_TREND = "CHURN_ABOVE_TREND"
REASON_POST_TENTPOLE_CHURN = "POST_TENTPOLE_CHURN"
REASON_ENGAGEMENT_SOFTENING = "ENGAGEMENT_SOFTENING"
REASON_PRICE_ELASTIC_SEGMENT = "PRICE_ELASTIC_SEGMENT"
REASON_PRICE_INCREASE_ACCRETIVE = "PRICE_INCREASE_ACCRETIVE"
REASON_STABLE_GROWTH = "STABLE_GROWTH"

_FLAG_TO_ACTION = {
    RISK_NEGATIVE_NET_ADDS: (ACTION_RETENTION_OFFER, REASON_ADDS_BELOW_CHURN),
    RISK_CHURN_SPIKE: (ACTION_RETENTION_OFFER, REASON_CHURN_ABOVE_TREND),
    RISK_TENTPOLE_CLIFF: (ACTION_ANNUAL_PLAN_UPSELL, REASON_POST_TENTPOLE_CHURN),
    RISK_USAGE_DECLINE: (ACTION_ENGAGEMENT_PUSH, REASON_ENGAGEMENT_SOFTENING),
    RISK_PRICE_SENSITIVE: (ACTION_HOLD_PRICE, REASON_PRICE_ELASTIC_SEGMENT),
}


# ---------------------------------------------------------------------------
# Elasticity and price-change math
# ---------------------------------------------------------------------------
def get_elasticity(config: Dict[str, Any], kind: str, tier: str) -> float:
    """Resolve the ``churn`` or ``acquisition`` elasticity for a tier."""
    el = config["scenarios"]["elasticity"][kind]
    by_tier = el.get("by_tier", {}) or {}
    return float(by_tier.get(tier, el["default"]))


def channel_scale(config: Dict[str, Any], channel: str) -> float:
    """Share of a list-price change felt by subscribers in this channel."""
    scales = config["scenarios"]["elasticity"].get("channel_scale", {}) or {}
    return float(scales.get(channel, 1.0))


def price_response(price_ratio: float, elasticity: float, active_fraction: float = 1.0) -> float:
    """Multiplier on a flow when price changes by ``price_ratio`` (new / old).

    ``ratio ** elasticity`` applied to the share of the horizon on or after the
    effective date; the remaining share is unaffected.
    """
    if price_ratio <= 0:
        return 1.0
    full = price_ratio ** elasticity
    return 1.0 + active_fraction * (full - 1.0)


def find_price_change(config: Dict[str, Any], tier: str) -> Optional[Dict[str, Any]]:
    """Return the first configured price change for ``tier`` (if any)."""
    for change in config["scenarios"].get("price_changes", []) or []:
        if change.get("tier") == tier:
            return change
    return None


def active_fraction(start: pd.Timestamp, end: pd.Timestamp, effective: pd.Timestamp) -> float:
    """Share of days in ``[start, end]`` on or after ``effective``."""
    total = (end - start).days + 1
    if total <= 0 or effective > end:
        return 0.0
    active = (end - max(start, effective)).days + 1
    return max(0.0, min(1.0, active / total))


def objective_score(revenue: float, churned: float, price: float, config: Dict[str, Any]) -> float:
    """Revenue minus a penalty for churned subs valued at N months of price."""
    w = config["scenarios"]["objective_weights"]
    return w["revenue"] * revenue - w["churn_penalty_months"] * churned * price


# ---------------------------------------------------------------------------
# Trailing baselines and tentpole cliff detection
# ---------------------------------------------------------------------------
def _trailing_rates(history: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Per-series trailing daily churn rate and hours per sub per day."""
    date_col = config["data"]["date_col"]
    risk = config["scenarios"]["risk"]
    rows = []
    for sid, g in history.sort_values(date_col).groupby("series_id"):
        churn_win = g.tail(int(risk["trailing_churn_days"]))
        usage_win = g.tail(int(risk["trailing_usage_days"]))
        rows.append(
            {
                "series_id": sid,
                "trailing_churn_rate": float(
                    safe_divide(churn_win["churned_subs"].sum(), churn_win["paid_subs_bod"].sum())
                ),
                "trailing_hours_per_sub_day": float(
                    safe_divide(usage_win["hours_watched"].sum(), usage_win["paid_subs_bod"].sum())
                ),
                "last_date": g[date_col].max(),
                "list_price": float(g["list_price"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def _big_event_dates(
    calendar: Optional[pd.DataFrame], history: pd.DataFrame, config: Dict[str, Any]
) -> pd.Series:
    """Dates with a tentpole at or above the cliff intensity threshold."""
    date_col = config["data"]["date_col"]
    min_int = config["scenarios"]["risk"]["cliff_min_intensity"]
    source = calendar if calendar is not None and not calendar.empty else history
    if "tentpole_intensity" not in source.columns:
        return pd.Series([], dtype="datetime64[ns]")
    big = source.loc[source["tentpole_intensity"] >= min_int, date_col]
    return pd.Series(pd.to_datetime(big).drop_duplicates().sort_values().to_numpy())


def detect_tentpole_cliff(
    big_dates: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lookback_days: int,
) -> bool:
    """True when a big tentpole just ended (or ends in-window) with none after it.

    * recently ended: a big day within ``lookback_days`` before ``start`` and
      none inside ``[start, end]``; or
    * ends in-window: the last big day in the window falls at least a week
      before ``end``, leaving post-event churn inside the horizon.
    """
    if big_dates.empty:
        return False
    in_window = big_dates[(big_dates >= start) & (big_dates <= end)]
    recent = big_dates[(big_dates < start) & (big_dates >= start - pd.Timedelta(days=lookback_days))]
    if in_window.empty:
        return not recent.empty
    return bool(in_window.max() <= end - pd.Timedelta(days=7))


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
def _classify(
    net_adds: float,
    fc_churn_rate: float,
    trailing_churn_rate: float,
    usage_change_pct: float,
    cliff: bool,
    price_hurts: bool,
    config: Dict[str, Any],
) -> str:
    """Return the highest-priority risk flag that applies."""
    risk = config["scenarios"]["risk"]
    if net_adds < 0:
        return RISK_NEGATIVE_NET_ADDS
    if trailing_churn_rate > 0 and fc_churn_rate > risk["churn_spike_multiple"] * trailing_churn_rate:
        return RISK_CHURN_SPIKE
    if cliff:
        return RISK_TENTPOLE_CLIFF
    if usage_change_pct < -risk["usage_decline_pct"]:
        return RISK_USAGE_DECLINE
    if price_hurts:
        return RISK_PRICE_SENSITIVE
    return RISK_OK


def _explain(r: Dict[str, Any]) -> str:
    """Plain-English explanation for one segment/horizon decision."""
    seg = f"{r['tier']} via {r['acquisition_channel']}"
    text = (
        f"{seg}: over {r['forecast_horizon']} days expect {r['forecast_gross_adds']:,.0f} adds and "
        f"{r['forecast_churned_subs']:,.0f} churn ({r['forecast_net_adds']:+,.0f} net), ending at "
        f"{r['ending_paid_subs']:,.0f} paid subs. Daily churn rate {r['forecast_churn_rate']*100:.3f}% "
        f"vs trailing {r['trailing_churn_rate']*100:.3f}%; usage {r['hours_per_paid_sub_month']:.1f} "
        f"hrs/sub/month ({r['usage_change_pct']*100:+.1f}% vs trailing)."
    )
    if r["price_decision"] != PRICE_NONE:
        text += (
            f" Price ${r['list_price']:.2f} -> ${r['scenario_new_price']:.2f} changes net adds by "
            f"{r['net_adds_delta']:+,.0f} and revenue by {r['revenue_delta']:+,.0f}"
            f" ({r['price_decision'].replace('_', ' ').lower()})."
        )
    text += (
        f" Recommended: {r['recommended_action'].replace('_', ' ').lower()} "
        f"(reason: {r['reason_code']})."
    )
    return text


def generate_recommendations(
    horizon_rollups: pd.DataFrame,
    history: pd.DataFrame,
    config: Dict[str, Any],
    calendar: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Produce the segment decision table from horizon rollups.

    Parameters
    ----------
    horizon_rollups:
        Output of :func:`src.forecasting_pipeline.horizon_rollups` - one row
        per series per horizon.
    history:
        Cleaned history (for trailing churn and usage baselines and prices).
    calendar:
        Optional content calendar for tentpole-cliff detection.
    """
    trailing = _trailing_rates(history, config)
    merged = horizon_rollups.merge(trailing, on="series_id", how="left")
    big_dates = _big_event_dates(calendar, history, config)
    lookback = int(config["scenarios"]["risk"]["cliff_lookback_days"])

    records: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        h = int(row["forecast_horizon"])
        tier = str(row["tier"])
        channel = str(row["acquisition_channel"])
        start = pd.Timestamp(row["forecast_start"])
        end = pd.Timestamp(row["forecast_end"])
        price = float(row["list_price"])

        adds = float(row.get("forecast_gross_adds", 0.0))
        churn = float(row.get("forecast_churned_subs", 0.0))
        net = adds - churn
        opening = float(row["opening_paid_subs"])
        ending = float(row["ending_paid_subs"])
        avg_subs = float(row["avg_paid_subs"])
        hours = float(row.get("forecast_hours_watched", 0.0))

        fc_churn_rate = float(safe_divide(churn, avg_subs * h))
        fc_hours_day = float(safe_divide(hours, avg_subs * h))
        trailing_hours = float(row["trailing_hours_per_sub_day"])
        usage_change = float(safe_divide(fc_hours_day - trailing_hours, trailing_hours))

        baseline_revenue = avg_subs * price * h / 30.0
        baseline_score = objective_score(baseline_revenue, churn, price, config)

        change = find_price_change(config, tier)
        new_price = price
        scen_adds, scen_churn = adds, churn
        scen_revenue, scen_score = baseline_revenue, baseline_score
        price_decision = PRICE_NONE
        if change is not None:
            new_price = float(change["new_price"])
            frac = active_fraction(start, end, pd.Timestamp(change["effective_date"]))
            ratio = new_price / price if price > 0 else 1.0
            scale = channel_scale(config, channel)
            churn_mult = price_response(ratio, get_elasticity(config, "churn", tier) * scale, frac)
            adds_mult = price_response(ratio, get_elasticity(config, "acquisition", tier) * scale, frac)
            scen_adds = adds * adds_mult
            scen_churn = churn * churn_mult
            scen_avg = avg_subs + 0.5 * ((scen_adds - scen_churn) - net)
            realized_price = price * (1 - frac) + new_price * frac
            scen_revenue = scen_avg * realized_price * h / 30.0
            scen_score = objective_score(scen_revenue, scen_churn, realized_price, config)
            if frac > 0:
                price_decision = PRICE_PROCEED if scen_score >= baseline_score else PRICE_HOLD

        scen_net = scen_adds - scen_churn
        cliff = detect_tentpole_cliff(big_dates, start, end, lookback)
        flag = _classify(
            net,
            fc_churn_rate,
            float(row["trailing_churn_rate"]),
            usage_change,
            cliff,
            price_decision == PRICE_HOLD,
            config,
        )
        if flag in _FLAG_TO_ACTION:
            action, reason = _FLAG_TO_ACTION[flag]
        elif price_decision == PRICE_PROCEED:
            action, reason = ACTION_PROCEED_PRICE_CHANGE, REASON_PRICE_INCREASE_ACCRETIVE
        else:
            action, reason = ACTION_MONITOR, REASON_STABLE_GROWTH

        chosen_score = scen_score if price_decision == PRICE_PROCEED else baseline_score
        rec = {
            "date": row["last_date"],
            "forecast_start": start,
            "forecast_end": end,
            "series_id": row["series_id"],
            "tier": tier,
            "acquisition_channel": channel,
            "distribution_partner": row.get("distribution_partner"),
            "forecast_horizon": h,
            "opening_paid_subs": round(opening, 0),
            "forecast_gross_adds": round(adds, 1),
            "forecast_churned_subs": round(churn, 1),
            "forecast_net_adds": round(net, 1),
            "ending_paid_subs": round(ending, 0),
            "avg_paid_subs": round(avg_subs, 0),
            "forecast_hours_watched": round(hours, 1),
            "hours_per_paid_sub_month": round(float(row.get("hours_per_paid_sub_month", 0.0)), 3),
            "forecast_churn_rate": round(fc_churn_rate, 6),
            "trailing_churn_rate": round(float(row["trailing_churn_rate"]), 6),
            "usage_change_pct": round(usage_change, 4),
            "tentpole_days": int(row.get("tentpole_days", 0) or 0),
            "list_price": round(price, 2),
            "scenario_new_price": round(new_price, 2),
            "scenario_gross_adds": round(scen_adds, 1),
            "scenario_churned_subs": round(scen_churn, 1),
            "scenario_net_adds": round(scen_net, 1),
            "scenario_ending_paid_subs": round(opening + scen_net, 0),
            "net_adds_delta": round(scen_net - net, 1),
            "baseline_revenue": round(baseline_revenue, 2),
            "scenario_revenue": round(scen_revenue, 2),
            "revenue_delta": round(scen_revenue - baseline_revenue, 2),
            "risk_flag": flag,
            "recommended_action": action,
            "price_decision": price_decision,
            "objective_score": round(chosen_score, 2),
            "reason_code": reason,
        }
        rec["explanation"] = _explain(rec)
        records.append(rec)

    recs = pd.DataFrame.from_records(records)
    logger.info("Generated %d segment recommendations", len(recs))
    return recs


# ---------------------------------------------------------------------------
# Growth OKR rollup
# ---------------------------------------------------------------------------
def _okr_row(sub: pd.DataFrame, hv: pd.Series, h: int, scenario: str) -> Dict[str, Any]:
    if scenario == "baseline":
        net, ending, revenue = "forecast_net_adds", "ending_paid_subs", "baseline_revenue"
        subs_scale = pd.Series(1.0, index=sub.index)
    else:
        net, ending, revenue = "scenario_net_adds", "scenario_ending_paid_subs", "scenario_revenue"
        subs_scale = pd.Series(
            safe_divide(
                sub["avg_paid_subs"] + 0.5 * (sub["scenario_net_adds"] - sub["forecast_net_adds"]),
                sub["avg_paid_subs"],
                fill=1.0,
            ),
            index=sub.index,
        )
    avg_subs = sub["avg_paid_subs"] * subs_scale
    hours = sub["forecast_hours_watched"] * subs_scale
    return {
        "forecast_horizon": h,
        "scenario": scenario,
        "forecast_start": sub["forecast_start"].min(),
        "forecast_end": sub["forecast_end"].max(),
        "high_value_net_adds": round(float(sub.loc[hv, net].sum()), 1),
        "high_value_paid_subs_end": round(float(sub.loc[hv, ending].sum()), 0),
        "high_value_gross_adds": round(float(sub.loc[hv, "forecast_gross_adds"].sum()), 1),
        "total_net_adds": round(float(sub[net].sum()), 1),
        "total_paid_subs_end": round(float(sub[ending].sum()), 0),
        "hours_per_paid_sub_month": round(
            float(safe_divide(hours.sum(), avg_subs.sum())) * 30.0 / h, 3
        ),
        "high_value_hours_per_paid_sub_month": round(
            float(safe_divide(hours[hv].sum(), avg_subs[hv].sum())) * 30.0 / h, 3
        ),
        "revenue": round(float(sub[revenue].sum()), 2),
    }


def build_okr_summary(recs: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Roll segment rows up to the Growth OKRs, baseline vs price change.

    High-value subscribers are the configured tiers sold through the
    configured (directly billed) channels.
    """
    okr_cfg = config.get("okr", {})
    hv_tiers = set(okr_cfg.get("high_value_tiers", []))
    hv_channels = set(okr_cfg.get("high_value_channels", []))
    has_scenario = bool(config["scenarios"].get("price_changes"))

    rows: List[Dict[str, Any]] = []
    for h, sub in recs.groupby("forecast_horizon"):
        hv = sub["tier"].isin(hv_tiers) if hv_tiers else pd.Series(True, index=sub.index)
        if hv_channels:
            hv = hv & sub["acquisition_channel"].isin(hv_channels)
        rows.append(_okr_row(sub, hv, int(h), "baseline"))
        if has_scenario:
            rows.append(_okr_row(sub, hv, int(h), "price_change"))
    return pd.DataFrame(rows)


def save_recommendations(recs: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Persist the segment decision table to ``outputs/recommendations.csv``."""
    path = resolve_path(config["paths"]["recommendations_csv"])
    path.parent.mkdir(parents=True, exist_ok=True)
    recs.to_csv(path, index=False)
    logger.info("Saved recommendations to %s", path)


def save_okr_summary(okr: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Persist the Growth OKR rollup to ``outputs/okr_summary.csv``."""
    path = resolve_path(config["paths"]["okr_summary_csv"])
    path.parent.mkdir(parents=True, exist_ok=True)
    okr.to_csv(path, index=False)
    logger.info("Saved OKR summary to %s", path)
