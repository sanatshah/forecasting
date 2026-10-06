"""Dashboard-sized JSON aggregations from pipeline CSV artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..evaluation import mae, wape
from ..plotting.datasets import load_dataset
from ..scenario_engine import (
    ACTION_ANNUAL_PLAN_UPSELL,
    ACTION_ENGAGEMENT_PUSH,
    ACTION_HOLD_PRICE,
    ACTION_MONITOR,
    ACTION_PROCEED_PRICE_CHANGE,
    ACTION_RETENTION_OFFER,
)
from ..utils import get_targets

MAX_RECOMMENDATIONS = 500

ACTIONS = [
    ACTION_RETENTION_OFFER,
    ACTION_ENGAGEMENT_PUSH,
    ACTION_ANNUAL_PLAN_UPSELL,
    ACTION_PROCEED_PRICE_CHANGE,
    ACTION_HOLD_PRICE,
    ACTION_MONITOR,
]


def snapshot_date(df: pd.DataFrame, col: str = "date") -> str:
    if col not in df.columns or df[col].isna().all():
        return "unknown"
    return pd.to_datetime(df[col]).max().strftime("%Y-%m-%d")


def _fmt_date(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _primary_recs(config: Dict[str, Any]) -> pd.DataFrame:
    recs = load_dataset(config, "recommendations")
    primary = config["forecast"]["primary_horizon"]
    return recs[recs["forecast_horizon"] == primary].copy()


def _best_models(metrics: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Best model and overall WAPE per target."""
    overall = metrics[metrics["level"] == "overall"]
    out: Dict[str, Dict[str, Any]] = {}
    for target, g in overall.groupby("target"):
        best = g.sort_values("wape").iloc[0]
        out[str(target)] = {"model": str(best["model"]), "wape": float(best["wape"])}
    return out


def _okr_record(row: pd.Series) -> Dict[str, Any]:
    return {
        "horizon": int(row["forecast_horizon"]),
        "scenario": str(row["scenario"]),
        "forecastStart": _fmt_date(row.get("forecast_start")),
        "forecastEnd": _fmt_date(row.get("forecast_end")),
        "highValueNetAdds": float(row["high_value_net_adds"]),
        "highValuePaidSubsEnd": float(row["high_value_paid_subs_end"]),
        "highValueGrossAdds": float(row["high_value_gross_adds"]),
        "totalNetAdds": float(row["total_net_adds"]),
        "totalPaidSubsEnd": float(row["total_paid_subs_end"]),
        "hoursPerPaidSubMonth": float(row["hours_per_paid_sub_month"]),
        "highValueHoursPerPaidSubMonth": float(row["high_value_hours_per_paid_sub_month"]),
        "revenue": float(row["revenue"]),
    }


def action_breakdown(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recommended actions by tier at the primary horizon."""
    df = _primary_recs(config)
    tiers = sorted(df["tier"].dropna().unique().tolist())

    breakdown: Dict[str, Dict[str, int]] = {}
    for tier in tiers:
        counts = df[df["tier"] == tier].groupby("recommended_action").size()
        breakdown[tier] = {a: int(counts.get(a, 0)) for a in ACTIONS}
    totals = {a: sum(breakdown[t][a] for t in tiers) for a in ACTIONS}

    return {
        "meta": {
            "recipe": "action-breakdown",
            "source": str(Path(config["paths"]["recommendations_csv"]).resolve()),
            "snapshotDate": snapshot_date(df),
            "horizon": int(config["forecast"]["primary_horizon"]),
            "grandTotal": int(len(df)),
        },
        "tiers": tiers,
        "actions": ACTIONS,
        "breakdown": breakdown,
        "totals": totals,
    }


def executive_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """OKR KPIs at the primary horizon plus model quality and risk counts."""
    metrics = load_dataset(config, "metrics")
    recs = _primary_recs(config)
    okr = load_dataset(config, "okr_summary")
    primary = int(config["forecast"]["primary_horizon"])

    best = _best_models(metrics)
    lead_target = get_targets(config)[0]
    lead = best.get(lead_target, next(iter(best.values()), {"model": "unknown", "wape": None}))

    okr_h = okr[okr["forecast_horizon"] == primary]
    baseline = okr_h[okr_h["scenario"] == "baseline"]
    scenario = okr_h[okr_h["scenario"] == "price_change"]

    return {
        "meta": {
            "recipe": "executive-summary",
            "metricsSource": str(Path(config["paths"]["metrics_csv"]).resolve()),
            "recommendationsSource": str(Path(config["paths"]["recommendations_csv"]).resolve()),
            "snapshotDate": snapshot_date(recs),
            "horizon": primary,
            "segmentCount": int(len(recs)),
        },
        "bestModel": lead["model"],
        "bestWape": lead["wape"],
        "bestModels": best,
        "okr": _okr_record(baseline.iloc[0]) if len(baseline) else None,
        "okrScenario": _okr_record(scenario.iloc[0]) if len(scenario) else None,
        "riskCounts": {str(k): int(v) for k, v in recs.groupby("risk_flag").size().items()},
        "actionCounts": {
            str(k): int(v) for k, v in recs.groupby("recommended_action").size().items()
        },
    }


def okr_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """Every horizon x scenario row of the Growth OKR rollup."""
    okr = load_dataset(config, "okr_summary")
    okr_cfg = config.get("okr", {})
    return {
        "meta": {
            "recipe": "okr",
            "source": str(Path(config["paths"]["okr_summary_csv"]).resolve()),
            "highValueTiers": list(okr_cfg.get("high_value_tiers", [])),
            "highValueChannels": list(okr_cfg.get("high_value_channels", [])),
            "priceChanges": list(config["scenarios"].get("price_changes", []) or []),
        },
        "rows": [_okr_record(r) for _, r in okr.sort_values(["forecast_horizon", "scenario"]).iterrows()],
    }


def _daily_series(sub: pd.DataFrame, col: str, dates: List[str]) -> List[float]:
    if col not in sub.columns:
        return [0.0] * len(dates)
    s = sub.groupby("date")[col].sum().reindex(pd.to_datetime(dates)).fillna(0)
    return [round(float(v), 4) for v in s.tolist()]


def segment_forecasts(config: Dict[str, Any]) -> Dict[str, Any]:
    """Forward daily KPIs per segment (tier x acquisition channel)."""
    df = load_dataset(config, "forecasts").copy()
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].dt.strftime("%Y-%m-%d").unique().tolist())

    segments: List[Dict[str, Any]] = []
    for sid, sub in df.groupby("series_id"):
        meta_row = sub.iloc[0]
        net_adds = _daily_series(sub, "forecast_net_adds", dates)
        paid_subs = _daily_series(sub, "forecast_paid_subs", dates)
        segments.append(
            {
                "segmentId": str(sid),
                "tier": str(meta_row.get("tier", "")),
                "channel": str(meta_row.get("acquisition_channel", "")),
                "partner": str(meta_row.get("distribution_partner", "")),
                "openingPaidSubs": float(meta_row.get("opening_paid_subs", 0)),
                "grossAdds": _daily_series(sub, "forecast_gross_adds", dates),
                "churnedSubs": _daily_series(sub, "forecast_churned_subs", dates),
                "netAdds": net_adds,
                "paidSubs": paid_subs,
                "hoursPerPaidSub": _daily_series(sub, "forecast_hours_per_paid_sub", dates),
                "totalNetAdds": round(sum(net_adds), 1),
                "endingPaidSubs": paid_subs[-1] if paid_subs else 0.0,
            }
        )
    segments.sort(key=lambda s: s["endingPaidSubs"], reverse=True)

    return {
        "meta": {
            "recipe": "segment-forecasts",
            "source": str(Path(config["paths"]["forecasts_csv"]).resolve()),
            "snapshotDate": snapshot_date(df),
            "horizonDays": len(dates),
            "forecastStart": dates[0] if dates else None,
            "forecastEnd": dates[-1] if dates else None,
        },
        "dates": dates,
        "segments": segments,
    }


def holdout_forecasts(config: Dict[str, Any], segment: str, target: str) -> Dict[str, Any]:
    """Holdout actual vs predicted for one segment and target."""
    df = load_dataset(config, "holdout_predictions").copy()
    df["date"] = pd.to_datetime(df["date"])

    sub = df[df["series_id"].astype(str) == str(segment)]
    if "target" in sub.columns:
        sub = sub[sub["target"] == target]
    if sub.empty:
        raise KeyError(f"Segment {segment!r} / target {target!r} not found in holdout predictions")

    meta_row = sub.iloc[0]
    daily = sub.groupby("date")[["actual", "forecast"]].sum().sort_index()
    dates = daily.index.strftime("%Y-%m-%d").tolist()
    actual_arr = daily["actual"].to_numpy(dtype="float64")
    forecast_arr = daily["forecast"].to_numpy(dtype="float64")
    wape_val = wape(actual_arr, forecast_arr)

    return {
        "meta": {
            "recipe": "holdout-forecasts",
            "source": str(Path(config["paths"]["holdout_predictions_csv"]).resolve()),
            "segmentId": str(segment),
            "target": target,
            "snapshotDate": dates[-1] if dates else None,
            "holdoutStart": dates[0] if dates else None,
            "holdoutEnd": dates[-1] if dates else None,
            "horizonDays": len(dates),
            "tier": str(meta_row.get("tier", "")),
            "channel": str(meta_row.get("acquisition_channel", "")),
        },
        "dates": dates,
        "actuals": daily["actual"].astype(float).tolist(),
        "predictions": daily["forecast"].astype(float).tolist(),
        "metrics": {
            "mae": round(mae(actual_arr, forecast_arr), 4),
            "wape": round(wape_val, 6) if not pd.isna(wape_val) else None,
        },
    }


def recommendations_list(
    config: Dict[str, Any],
    *,
    risk: Optional[str] = None,
    action: Optional[str] = None,
    tier: Optional[str] = None,
    horizon: Optional[int] = None,
    limit: int = MAX_RECOMMENDATIONS,
) -> Dict[str, Any]:
    """Filterable segment scenario table."""
    recs_full = load_dataset(config, "recommendations")
    h = int(horizon or config["forecast"]["primary_horizon"])
    df = recs_full[recs_full["forecast_horizon"] == h].copy()

    if risk:
        df = df[df["risk_flag"] == risk]
    if action:
        df = df[df["recommended_action"] == action]
    if tier:
        df = df[df["tier"] == tier]

    df = df.sort_values(["risk_flag", "tier", "acquisition_channel"]).head(limit)

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "date": _fmt_date(row["date"]),
                "segmentId": str(row["series_id"]),
                "tier": str(row["tier"]),
                "channel": str(row["acquisition_channel"]),
                "partner": str(row.get("distribution_partner", "")),
                "forecastHorizon": int(row["forecast_horizon"]),
                "openingPaidSubs": float(row["opening_paid_subs"]),
                "grossAdds": float(row["forecast_gross_adds"]),
                "churnedSubs": float(row["forecast_churned_subs"]),
                "netAdds": float(row["forecast_net_adds"]),
                "endingPaidSubs": float(row["ending_paid_subs"]),
                "hoursPerPaidSubMonth": float(row["hours_per_paid_sub_month"]),
                "churnRate": float(row["forecast_churn_rate"]),
                "trailingChurnRate": float(row["trailing_churn_rate"]),
                "usageChangePct": float(row["usage_change_pct"]),
                "listPrice": float(row["list_price"]),
                "scenarioPrice": float(row["scenario_new_price"]),
                "scenarioNetAdds": float(row["scenario_net_adds"]),
                "netAddsDelta": float(row["net_adds_delta"]),
                "baselineRevenue": float(row["baseline_revenue"]),
                "scenarioRevenue": float(row["scenario_revenue"]),
                "revenueDelta": float(row["revenue_delta"]),
                "riskFlag": str(row["risk_flag"]),
                "recommendedAction": str(row["recommended_action"]),
                "priceDecision": str(row["price_decision"]),
                "objectiveScore": float(row["objective_score"]),
                "reasonCode": str(row["reason_code"]),
                "explanation": str(row.get("explanation", "")),
            }
        )

    return {
        "meta": {
            "snapshotDate": snapshot_date(recs_full),
            "horizon": h,
            "totalMatching": int(len(df)),
            "returned": len(rows),
            "filters": {"risk": risk, "action": action, "tier": tier},
        },
        "filters": {
            "tiers": sorted(recs_full["tier"].dropna().unique().tolist()),
            "riskFlags": sorted(recs_full["risk_flag"].dropna().unique().tolist()),
            "actions": sorted(recs_full["recommended_action"].dropna().unique().tolist()),
            "horizons": sorted(int(x) for x in recs_full["forecast_horizon"].unique()),
        },
        "rows": rows,
    }


def segment_metrics(config: Dict[str, Any], target: Optional[str] = None) -> Dict[str, Any]:
    """Best-model accuracy by tier and acquisition channel for one target."""
    metrics = load_dataset(config, "metrics")
    targets = sorted(metrics["target"].dropna().unique().tolist())
    target = target or get_targets(config)[0]
    if target not in targets:
        raise KeyError(f"Target {target!r} not found in metrics. Available: {targets}")

    best = _best_models(metrics)[target]["model"]
    m = metrics[(metrics["target"] == target) & (metrics["model"] == best)]

    def _rows(level: str) -> List[Dict[str, Any]]:
        sub = m[m["level"] == level].sort_values("wape")
        return [
            {
                "group": str(r["group"]),
                "wape": float(r["wape"]),
                "mape": float(r["mape"]),
                "mae": float(r["mae"]),
                "rmse": float(r["rmse"]),
                "bias": float(r["bias"]),
                "n": int(r["n"]),
            }
            for _, r in sub.iterrows()
        ]

    return {
        "meta": {
            "snapshotDate": snapshot_date(load_dataset(config, "holdout_predictions")),
            "model": best,
            "target": target,
            "targets": targets,
            "source": str(Path(config["paths"]["metrics_csv"]).resolve()),
        },
        "tiers": _rows("tier"),
        "channels": _rows("acquisition_channel"),
    }
