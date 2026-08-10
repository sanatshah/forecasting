"""Dashboard-sized JSON aggregations from pipeline CSV artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..evaluation import mae, wape
from ..plotting.datasets import load_dataset

MAX_RECOMMENDATIONS = 500


def snapshot_date(df: pd.DataFrame, col: str = "date") -> str:
    if col not in df.columns or df[col].isna().all():
        return "unknown"
    return pd.to_datetime(df[col]).max().strftime("%Y-%m-%d")


def normalize_action(action: str) -> str:
    if action == "REDUCE_EXPOSURE":
        return "REDUCE"
    return action


def action_breakdown(config: Dict[str, Any]) -> Dict[str, Any]:
    df = load_dataset(config, "recommendations")
    df = df.copy()
    action_col = "recommended_action" if "recommended_action" in df.columns else "action"
    df["action"] = df[action_col].map(normalize_action)

    departments = sorted(df["department"].dropna().unique().tolist())
    actions = ["HOLD", "REPLENISH", "REDUCE", "TRANSFER"]

    breakdown: Dict[str, Dict[str, int]] = {}
    for dept in departments:
        sub = df[df["department"] == dept]
        counts = sub.groupby("action").size()
        breakdown[dept] = {a: int(counts.get(a, 0)) for a in actions}

    totals = {a: sum(breakdown[d][a] for d in departments) for a in actions}

    return {
        "meta": {
            "recipe": "action-breakdown",
            "source": str(Path(config["paths"]["recommendations_csv"]).resolve()),
            "snapshotDate": snapshot_date(df),
            "grandTotal": int(len(df)),
        },
        "departments": departments,
        "actions": actions,
        "breakdown": breakdown,
        "totals": totals,
    }


def executive_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    metrics = load_dataset(config, "metrics")
    recs = load_dataset(config, "recommendations")

    overall = metrics[metrics["level"] == "overall"]
    best_row = overall.sort_values("wape").head(1)
    best_model = str(best_row["model"].iloc[0]) if len(best_row) else "unknown"
    best_wape = float(best_row["wape"].iloc[0]) if len(best_row) else None

    risk_counts = recs.groupby("risk_flag").size().to_dict()
    action_col = "recommended_action" if "recommended_action" in recs.columns else "action"
    action_counts = recs.assign(
        action=recs[action_col].map(normalize_action)
    ).groupby("action").size().to_dict()

    return {
        "meta": {
            "recipe": "executive-summary",
            "metricsSource": str(Path(config["paths"]["metrics_csv"]).resolve()),
            "recommendationsSource": str(Path(config["paths"]["recommendations_csv"]).resolve()),
            "snapshotDate": snapshot_date(recs),
            "recommendationCount": int(len(recs)),
        },
        "bestModel": best_model,
        "bestWape": best_wape,
        "riskCounts": {str(k): int(v) for k, v in risk_counts.items()},
        "actionCounts": {str(k): int(v) for k, v in action_counts.items()},
    }


def sku_forecasts(config: Dict[str, Any], top_skus: int) -> Dict[str, Any]:
    df = load_dataset(config, "forecasts")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    sku_totals = (
        df.groupby("sku_id")["forecast_units"]
        .sum()
        .sort_values(ascending=False)
        .head(top_skus)
    )
    top = sku_totals.index.tolist()
    dates = sorted(df["date"].dt.strftime("%Y-%m-%d").unique().tolist())

    skus: List[Dict[str, Any]] = []
    for sku in top:
        sub = df[df["sku_id"] == sku]
        meta_row = sub.iloc[0]
        lifecycle_col = (
            "product_lifecycle_status"
            if "product_lifecycle_status" in sub.columns
            else "lifecycle"
        )
        daily = (
            sub.groupby("date")["forecast_units"]
            .sum()
            .reindex(pd.to_datetime(dates))
            .fillna(0)
            .astype(int)
            .tolist()
        )
        series_rows: List[Dict[str, Any]] = []
        for (loc, ch), grp in sub.groupby(["location_id", "channel"]):
            by_date = (
                grp.groupby("date")["forecast_units"]
                .sum()
                .reindex(pd.to_datetime(dates))
                .fillna(0)
                .astype(int)
                .tolist()
            )
            series_rows.append(
                {
                    "location": str(loc),
                    "channel": str(ch),
                    "data": by_date,
                    "total": int(sum(by_date)),
                }
            )
        series_rows.sort(key=lambda r: r["total"], reverse=True)
        skus.append(
            {
                "skuId": str(sku),
                "department": str(meta_row.get("department", "")),
                "class": str(meta_row.get("class", "")),
                "subclass": str(meta_row.get("subclass", "")),
                "lifecycle": str(meta_row.get(lifecycle_col, "")),
                "aggregate": daily,
                "totalUnits": int(sum(daily)),
                "avgDaily": round(sum(daily) / max(len(daily), 1), 2),
                "series": series_rows[:5],
            }
        )

    return {
        "meta": {
            "recipe": "sku-forecasts",
            "source": str(Path(config["paths"]["forecasts_csv"]).resolve()),
            "snapshotDate": snapshot_date(df),
            "horizonDays": len(dates),
            "forecastStart": dates[0] if dates else None,
            "forecastEnd": dates[-1] if dates else None,
            "topSkus": top_skus,
        },
        "dates": dates,
        "skus": skus,
    }


def holdout_forecasts(config: Dict[str, Any], sku_id: str) -> Dict[str, Any]:
    df = load_dataset(config, "holdout_predictions")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    sub = df[df["sku_id"].astype(str) == str(sku_id)]
    if sub.empty:
        raise KeyError(f"SKU {sku_id!r} not found in holdout predictions")

    meta_row = sub.iloc[0]
    lifecycle_col = (
        "product_lifecycle_status"
        if "product_lifecycle_status" in sub.columns
        else "lifecycle"
    )

    daily = (
        sub.groupby("date")[["actual", "forecast"]]
        .sum()
        .sort_index()
    )
    dates = daily.index.strftime("%Y-%m-%d").tolist()
    actuals = daily["actual"].astype(float).tolist()
    predictions = daily["forecast"].astype(float).tolist()

    actual_arr = daily["actual"].to_numpy(dtype="float64")
    forecast_arr = daily["forecast"].to_numpy(dtype="float64")
    wape_val = wape(actual_arr, forecast_arr)
    mae_val = mae(actual_arr, forecast_arr)

    return {
        "meta": {
            "recipe": "holdout-forecasts",
            "source": str(Path(config["paths"]["holdout_predictions_csv"]).resolve()),
            "skuId": str(sku_id),
            "snapshotDate": dates[-1] if dates else None,
            "holdoutStart": dates[0] if dates else None,
            "holdoutEnd": dates[-1] if dates else None,
            "horizonDays": len(dates),
            "department": str(meta_row.get("department", "")),
            "class": str(meta_row.get("class", "")),
            "subclass": str(meta_row.get("subclass", "")),
            "lifecycle": str(meta_row.get(lifecycle_col, "")),
        },
        "dates": dates,
        "actuals": actuals,
        "predictions": predictions,
        "metrics": {
            "mae": round(mae_val, 4),
            "wape": round(wape_val, 6) if not pd.isna(wape_val) else None,
        },
    }


def recommendations_list(
    config: Dict[str, Any],
    *,
    risk: Optional[str] = None,
    action: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = MAX_RECOMMENDATIONS,
) -> Dict[str, Any]:
    df = load_dataset(config, "recommendations").copy()
    action_col = "recommended_action" if "recommended_action" in df.columns else "action"
    df["action"] = df[action_col].map(normalize_action)

    if risk:
        df = df[df["risk_flag"] == risk]
    if action:
        df = df[df["action"] == action]
    if department:
        df = df[df["department"] == department]

    df = df.sort_values(["risk_flag", "department", "sku_id"]).head(limit)

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                "skuId": str(row["sku_id"]),
                "locationId": str(row["location_id"]),
                "channel": str(row.get("channel", "")),
                "department": str(row.get("department", "")),
                "forecastHorizon": int(row.get("forecast_horizon", 0)),
                "forecastUnits": float(row.get("forecast_units", 0)),
                "inventoryOnHand": float(row.get("inventory_on_hand", 0)),
                "weeksOfSupply": float(row.get("weeks_of_supply", 0)),
                "riskFlag": str(row.get("risk_flag", "")),
                "recommendedAction": str(row["action"]),
                "recommendedMarkdownPct": float(row.get("recommended_markdown_pct", 0)),
                "expectedSales": float(row.get("expected_sales", 0)),
                "expectedMargin": float(row.get("expected_margin", 0)),
                "objectiveScore": float(row.get("objective_score", 0)),
                "reasonCode": str(row.get("reason_code", "")),
                "explanation": str(row.get("explanation", "")),
            }
        )

    recs_full = load_dataset(config, "recommendations")
    action_col_full = (
        "recommended_action" if "recommended_action" in recs_full.columns else "action"
    )
    recs_full = recs_full.assign(action=recs_full[action_col_full].map(normalize_action))

    return {
        "meta": {
            "snapshotDate": snapshot_date(recs_full),
            "totalMatching": int(len(df)),
            "returned": len(rows),
            "filters": {
                "risk": risk,
                "action": action,
                "department": department,
            },
        },
        "filters": {
            "departments": sorted(recs_full["department"].dropna().unique().tolist()),
            "riskFlags": sorted(recs_full["risk_flag"].dropna().unique().tolist()),
            "actions": sorted(recs_full["action"].dropna().unique().tolist()),
        },
        "rows": rows,
    }


def department_metrics(config: Dict[str, Any]) -> Dict[str, Any]:
    metrics = load_dataset(config, "metrics")
    recs = load_dataset(config, "recommendations")
    overall = metrics[metrics["level"] == "overall"].sort_values("wape")
    best_model = str(overall["model"].iloc[0]) if len(overall) else "unknown"

    dept = metrics[(metrics["level"] == "department") & (metrics["model"] == best_model)]
    dept = dept.sort_values("wape")

    rows = [
        {
            "department": str(row["group"]),
            "wape": float(row["wape"]),
            "mape": float(row["mape"]),
            "mae": float(row["mae"]),
            "rmse": float(row["rmse"]),
            "bias": float(row["bias"]),
            "n": int(row["n"]),
        }
        for _, row in dept.iterrows()
    ]

    return {
        "meta": {
            "snapshotDate": snapshot_date(recs),
            "model": best_model,
            "source": str(Path(config["paths"]["metrics_csv"]).resolve()),
        },
        "departments": rows,
    }
