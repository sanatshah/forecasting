#!/usr/bin/env python3
"""Extract dashboard-sized JSON from retail forecasting pipeline CSVs.

Run from retail_forecasting_optimization/:

    ./.venv/bin/python ../.cursor/skills/pipeline-canvas-dashboard/scripts/extract_canvas_data.py action-breakdown
    ./.venv/bin/python ../.cursor/skills/pipeline-canvas-dashboard/scripts/extract_canvas_data.py sku-forecasts --top-skus 8 -o /tmp/sku.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Allow importing from retail_forecasting_optimization when run from that cwd.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC = _REPO_ROOT / "retail_forecasting_optimization"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.utils import load_config  # noqa: E402
from src.plotting.datasets import load_dataset  # noqa: E402

RECIPES = ("action-breakdown", "executive-summary", "sku-forecasts")


def _snapshot_date(df: pd.DataFrame, col: str = "date") -> str:
    if col not in df.columns or df[col].isna().all():
        return "unknown"
    return pd.to_datetime(df[col]).max().strftime("%Y-%m-%d")


def _normalize_action(action: str) -> str:
    if action == "REDUCE_EXPOSURE":
        return "REDUCE"
    return action


def recipe_action_breakdown(config: Dict[str, Any]) -> Dict[str, Any]:
    df = load_dataset(config, "recommendations")
    df = df.copy()
    action_col = "recommended_action" if "recommended_action" in df.columns else "action"
    df["action"] = df[action_col].map(_normalize_action)

    departments = sorted(df["department"].dropna().unique().tolist())
    actions = ["HOLD", "REPLENISH", "REDUCE", "TRANSFER"]

    breakdown: Dict[str, Dict[str, int]] = {}
    for dept in departments:
        sub = df[df["department"] == dept]
        counts = sub.groupby("action").size()
        breakdown[dept] = {a: int(counts.get(a, 0)) for a in actions}

    totals = {a: sum(breakdown[d][a] for d in departments) for a in actions}
    grand_total = int(len(df))

    return {
        "meta": {
            "recipe": "action-breakdown",
            "source": str(Path(config["paths"]["recommendations_csv"]).resolve()),
            "snapshotDate": _snapshot_date(df),
            "grandTotal": grand_total,
        },
        "departments": departments,
        "actions": actions,
        "breakdown": breakdown,
        "totals": totals,
    }


def recipe_executive_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    metrics = load_dataset(config, "metrics")
    recs = load_dataset(config, "recommendations")

    overall = metrics[metrics["level"] == "overall"]
    best_row = overall.sort_values("wape").head(1)
    best_model = str(best_row["model"].iloc[0]) if len(best_row) else "unknown"
    best_wape = float(best_row["wape"].iloc[0]) if len(best_row) else None

    risk_counts = recs.groupby("risk_flag").size().to_dict()
    action_col = "recommended_action" if "recommended_action" in recs.columns else "action"
    action_counts = recs.assign(
        action=recs[action_col].map(_normalize_action)
    ).groupby("action").size().to_dict()

    return {
        "meta": {
            "recipe": "executive-summary",
            "metricsSource": str(Path(config["paths"]["metrics_csv"]).resolve()),
            "recommendationsSource": str(Path(config["paths"]["recommendations_csv"]).resolve()),
            "snapshotDate": _snapshot_date(recs),
            "recommendationCount": int(len(recs)),
        },
        "bestModel": best_model,
        "bestWape": best_wape,
        "riskCounts": {str(k): int(v) for k, v in risk_counts.items()},
        "actionCounts": {str(k): int(v) for k, v in action_counts.items()},
    }


def recipe_sku_forecasts(config: Dict[str, Any], top_skus: int) -> Dict[str, Any]:
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
            "snapshotDate": _snapshot_date(df),
            "horizonDays": len(dates),
            "forecastStart": dates[0] if dates else None,
            "forecastEnd": dates[-1] if dates else None,
            "topSkus": top_skus,
        },
        "dates": dates,
        "skus": skus,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recipe",
        choices=RECIPES,
        help="Built-in aggregation recipe for canvas embedding.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Pipeline config path (relative to retail_forecasting_optimization/).",
    )
    parser.add_argument(
        "--top-skus",
        type=int,
        default=8,
        help="For sku-forecasts: number of SKUs by total forecast units.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write JSON to file instead of stdout.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.recipe == "action-breakdown":
        payload = recipe_action_breakdown(config)
    elif args.recipe == "executive-summary":
        payload = recipe_executive_summary(config)
    elif args.recipe == "sku-forecasts":
        payload = recipe_sku_forecasts(config, args.top_skus)
    else:
        parser.error(f"Unknown recipe: {args.recipe}")
        return 2

    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
