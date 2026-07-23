"""End-to-end retail forecasting & optimization pipeline.

Run:
    python main.py                # uses config/config.yaml
    python main.py --config path  # custom config
    python main.py --quick        # cap series for a fast smoke run

Stages: load -> validate -> features -> train/select -> forecast -> optimize
-> save outputs -> plots -> executive summary.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict

import pandas as pd

from src.data_loader import add_series_id, load_raw_data
from src.data_validation import validate_and_clean
from src.explainability import generate_explainability
from src.feature_engineering import build_features
from src.forecasting_pipeline import (
    generate_forecasts,
    horizon_rollups,
)
from src.model_selection import compare_models, save_metrics
from src.optimization_engine import generate_recommendations, save_recommendations
from src.utils import ensure_dir, get_logger, load_config, resolve_path, set_global_seed
from src.visualization import generate_all_plots

logger = get_logger("main")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retail forecasting & optimization pipeline")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Limit to a few series for a fast smoke test.",
    )
    return parser.parse_args()


def _maybe_subset(df: pd.DataFrame, quick: bool) -> pd.DataFrame:
    """In quick mode, keep only the first few series for speed."""
    if not quick:
        return df
    keep = df["series_id"].drop_duplicates().head(4)
    logger.info("Quick mode: limiting to %d series", len(keep))
    return df[df["series_id"].isin(keep)].copy()


def run_pipeline(config: Dict[str, Any], quick: bool = False) -> Dict[str, Any]:
    """Execute the full pipeline and return a dict of key artifacts."""
    set_global_seed(config["project"]["random_seed"])
    ensure_dir(config["paths"]["outputs_dir"])
    ensure_dir(config["paths"]["plots_dir"])
    ensure_dir(config["paths"]["processed_dir"])

    # 1. Load ---------------------------------------------------------------
    raw = load_raw_data(config)
    raw = add_series_id(raw, config)
    raw = _maybe_subset(raw, quick)

    # 2. Validate & clean ---------------------------------------------------
    cleaned, quality = validate_and_clean(raw, config)
    quality_path = resolve_path(config["paths"]["quality_report_csv"])
    quality.to_frame().to_csv(quality_path, index=False)
    processed_path = resolve_path(config["paths"]["processed_dir"]) / "cleaned.csv"
    cleaned.to_csv(processed_path, index=False)
    logger.info("Saved data quality report to %s", quality_path)
    if not quality.passed:
        logger.error("Validation failed; aborting. See %s", quality_path)
        raise SystemExit(2)

    # 3. Feature engineering -----------------------------------------------
    features = build_features(cleaned, config)

    # 4-5. Train models & select best --------------------------------------
    comparison = compare_models(cleaned, features, config)
    save_metrics(comparison, config)

    # 6. Forward forecasts --------------------------------------------------
    daily_forecast = generate_forecasts(comparison, cleaned, config)
    rollups = horizon_rollups(daily_forecast, config)

    # 7. Optimization engine ------------------------------------------------
    date_col = config["data"]["date_col"]
    latest_state = (
        cleaned.sort_values(date_col).groupby("series_id").tail(1).reset_index(drop=True)
    )
    recommendations = generate_recommendations(rollups, latest_state, config)
    save_recommendations(recommendations, config)

    # 8-9. Explainability & plots ------------------------------------------
    explain = generate_explainability(config, comparison["ml_model"], recommendations)
    best_pred = comparison["predictions"][comparison["best_model_name"]]
    holdout_path = resolve_path(config["paths"]["holdout_predictions_csv"])
    best_pred.to_csv(holdout_path, index=False)
    logger.info("Saved holdout predictions to %s", holdout_path)
    plots = generate_all_plots(
        config, best_pred, comparison["metrics_table"], daily_forecast, recommendations
    )

    return {
        "cleaned": cleaned,
        "quality": quality,
        "comparison": comparison,
        "daily_forecast": daily_forecast,
        "rollups": rollups,
        "recommendations": recommendations,
        "explain": explain,
        "plots": plots,
    }


def print_executive_summary(config: Dict[str, Any], artifacts: Dict[str, Any]) -> None:
    """Print the concise executive summary required by the spec."""
    cleaned = artifacts["cleaned"]
    comparison = artifacts["comparison"]
    recs = artifacts["recommendations"]
    summary = comparison["summary"]
    best = comparison["best_model_name"]
    overall_wape = float(summary.iloc[0]["wape"])

    n_sku = cleaned["sku_id"].nunique() if "sku_id" in cleaned else 0
    n_prod = cleaned["product_id"].nunique() if "product_id" in cleaned else 0
    n_loc = cleaned["location_id"].nunique() if "location_id" in cleaned else 0

    # Top departments by forecast error (MAE) for the best model.
    mt = comparison["metrics_table"]
    dept_err = mt[(mt["model"] == best) & (mt["level"] == "department")]
    top_depts = (
        dept_err.sort_values("wape", ascending=False)[["group", "wape"]].head(3)
        if not dept_err.empty
        else pd.DataFrame(columns=["group", "wape"])
    )

    stockout_n = int((recs["risk_flag"] == "STOCKOUT").sum())
    overstock_n = int((recs["risk_flag"] == "OVERSTOCK").sum())
    # Business opportunity proxy: total expected margin at the primary horizon.
    primary = config["forecast"]["primary_horizon"]
    opp = float(recs[recs["forecast_horizon"] == primary]["expected_margin"].sum())

    drivers = artifacts["explain"].get("top_drivers", [])[:5]

    line = "=" * 68
    print("\n" + line)
    print("EXECUTIVE SUMMARY - Retail Demand Forecasting & Optimization")
    print(line)
    print(f"SKUs processed              : {n_sku}")
    print(f"Products processed          : {n_prod}")
    print(f"Locations processed         : {n_loc}")
    print(f"Forecast horizons (days)    : {config['forecast']['horizons']}")
    print(f"Best model (by WAPE)        : {best}")
    print(f"Overall WAPE                : {overall_wape:.4f}  "
          f"(accuracy ~{max(0.0, 1 - overall_wape) * 100:.1f}%)")
    print("\nModel leaderboard (WAPE):")
    for _, r in summary.iterrows():
        print(f"  - {r['model']:<26} WAPE={r['wape']:.4f}  MAE={r['mae']:.2f}  bias={r['bias']:.2f}")
    print("\nTop departments by forecast error (WAPE):")
    if top_depts.empty:
        print("  - n/a")
    else:
        for _, r in top_depts.iterrows():
            print(f"  - {r['group']:<20} WAPE={r['wape']:.4f}")
    print(f"\nStockout-risk recommendations : {stockout_n}")
    print(f"Overstock-risk recommendations: {overstock_n}")
    print(f"Est. business opportunity     : expected margin at {primary}-day "
          f"horizon = {opp:,.0f}")
    if drivers:
        print("\nTop forecast drivers:")
        for d in drivers:
            print(f"  - {d}")
    print("\nOutputs written to:")
    for key in [
        "forecasts_csv",
        "recommendations_csv",
        "metrics_csv",
        "holdout_predictions_csv",
        "quality_report_csv",
    ]:
        print(f"  - {resolve_path(config['paths'][key])}")
    print(f"  - {ensure_dir(config['paths']['plots_dir'])} (plots)")
    print(line + "\n")


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    logger.info("Starting pipeline (quick=%s)", args.quick)
    artifacts = run_pipeline(config, quick=args.quick)
    print_executive_summary(config, artifacts)
    logger.info("Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
