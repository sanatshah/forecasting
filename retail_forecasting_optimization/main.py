"""End-to-end Peacock subscriber forecasting & scenario pipeline.

Run:
    python main.py                # uses config/config.yaml
    python main.py --config path  # custom config
    python main.py --quick        # cap series for a fast smoke run

Stages: load -> validate -> per target (features -> train/select -> forecast)
-> derive net adds / paid subs / usage per paid sub -> scenarios + OKRs
-> save outputs -> plots -> executive summary.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List

import pandas as pd

from src.data_loader import add_series_id, load_content_calendar, load_raw_data
from src.data_validation import validate_and_clean
from src.explainability import generate_explainability
from src.feature_engineering import build_features
from src.forecasting_pipeline import (
    combine_target_forecasts,
    forecast_target,
    horizon_rollups,
    save_forecasts,
)
from src.model_selection import compare_models, save_metrics
from src.scenario_engine import (
    build_okr_summary,
    generate_recommendations,
    save_okr_summary,
    save_recommendations,
)
from src.utils import (
    config_for_target,
    ensure_dir,
    get_logger,
    get_targets,
    load_config,
    resolve_path,
    set_global_seed,
)
from src.visualization import generate_all_plots

logger = get_logger("main")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Peacock subscriber forecasting pipeline")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Limit to a few series for a fast smoke test.",
    )
    return parser.parse_args()


def _maybe_subset(df: pd.DataFrame, quick: bool) -> pd.DataFrame:
    """In quick mode, keep a few series spread across tiers for speed."""
    if not quick:
        return df
    # Round-robin over tiers so the OKR rollup and price scenarios still have
    # high-value / price-change segments to report on.
    ids = df[["series_id", "tier"]].drop_duplicates("series_id")
    ids = ids.assign(_rank=ids.groupby("tier").cumcount())
    keep = ids.sort_values(["_rank", "tier"], kind="stable")["series_id"].head(4)
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
    calendar = load_content_calendar(config)

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

    # 3-6. Per target: features -> backtest/select -> forward forecast -------
    comparisons: Dict[str, Dict[str, Any]] = {}
    per_target: Dict[str, pd.DataFrame] = {}
    for target in get_targets(config):
        logger.info("=== Target: %s ===", target)
        tcfg = config_for_target(config, target)
        features = build_features(cleaned, tcfg)
        comparisons[target] = compare_models(cleaned, features, tcfg)
        per_target[target] = forecast_target(comparisons[target], cleaned, tcfg, calendar)

    metrics_table = pd.concat(
        [c["metrics_table"] for c in comparisons.values()], ignore_index=True
    )
    save_metrics(metrics_table, config)

    daily_forecast = combine_target_forecasts(per_target, cleaned, config)
    save_forecasts(daily_forecast, config)
    rollups = horizon_rollups(daily_forecast, config)

    # 7. Scenario engine + OKRs ---------------------------------------------
    recommendations = generate_recommendations(rollups, cleaned, config, calendar)
    save_recommendations(recommendations, config)
    okr = build_okr_summary(recommendations, config)
    save_okr_summary(okr, config)

    # 8-9. Explainability & plots ------------------------------------------
    primary_target = get_targets(config)[0]
    explain = generate_explainability(
        config, comparisons[primary_target]["ml_model"], recommendations
    )
    holdout = pd.concat(
        [c["predictions"][c["best_model_name"]] for c in comparisons.values()],
        ignore_index=True,
    )
    holdout_path = resolve_path(config["paths"]["holdout_predictions_csv"])
    holdout.to_csv(holdout_path, index=False)
    logger.info("Saved holdout predictions to %s", holdout_path)
    plots = generate_all_plots(
        config, holdout, metrics_table, daily_forecast, recommendations, okr
    )

    return {
        "cleaned": cleaned,
        "quality": quality,
        "comparisons": comparisons,
        "metrics_table": metrics_table,
        "daily_forecast": daily_forecast,
        "rollups": rollups,
        "recommendations": recommendations,
        "okr": okr,
        "explain": explain,
        "plots": plots,
    }


def _fmt_int(value: float) -> str:
    return f"{value:,.0f}"


def print_executive_summary(config: Dict[str, Any], artifacts: Dict[str, Any]) -> None:
    """Print the concise executive summary for the Growth OKR review."""
    cleaned = artifacts["cleaned"]
    comparisons = artifacts["comparisons"]
    recs = artifacts["recommendations"]
    okr = artifacts["okr"]
    primary = config["forecast"]["primary_horizon"]

    n_seg = cleaned["series_id"].nunique()
    n_tier = cleaned["tier"].nunique() if "tier" in cleaned else 0
    n_ch = cleaned["acquisition_channel"].nunique() if "acquisition_channel" in cleaned else 0

    line = "=" * 72
    print("\n" + line)
    print("EXECUTIVE SUMMARY - Peacock Subscriber & Engagement Forecast")
    print(line)
    print(f"Segments processed          : {n_seg} ({n_tier} tiers x {n_ch} channels)")
    print(f"History                     : {cleaned['date'].min().date()} -> {cleaned['date'].max().date()}")
    print(f"Forecast horizons (days)    : {config['forecast']['horizons']}")

    print("\nBest model per target (WAPE):")
    for target, comp in comparisons.items():
        best = comp["best_model_name"]
        w = float(comp["summary"].iloc[0]["wape"])
        print(f"  - {target:<16} {best:<26} WAPE={w:.4f} (accuracy ~{max(0.0, 1 - w) * 100:.1f}%)")

    base = okr[(okr["forecast_horizon"] == primary) & (okr["scenario"] == "baseline")]
    scen = okr[(okr["forecast_horizon"] == primary) & (okr["scenario"] == "price_change")]
    if not base.empty:
        b = base.iloc[0]
        print(f"\nGrowth OKRs ({primary}-day horizon, baseline):")
        print(f"  High-value net adds         : {_fmt_int(b['high_value_net_adds'])}")
        print(f"  High-value paid subs (end)  : {_fmt_int(b['high_value_paid_subs_end'])}")
        print(f"  Total net adds              : {_fmt_int(b['total_net_adds'])}")
        print(f"  Total paid subs (end)       : {_fmt_int(b['total_paid_subs_end'])}")
        print(f"  Hours per paid sub / month  : {b['hours_per_paid_sub_month']:.2f}")
        print(f"  Revenue                     : ${_fmt_int(b['revenue'])}")
        if not scen.empty:
            s = scen.iloc[0]
            print("  With planned price change   :")
            print(f"    High-value net adds       : {_fmt_int(s['high_value_net_adds'])} "
                  f"({s['high_value_net_adds'] - b['high_value_net_adds']:+,.0f})")
            print(f"    Revenue                   : ${_fmt_int(s['revenue'])} "
                  f"({s['revenue'] - b['revenue']:+,.0f})")

    primary_recs = recs[recs["forecast_horizon"] == primary]
    print(f"\nRisk flags ({primary}-day horizon):")
    for flag, n in primary_recs["risk_flag"].value_counts().items():
        print(f"  - {flag:<20} {n}")
    print("Recommended actions:")
    for action, n in primary_recs["recommended_action"].value_counts().items():
        print(f"  - {action:<20} {n}")

    drivers: List[str] = artifacts["explain"].get("top_drivers", [])[:5]
    if drivers:
        print(f"\nTop drivers ({get_targets(config)[0]} model):")
        for d in drivers:
            print(f"  - {d}")
    print("\nOutputs written to:")
    for key in [
        "forecasts_csv",
        "recommendations_csv",
        "okr_summary_csv",
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
