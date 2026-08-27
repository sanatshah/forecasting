"""Frozen baseline record and eval-protocol guards for model comparison.

The in-repo baseline (``eval/frozen_baseline.json``) captures the 2026-08-27
ml_gradient_boosting holdout score under the *actual-covariate* backtest
protocol. That protocol is optimistic relative to production forward
forecasting, which zeros promo/holiday flags and carry-forwards price/inventory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .model_ml import MLForecaster
from .utils import get_logger

logger = get_logger(__name__)

# Covariate protocols (holdout backtest vs production forward path).
COVARIATE_PROTOCOL_ACTUAL_HOLDOUT = "actual_holdout_covariates"
COVARIATE_PROTOCOL_PRODUCTION = "production_carry_forward"

HOLDOUT_ML_COVARIATE_PROTOCOL = COVARIATE_PROTOCOL_ACTUAL_HOLDOUT
HOLDOUT_ML_OPTIMISTIC_EVAL = True

FROZEN_BASELINE_PATH = Path(__file__).resolve().parent.parent / "eval" / "frozen_baseline.json"


def load_frozen_baseline(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the frozen baseline record from disk."""
    baseline_path = path or FROZEN_BASELINE_PATH
    with baseline_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def holdout_ml_covariate_protocol_label() -> str:
    """Human-readable label for the holdout ML covariate protocol."""
    return (
        "actual-promo/actual-covariate holdout "
        "(optimistic vs production carry-forward)"
    )


def assess_baseline_comparison_validity(
    config: Dict[str, Any],
    comparison: Dict[str, Any],
    *,
    quick: bool = False,
    baseline: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """Return whether a run is comparable to the frozen baseline and why not."""
    baseline = baseline or load_frozen_baseline()
    reasons_invalid: List[str] = []
    protocol = baseline["protocol"]
    dataset = baseline.get("dataset", {})

    if quick:
        reasons_invalid.append("quick mode subsets series (--quick); baseline used full sample")

    holdout_days = config["forecast"]["holdout_days"]
    if holdout_days != protocol["holdout_days"]:
        reasons_invalid.append(
            f"holdout_days={holdout_days} != baseline {protocol['holdout_days']}"
        )

    summary = comparison["summary"]
    if MLForecaster.name not in set(summary["model"]):
        reasons_invalid.append(f"{MLForecaster.name} missing from model summary")

    if comparison.get("holdout_covariate_protocol") != protocol["covariate_protocol"]:
        reasons_invalid.append(
            "holdout covariate protocol differs from baseline "
            f"({protocol['covariate_protocol']})"
        )

    if not comparison.get("optimistic_eval", False):
        reasons_invalid.append("holdout ML did not use optimistic actual-covariate protocol")

    cleaned_n_series = comparison.get("n_series")
    baseline_n_series = dataset.get("n_series")
    if (
        cleaned_n_series is not None
        and baseline_n_series is not None
        and cleaned_n_series != baseline_n_series
    ):
        reasons_invalid.append(
            f"n_series={cleaned_n_series} != baseline {baseline_n_series}"
        )

    return len(reasons_invalid) == 0, reasons_invalid


def compare_to_frozen_baseline(
    comparison: Dict[str, Any],
    config: Dict[str, Any],
    *,
    quick: bool = False,
    baseline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare this run's ml_gradient_boosting WAPE to the frozen baseline."""
    baseline = baseline or load_frozen_baseline()
    valid, invalid_reasons = assess_baseline_comparison_validity(
        config, comparison, quick=quick, baseline=baseline
    )

    summary = comparison["summary"]
    ml_row = summary[summary["model"] == MLForecaster.name]
    if ml_row.empty:
        current_wape = float("nan")
    else:
        current_wape = float(ml_row.iloc[0]["wape"])

    baseline_wape = float(baseline["metrics"]["wape"])
    beat_baseline = (
        valid
        and not pd.isna(current_wape)
        and current_wape < baseline_wape
    )

    return {
        "baseline_model": baseline["model"],
        "baseline_wape": baseline_wape,
        "baseline_mae": baseline["metrics"].get("mae"),
        "baseline_rmse": baseline["metrics"].get("rmse"),
        "baseline_bias": baseline["metrics"].get("bias"),
        "current_model": MLForecaster.name,
        "current_wape": current_wape,
        "beat_baseline": beat_baseline,
        "comparison_valid": valid,
        "invalid_reasons": invalid_reasons,
        "covariate_protocol": comparison.get(
            "holdout_covariate_protocol", HOLDOUT_ML_COVARIATE_PROTOCOL
        ),
        "covariate_protocol_label": holdout_ml_covariate_protocol_label(),
        "optimistic_eval": comparison.get("optimistic_eval", HOLDOUT_ML_OPTIMISTIC_EVAL),
        "holdout_days": config["forecast"]["holdout_days"],
        "wape_definition": baseline["protocol"]["wape_definition"],
    }


def format_baseline_comparison_report(result: Dict[str, Any]) -> str:
    """Format the baseline beat/validity block for stdout or logs."""
    lines = [
        "Frozen baseline comparison (actual-promo/actual-covariate holdout, NOT production):",
        (
            f"  Frozen {result['baseline_model']} WAPE: {result['baseline_wape']:.4f} "
            f"(MAE {result['baseline_mae']}, RMSE {result['baseline_rmse']}, "
            f"bias {result['baseline_bias']:+.2f})"
        ),
    ]
    if pd.isna(result["current_wape"]):
        lines.append(f"  This run {result['current_model']} WAPE: n/a")
    else:
        lines.append(
            f"  This run {result['current_model']} WAPE: {result['current_wape']:.4f}"
        )

    if result["comparison_valid"]:
        status = "BEAT" if result["beat_baseline"] else "DID NOT BEAT"
        lines.append(f"  Result: {status} frozen baseline (lower WAPE is better)")
    else:
        lines.append("  Result: comparison INVALID — beat/fail not scored")
        for reason in result["invalid_reasons"]:
            lines.append(f"    - {reason}")

    lines.append(
        f"  Eval protocol: {result['covariate_protocol']} — "
        f"{result['covariate_protocol_label']}"
    )
    if result["optimistic_eval"]:
        lines.append(
            "  optimistic_eval: true (holdout ML uses actual future promo/holiday/price/inventory)"
        )
    lines.append(
        f"  Metric: WAPE = {result['wape_definition']}; holdout_days={result['holdout_days']}"
    )
    return "\n".join(lines)


def log_baseline_comparison(
    comparison: Dict[str, Any],
    config: Dict[str, Any],
    *,
    quick: bool = False,
) -> Dict[str, Any]:
    """Log baseline comparison after compare_models."""
    result = compare_to_frozen_baseline(comparison, config, quick=quick)
    logger.info("\n%s", format_baseline_comparison_report(result))
    return result


def print_baseline_comparison(
    comparison: Dict[str, Any],
    config: Dict[str, Any],
    *,
    quick: bool = False,
) -> Dict[str, Any]:
    """Print baseline comparison in the executive summary."""
    result = compare_to_frozen_baseline(comparison, config, quick=quick)
    print(format_baseline_comparison_report(result))
    return result
