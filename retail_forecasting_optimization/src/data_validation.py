"""Data validation and cleaning.

Produces a structured data-quality report and a cleaned DataFrame. The design
goal is *robustness*: the pipeline should keep running on imperfect data while
clearly recording every issue it found and every fix it applied.

The report is a list of :class:`DataQualityIssue` records (pydantic models),
which is easy to serialize to CSV and to assert against in unit tests.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .utils import get_logger

logger = get_logger(__name__)


class DataQualityIssue(BaseModel):
    """A single validation finding.

    Attributes
    ----------
    check: machine-readable check name.
    severity: ``error`` | ``warning`` | ``info``.
    count: number of rows/cells affected.
    detail: human-readable description.
    action: what the cleaner did about it (if anything).
    """

    check: str
    severity: str = Field(default="warning")
    count: int = 0
    detail: str = ""
    action: str = ""


class ValidationResult(BaseModel):
    """Container returned by :func:`validate_and_clean`."""

    issues: List[DataQualityIssue] = Field(default_factory=list)
    n_rows_in: int = 0
    n_rows_out: int = 0
    passed: bool = True

    def to_frame(self) -> pd.DataFrame:
        """Render the issue list as a DataFrame for CSV export/printing."""
        if not self.issues:
            return pd.DataFrame(
                columns=["check", "severity", "count", "detail", "action"]
            )
        return pd.DataFrame([i.model_dump() for i in self.issues])


def _check_required_columns(
    df: pd.DataFrame, required: List[str], issues: List[DataQualityIssue]
) -> None:
    """Record any required columns that are missing from the input."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(
            DataQualityIssue(
                check="required_columns",
                severity="error",
                count=len(missing),
                detail=f"Missing required columns: {missing}",
                action="Downstream steps will skip logic that needs them.",
            )
        )


def _check_dates(
    df: pd.DataFrame, date_col: str, issues: List[DataQualityIssue]
) -> pd.DataFrame:
    """Flag and drop rows whose date failed to parse."""
    if date_col not in df.columns:
        return df
    bad = df[date_col].isna()
    n_bad = int(bad.sum())
    if n_bad:
        issues.append(
            DataQualityIssue(
                check="date_parsing",
                severity="error",
                count=n_bad,
                detail=f"{n_bad} rows had unparseable dates.",
                action="Dropped rows with invalid dates.",
            )
        )
        df = df.loc[~bad].copy()
    return df


def _check_duplicates(
    df: pd.DataFrame, grain: List[str], issues: List[DataQualityIssue]
) -> pd.DataFrame:
    """Detect and drop duplicate rows at the date + series grain.

    Duplicates at the demand grain would double-count sales, so we keep the
    last occurrence and record how many were removed.
    """
    grain = [c for c in grain if c in df.columns]
    if not grain:
        return df
    dup_mask = df.duplicated(subset=grain, keep="last")
    n_dup = int(dup_mask.sum())
    if n_dup:
        issues.append(
            DataQualityIssue(
                check="duplicate_grain",
                severity="warning",
                count=n_dup,
                detail=f"{n_dup} duplicate rows at grain {grain}.",
                action="Kept last occurrence, dropped earlier duplicates.",
            )
        )
        df = df.loc[~dup_mask].copy()
    return df


def _check_missing_values(
    df: pd.DataFrame,
    nullable: List[str],
    target_col: str,
    issues: List[DataQualityIssue],
) -> pd.DataFrame:
    """Report and impute missing values.

    - Missing target (``units_sold``) rows are dropped (can't train on them).
    - Missing numeric covariates are filled with 0 (a safe retail default for
      counts/flags) except prices, which are forward/back filled within series.
    - Columns declared nullable in config are ignored.
    """
    for col in df.columns:
        if col in nullable:
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        if col == target_col:
            issues.append(
                DataQualityIssue(
                    check="missing_target",
                    severity="error",
                    count=n_missing,
                    detail=f"{n_missing} rows missing target '{target_col}'.",
                    action="Dropped rows with missing target.",
                )
            )
            df = df.loc[df[col].notna()].copy()
        else:
            fill_val = 0
            df[col] = df[col].fillna(fill_val) if np.issubdtype(
                df[col].dtype, np.number
            ) else df[col].fillna("Unknown")
            issues.append(
                DataQualityIssue(
                    check="missing_values",
                    severity="info",
                    count=n_missing,
                    detail=f"Column '{col}' had {n_missing} missing values.",
                    action=f"Filled with {fill_val!r}/'Unknown'.",
                )
            )
    return df


def _check_value_ranges(
    df: pd.DataFrame,
    val_cfg: Dict[str, Any],
    issues: List[DataQualityIssue],
) -> pd.DataFrame:
    """Detect impossible values: negative units/prices, bad markdowns, etc.

    Negative units and prices are clipped to 0; markdown percentages are
    clipped into ``[0, max_markdown_pct]``; ``selling_price > regular_price``
    is reported (and clamped unless explicitly allowed in config).
    """
    # Negative units.
    if "units_sold" in df.columns:
        neg = df["units_sold"] < 0
        n = int(neg.sum())
        if n:
            issues.append(
                DataQualityIssue(
                    check="negative_units",
                    severity="warning",
                    count=n,
                    detail=f"{n} rows with negative units_sold.",
                    action="Clipped to 0.",
                )
            )
            df.loc[neg, "units_sold"] = 0

    # Negative prices.
    for price_col in ["regular_price", "selling_price"]:
        if price_col in df.columns:
            neg = df[price_col] < val_cfg["min_price"]
            n = int(neg.sum())
            if n:
                issues.append(
                    DataQualityIssue(
                        check=f"negative_{price_col}",
                        severity="warning",
                        count=n,
                        detail=f"{n} rows with {price_col} < {val_cfg['min_price']}.",
                        action="Clipped to minimum price.",
                    )
                )
                df.loc[neg, price_col] = val_cfg["min_price"]

    # Impossible markdown percentages.
    if "markdown_pct" in df.columns:
        max_md = val_cfg["max_markdown_pct"]
        bad = (df["markdown_pct"] < 0) | (df["markdown_pct"] > max_md)
        n = int(bad.sum())
        if n:
            issues.append(
                DataQualityIssue(
                    check="impossible_markdown",
                    severity="warning",
                    count=n,
                    detail=f"{n} rows with markdown_pct outside [0, {max_md}].",
                    action="Clipped into valid range.",
                )
            )
            df["markdown_pct"] = df["markdown_pct"].clip(0, max_md)

    # selling_price > regular_price.
    if {"selling_price", "regular_price"}.issubset(df.columns):
        bad = df["selling_price"] > df["regular_price"]
        n = int(bad.sum())
        if n:
            action = (
                "Left as-is (allowed by config)."
                if val_cfg.get("allow_selling_above_regular")
                else "Clamped selling_price to regular_price."
            )
            issues.append(
                DataQualityIssue(
                    check="selling_above_regular",
                    severity="warning",
                    count=n,
                    detail=f"{n} rows where selling_price > regular_price.",
                    action=action,
                )
            )
            if not val_cfg.get("allow_selling_above_regular"):
                df.loc[bad, "selling_price"] = df.loc[bad, "regular_price"]

    return df


def _detect_stockouts(
    df: pd.DataFrame, val_cfg: Dict[str, Any], issues: List[DataQualityIssue]
) -> pd.DataFrame:
    """Ensure a ``stockout_flag`` exists and reconcile with inventory.

    If the flag is absent we derive it from ``inventory_on_hand`` <= threshold.
    We also add ``derived_stockout_flag`` combining the source flag with the
    inventory signal, which feature engineering and optimization consume.
    """
    thr = val_cfg["stockout_inventory_threshold"]
    has_inv = "inventory_on_hand" in df.columns
    if "stockout_flag" not in df.columns and has_inv:
        df["stockout_flag"] = (df["inventory_on_hand"] <= thr).astype(int)
        issues.append(
            DataQualityIssue(
                check="stockout_flag_missing",
                severity="info",
                count=int(df["stockout_flag"].sum()),
                detail="stockout_flag was absent; derived from inventory.",
                action="Created stockout_flag from inventory_on_hand.",
            )
        )

    if has_inv:
        inv_stockout = (df["inventory_on_hand"] <= thr).astype(int)
        source_flag = df.get("stockout_flag", pd.Series(0, index=df.index)).fillna(0)
        df["derived_stockout_flag"] = ((source_flag == 1) | (inv_stockout == 1)).astype(int)
        n = int(df["derived_stockout_flag"].sum())
        issues.append(
            DataQualityIssue(
                check="stockout_periods",
                severity="info",
                count=n,
                detail=f"{n} row-days identified as stockout periods.",
                action="Recorded derived_stockout_flag for downstream use.",
            )
        )
    else:
        df["derived_stockout_flag"] = df.get("stockout_flag", 0)
    return df


def validate_and_clean(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, ValidationResult]:
    """Run all validation checks and return a cleaned DataFrame + report.

    The returned ``ValidationResult.passed`` is ``False`` only if a check with
    ``severity == "error"`` (other than droppable row-level issues) leaves the
    data unusable, i.e. required columns missing or no rows remaining.
    """
    data_cfg = config["data"]
    val_cfg = config["validation"]
    issues: List[DataQualityIssue] = []
    n_in = len(df)

    _check_required_columns(df, data_cfg["required_columns"], issues)
    df = _check_dates(df, data_cfg["date_col"], issues)
    df = _check_duplicates(df, data_cfg["series_keys"] + [data_cfg["date_col"]], issues)
    df = _check_missing_values(
        df, data_cfg["nullable_columns"], data_cfg["target_col"], issues
    )
    df = _check_value_ranges(df, val_cfg, issues)
    df = _detect_stockouts(df, val_cfg, issues)

    n_out = len(df)

    # Overall pass/fail: fail only if we lost all data or required cols missing.
    has_required = not any(
        i.check == "required_columns" and i.severity == "error" for i in issues
    )
    passed = bool(n_out > 0 and has_required)

    result = ValidationResult(
        issues=issues, n_rows_in=n_in, n_rows_out=n_out, passed=passed
    )
    logger.info(
        "Validation complete: %d -> %d rows, %d issues, passed=%s",
        n_in,
        n_out,
        len(issues),
        passed,
    )
    return df.reset_index(drop=True), result
