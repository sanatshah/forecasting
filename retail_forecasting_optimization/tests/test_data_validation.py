"""Tests for data validation and cleaning."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_validation import validate_and_clean


def test_clean_data_passes(sample_df, config):
    """A well-formed dataset should validate cleanly with no row loss."""
    cleaned, result = validate_and_clean(sample_df, config)
    assert result.passed is True
    assert result.n_rows_out == result.n_rows_in
    assert not any(i.check == "base_identity" for i in result.issues)


def test_negative_gross_adds_are_clipped(sample_df, config):
    """Negative gross_adds should be clipped to 0 and reported."""
    df = sample_df.copy()
    df.loc[df.index[0], "gross_adds"] = -5
    cleaned, result = validate_and_clean(df, config)
    assert cleaned["gross_adds"].min() >= 0
    assert any(i.check == "negative_gross_adds" for i in result.issues)


def test_effective_above_list_is_clamped(sample_df, config):
    """effective_price > list_price should be clamped when not allowed."""
    df = sample_df.copy()
    df.loc[df.index[0], "effective_price"] = 999.0
    cleaned, result = validate_and_clean(df, config)
    assert (cleaned["effective_price"] <= cleaned["list_price"]).all()
    assert any(i.check == "effective_above_list" for i in result.issues)


def test_duplicate_grain_rows_removed(sample_df, config):
    """Duplicate rows at the segment+date grain should be dropped."""
    df = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    cleaned, result = validate_and_clean(df, config)
    grain = config["data"]["series_keys"] + [config["data"]["date_col"]]
    assert cleaned.duplicated(subset=grain).sum() == 0
    assert any(i.check == "duplicate_grain" for i in result.issues)


def test_impossible_discount_clipped(sample_df, config):
    """Discounts outside [0, max] are clipped into range."""
    df = sample_df.copy()
    df.loc[df.index[0], "discount_pct"] = 5.0
    cleaned, result = validate_and_clean(df, config)
    assert cleaned["discount_pct"].max() <= config["validation"]["max_discount_pct"]
    assert any(i.check == "impossible_discount" for i in result.issues)


def test_missing_any_target_rows_dropped(sample_df, config):
    """Rows missing any forecast target should be dropped and flagged."""
    df = sample_df.copy()
    df.loc[df.index[0], "churned_subs"] = np.nan
    df.loc[df.index[1], "hours_watched"] = np.nan
    cleaned, result = validate_and_clean(df, config)
    for target in config["data"]["targets"]:
        assert cleaned[target].notna().all()
    assert len(cleaned) == len(df) - 2
    assert any(i.check == "missing_target" for i in result.issues)


def test_broken_base_identity_reported_not_rewritten(sample_df, config):
    """A ledger break is reported as a warning and left untouched."""
    df = sample_df.copy()
    df.loc[df.index[5], "paid_subs_eod"] += 100
    cleaned, result = validate_and_clean(df, config)
    issue = next(i for i in result.issues if i.check == "base_identity")
    assert issue.count == 1
    assert issue.severity == "warning"
    assert result.passed is True
    assert cleaned.loc[5, "paid_subs_eod"] == df.loc[5, "paid_subs_eod"]
