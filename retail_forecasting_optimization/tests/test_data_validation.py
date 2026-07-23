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
    assert "derived_stockout_flag" in cleaned.columns


def test_negative_units_are_clipped(sample_df, config):
    """Negative units_sold should be clipped to 0 and reported."""
    df = sample_df.copy()
    df.loc[df.index[0], "units_sold"] = -5
    cleaned, result = validate_and_clean(df, config)
    assert cleaned["units_sold"].min() >= 0
    assert any(i.check == "negative_units" for i in result.issues)


def test_selling_above_regular_is_clamped(sample_df, config):
    """selling_price > regular_price should be clamped when not allowed."""
    df = sample_df.copy()
    df.loc[df.index[0], "selling_price"] = 999.0
    cleaned, result = validate_and_clean(df, config)
    assert (cleaned["selling_price"] <= cleaned["regular_price"]).all()
    assert any(i.check == "selling_above_regular" for i in result.issues)


def test_duplicate_grain_rows_removed(sample_df, config):
    """Duplicate rows at the series+date grain should be dropped."""
    df = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    cleaned, result = validate_and_clean(df, config)
    grain = config["data"]["series_keys"] + [config["data"]["date_col"]]
    assert cleaned.duplicated(subset=grain).sum() == 0
    assert any(i.check == "duplicate_grain" for i in result.issues)


def test_impossible_markdown_clipped(sample_df, config):
    """Markdown percentages outside [0, max] are clipped into range."""
    df = sample_df.copy()
    df.loc[df.index[0], "markdown_pct"] = 5.0  # 500% markdown is impossible
    cleaned, result = validate_and_clean(df, config)
    assert cleaned["markdown_pct"].max() <= config["validation"]["max_markdown_pct"]
    assert any(i.check == "impossible_markdown" for i in result.issues)


def test_missing_target_rows_dropped(sample_df, config):
    """Rows missing the target should be dropped and flagged."""
    df = sample_df.copy()
    df.loc[df.index[0], "units_sold"] = np.nan
    cleaned, result = validate_and_clean(df, config)
    assert cleaned["units_sold"].notna().all()
    assert any(i.check == "missing_target" for i in result.issues)
