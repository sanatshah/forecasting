"""Tests for the declarative PlotSpec engine."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from src.plotting.datasets import DATASET_NAMES, list_datasets, load_dataset
from src.plotting.renderer import apply_transforms, render, render_builtin_specs
from src.plotting.spec import PlotSpec
from src.utils import PROJECT_ROOT


@pytest.fixture()
def tiny_forecasts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]
            ),
            "series_id": ["A", "A", "B", "B"],
            "tier": ["Premium", "Premium", "Ad Tier", "Ad Tier"],
            "forecast_value": [10.0, 12.0, 3.0, 4.0],
            "forecast_net_adds": [5.0, -2.0, 1.0, 3.0],
            "forecast_paid_subs": [1005.0, 1003.0, 501.0, 504.0],
            "forecast_hours_per_paid_sub": [1.1, 1.2, 0.7, 0.8],
        }
    )


@pytest.fixture()
def tiny_recs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tier": ["Premium", "Premium", "Ad Tier"],
            "acquisition_channel": ["direct", "app_store", "direct"],
            "risk_flag": ["CHURN_SPIKE", "OK", "CHURN_SPIKE"],
            "forecast_horizon": [28, 28, 28],
            "forecast_net_adds": [1.0, 2.0, 3.0],
            "reason_code": ["CHURN_ABOVE_TREND", "STABLE_GROWTH", "CHURN_ABOVE_TREND"],
        }
    )


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


def test_plot_spec_rejects_unknown_dataset():
    with pytest.raises(ValidationError) as exc:
        PlotSpec.model_validate(
            {
                "dataset": "nope",
                "chart": {"kind": "bar", "x": "a", "y": "b"},
                "output": "x.png",
            }
        )
    msg = str(exc.value)
    assert "Unknown dataset" in msg
    for name in DATASET_NAMES:
        assert name in msg


def test_plot_spec_rejects_bad_output_and_kind():
    with pytest.raises(ValidationError) as exc:
        PlotSpec.model_validate(
            {
                "dataset": "forecasts",
                "chart": {"kind": "pie", "x": "a", "y": "b"},
                "output": "../evil.png",
            }
        )
    msg = str(exc.value)
    assert "Unknown chart kind" in msg or "output" in msg


def test_plot_spec_accepts_minimal_line():
    spec = PlotSpec.model_validate(
        {
            "dataset": "forecasts",
            "chart": {"kind": "line", "x": "date", "y": "forecast_net_adds"},
            "output": "ok.png",
        }
    )
    assert spec.dataset == "forecasts"
    assert spec.chart.kind == "line"


def test_all_builtin_specs_validate():
    specs_dir = PROJECT_ROOT / "plot_specs"
    files = sorted(specs_dir.glob("*.json"))
    assert files, "expected committed plot_specs/*.json"
    for path in files:
        PlotSpec.from_json_file(path)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def test_filter_groupby_sort(tiny_forecasts):
    spec = PlotSpec.model_validate(
        {
            "dataset": "forecasts",
            "transform": {
                "filters": [{"col": "tier", "op": "eq", "value": "Premium"}],
                "groupby": {"by": ["date"], "agg": {"forecast_value": "sum"}},
                "sort": {"by": "date", "ascending": True},
            },
            "chart": {"kind": "line", "x": "date", "y": "forecast_value"},
            "output": "t.png",
        }
    )
    out = apply_transforms(tiny_forecasts, spec.transform)
    assert list(out["forecast_value"]) == [10.0, 12.0]


def test_top_groups_and_size_groupby(tiny_forecasts, tiny_recs):
    from src.plotting.spec import TransformSpec

    tg = TransformSpec.model_validate(
        {"top_groups": {"by": "series_id", "value": "forecast_value", "n": 1, "agg": "sum"}}
    )
    top = apply_transforms(tiny_forecasts, tg)
    assert set(top["series_id"]) == {"A"}

    counted = apply_transforms(
        tiny_recs,
        TransformSpec.model_validate(
            {"groupby": {"by": ["reason_code"], "agg": {"count": "size"}}}
        ),
    )
    assert int(counted.loc[counted["reason_code"] == "CHURN_ABOVE_TREND", "count"].iloc[0]) == 2


def test_pivot_matrix(tiny_recs):
    from src.plotting.spec import TransformSpec

    matrix = apply_transforms(
        tiny_recs,
        TransformSpec.model_validate(
            {
                "pivot": {
                    "index": "tier",
                    "columns": "risk_flag",
                    "values": "forecast_net_adds",
                    "aggfunc": "count",
                }
            }
        ),
    )
    assert matrix.index.name == "tier"
    assert "CHURN_SPIKE" in matrix.columns


def test_unknown_column_error_lists_columns(tiny_forecasts):
    from src.plotting.spec import TransformSpec

    with pytest.raises(ValueError) as exc:
        apply_transforms(
            tiny_forecasts,
            TransformSpec.model_validate(
                {"filters": [{"col": "missing_col", "op": "eq", "value": 1}]}
            ),
        )
    msg = str(exc.value)
    assert "missing_col" in msg
    assert "forecast_value" in msg


# ---------------------------------------------------------------------------
# Render smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,extra_chart,extra_transform",
    [
        ("line", {}, {"groupby": {"by": ["date"], "agg": {"forecast_value": "sum"}}}),
        ("area", {}, {"groupby": {"by": ["date"], "agg": {"forecast_value": "sum"}}}),
        ("bar", {}, {"groupby": {"by": ["tier"], "agg": {"forecast_value": "sum"}}}),
        ("barh", {}, {"groupby": {"by": ["tier"], "agg": {"forecast_value": "sum"}}}),
        ("scatter", {}, {}),
        ("hist", {"x": "forecast_value", "y": None}, {}),
        (
            "heatmap",
            {"x": None, "y": None},
            {
                "pivot": {
                    "index": "tier",
                    "columns": "series_id",
                    "values": "forecast_value",
                    "aggfunc": "sum",
                }
            },
        ),
    ],
)
def test_render_each_kind(tmp_path, config, tiny_forecasts, kind, extra_chart, extra_transform):
    chart = {"kind": kind, "x": "date", "y": "forecast_value"}
    chart.update({k: v for k, v in extra_chart.items() if v is not None or k in ("x", "y")})
    if kind == "hist":
        chart = {"kind": "hist", "x": "forecast_value"}
    if kind == "heatmap":
        chart = {"kind": "heatmap"}
    if kind in ("bar", "barh"):
        chart["x"] = "tier"

    spec = {
        "dataset": "forecasts",
        "transform": extra_transform,
        "chart": chart,
        "style": {"title": f"smoke {kind}", "legend": False},
        "output": f"smoke_{kind}.png",
    }
    path = render(spec, config, df=tiny_forecasts, plots_dir=tmp_path)
    assert Path(path).is_file()
    assert Path(path).stat().st_size > 0


def test_render_builtin_specs_with_frames(tmp_path, config, tiny_forecasts, tiny_recs):
    # Minimal frames matching columns referenced by every builtin spec.
    holdout = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"]),
            "target": ["gross_adds", "gross_adds", "churned_subs"],
            "actual": [5.0, 6.0, 2.0],
            "forecast": [4.5, 6.5, 2.5],
        }
    )
    metrics = pd.DataFrame(
        {
            "model": ["m"] * 5,
            "target": ["gross_adds", "churned_subs", "gross_adds", "churned_subs", "churned_subs"],
            "level": ["tier", "tier", "acquisition_channel", "series_id", "series_id"],
            "group": ["Premium", "Premium", "direct", "Premium|direct", "Ad Tier|direct"],
            "wape": [0.2, 0.3, 0.1, 0.25, 0.35],
            "mae": [1.0, 2.0, 0.5, 1.5, 2.5],
        }
    )
    okr = pd.DataFrame(
        {
            "forecast_horizon": [28, 28, 7],
            "scenario": ["baseline", "price_change", "baseline"],
            "high_value_net_adds": [1200.0, 900.0, 300.0],
        }
    )
    cfg = dict(config)
    cfg["paths"] = dict(config["paths"])
    cfg["paths"]["plots_dir"] = str(tmp_path)

    paths = render_builtin_specs(
        cfg,
        data_by_dataset={
            "forecasts": tiny_forecasts,
            "recommendations": tiny_recs,
            "metrics": metrics,
            "holdout_predictions": holdout,
            "okr_summary": okr,
        },
    )
    n_specs = len(list((PROJECT_ROOT / "plot_specs").glob("*.json")))
    assert len(paths) == n_specs
    for p in paths:
        assert Path(p).is_file()


def test_list_datasets_shape(config):
    rows = list_datasets(config)
    names = {r["name"] for r in rows}
    assert names == set(DATASET_NAMES)
    assert "okr_summary" in names
    for r in rows:
        assert "exists" in r and "path" in r and "description" in r


def test_load_missing_dataset_message(config, tmp_path, monkeypatch):
    cfg = dict(config)
    cfg["paths"] = dict(config["paths"])
    cfg["paths"]["forecasts_csv"] = str(tmp_path / "missing_forecasts.csv")
    with pytest.raises(FileNotFoundError) as exc:
        load_dataset(cfg, "forecasts")
    assert "python main.py --quick" in str(exc.value)
