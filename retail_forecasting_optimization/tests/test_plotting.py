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
            "department": ["Home", "Home", "Mens Apparel", "Mens Apparel"],
            "forecast_units": [10.0, 12.0, 3.0, 4.0],
        }
    )


@pytest.fixture()
def tiny_recs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "department": ["Home", "Home", "Mens Apparel"],
            "risk_flag": ["STOCKOUT", "OVERSTOCK", "STOCKOUT"],
            "forecast_units": [1.0, 2.0, 3.0],
            "reason_code": ["LOW_WOS", "HIGH_WOS", "LOW_WOS"],
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
            "chart": {"kind": "line", "x": "date", "y": "forecast_units"},
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


def test_bias_by_department_spec_filters_best_model_and_renders(tmp_path, config):
    spec = PlotSpec.from_json_file(
        PROJECT_ROOT / "plot_specs" / "bias_by_department.json"
    )
    metrics = pd.DataFrame(
        {
            "model": ["ml_gradient_boosting"] * 3 + ["baseline"] * 3,
            "level": ["department"] * 6,
            "group": [
                "Home",
                "Mens Apparel",
                "Womens Apparel",
                "Home",
                "Mens Apparel",
                "Womens Apparel",
            ],
            "bias": [2.0, -3.0, 0.5, 100.0, 100.0, 100.0],
        }
    )

    transformed = apply_transforms(metrics, spec.transform)

    assert transformed["group"].tolist() == [
        "Mens Apparel",
        "Womens Apparel",
        "Home",
    ]
    assert transformed["bias"].tolist() == [-3.0, 0.5, 2.0]
    output = render(spec, config, df=metrics, plots_dir=tmp_path)
    assert Path(output).name == "bias_by_department.png"
    assert Path(output).stat().st_size > 0


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def test_filter_groupby_sort(tiny_forecasts):
    spec = PlotSpec.model_validate(
        {
            "dataset": "forecasts",
            "transform": {
                "filters": [{"col": "department", "op": "eq", "value": "Home"}],
                "groupby": {"by": ["date"], "agg": {"forecast_units": "sum"}},
                "sort": {"by": "date", "ascending": True},
            },
            "chart": {"kind": "line", "x": "date", "y": "forecast_units"},
            "output": "t.png",
        }
    )
    out = apply_transforms(tiny_forecasts, spec.transform)
    assert list(out["forecast_units"]) == [10.0, 12.0]


def test_top_groups_and_size_groupby(tiny_forecasts, tiny_recs):
    from src.plotting.spec import TransformSpec

    tg = TransformSpec.model_validate(
        {"top_groups": {"by": "series_id", "value": "forecast_units", "n": 1, "agg": "sum"}}
    )
    top = apply_transforms(tiny_forecasts, tg)
    assert set(top["series_id"]) == {"A"}

    counted = apply_transforms(
        tiny_recs,
        TransformSpec.model_validate(
            {"groupby": {"by": ["reason_code"], "agg": {"count": "size"}}}
        ),
    )
    assert int(counted.loc[counted["reason_code"] == "LOW_WOS", "count"].iloc[0]) == 2


def test_pivot_matrix(tiny_recs):
    from src.plotting.spec import TransformSpec

    matrix = apply_transforms(
        tiny_recs,
        TransformSpec.model_validate(
            {
                "pivot": {
                    "index": "department",
                    "columns": "risk_flag",
                    "values": "forecast_units",
                    "aggfunc": "count",
                }
            }
        ),
    )
    assert matrix.index.name == "department"
    assert "STOCKOUT" in matrix.columns


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
    assert "forecast_units" in msg


# ---------------------------------------------------------------------------
# Render smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,extra_chart,extra_transform",
    [
        ("line", {}, {"groupby": {"by": ["date"], "agg": {"forecast_units": "sum"}}}),
        ("area", {}, {"groupby": {"by": ["date"], "agg": {"forecast_units": "sum"}}}),
        ("bar", {}, {"groupby": {"by": ["department"], "agg": {"forecast_units": "sum"}}}),
        ("barh", {}, {"groupby": {"by": ["department"], "agg": {"forecast_units": "sum"}}}),
        ("scatter", {}, {}),
        ("hist", {"x": "forecast_units", "y": None}, {}),
        (
            "heatmap",
            {"x": None, "y": None},
            {
                "pivot": {
                    "index": "department",
                    "columns": "series_id",
                    "values": "forecast_units",
                    "aggfunc": "sum",
                }
            },
        ),
    ],
)
def test_render_each_kind(tmp_path, config, tiny_forecasts, kind, extra_chart, extra_transform):
    chart = {"kind": kind, "x": "date", "y": "forecast_units"}
    chart.update({k: v for k, v in extra_chart.items() if v is not None or k in ("x", "y")})
    if kind == "hist":
        chart = {"kind": "hist", "x": "forecast_units"}
    if kind == "heatmap":
        chart = {"kind": "heatmap"}
    if kind in ("bar", "barh"):
        chart["x"] = "department"

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
    # Minimal frames matching columns referenced by builtins that use these datasets.
    holdout = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "actual": [5.0, 6.0],
            "forecast": [4.5, 6.5],
        }
    )
    metrics = pd.DataFrame(
        {
            "model": ["ml_gradient_boosting"] * 4,
            "level": ["department", "department", "department", "channel"],
            "group": ["Home", "Mens Apparel", "Womens Apparel", "store"],
            "wape": [0.2, 0.3, 0.25, 0.1],
            "mae": [1.0, 2.0, 1.5, 0.5],
            "bias": [2.0, -3.0, 0.5, -0.25],
        }
    )
    # Point plots_dir at tmp so we don't clobber real outputs.
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
        },
    )
    assert "bias_by_department.png" in {Path(p).name for p in paths}
    for p in paths:
        assert Path(p).is_file()


def test_list_datasets_shape(config):
    rows = list_datasets(config)
    names = {r["name"] for r in rows}
    assert names == set(DATASET_NAMES)
    for r in rows:
        assert "exists" in r and "path" in r and "description" in r


def test_load_missing_dataset_message(config, tmp_path, monkeypatch):
    cfg = dict(config)
    cfg["paths"] = dict(config["paths"])
    cfg["paths"]["forecasts_csv"] = str(tmp_path / "missing_forecasts.csv")
    with pytest.raises(FileNotFoundError) as exc:
        load_dataset(cfg, "forecasts")
    assert "python main.py --quick" in str(exc.value)
