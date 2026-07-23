"""Deterministic PlotSpec renderer (transforms + matplotlib Agg).

No ``eval``, no generated code — only enum-dispatched chart kinds and a fixed
set of DataFrame transforms.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..utils import ensure_dir, get_logger  # noqa: E402
from .datasets import load_dataset  # noqa: E402
from .spec import (  # noqa: E402
    ChartSpec,
    FilterSpec,
    GroupBySpec,
    MeltSpec,
    PivotSpec,
    PlotSpec,
    SortSpec,
    StyleSpec,
    TopGroupsSpec,
    TransformSpec,
)

logger = get_logger(__name__)

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    _HAS_SNS = True
except Exception:  # pragma: no cover
    _HAS_SNS = False

_FILTER_OPS = {
    "eq": lambda s, v: s == v,
    "ne": lambda s, v: s != v,
    "gt": lambda s, v: s > v,
    "gte": lambda s, v: s >= v,
    "lt": lambda s, v: s < v,
    "lte": lambda s, v: s <= v,
    "in": lambda s, v: s.isin(list(v) if not isinstance(v, (list, tuple, set)) else v),
    "notin": lambda s, v: ~s.isin(list(v) if not isinstance(v, (list, tuple, set)) else v),
    "contains": lambda s, v: s.astype(str).str.contains(str(v), na=False),
}


def _require_columns(df: pd.DataFrame, cols: List[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{where}: unknown column(s) {missing}. "
            f"Available columns: {', '.join(map(str, df.columns))}"
        )


def _apply_filters(df: pd.DataFrame, filters: List[FilterSpec]) -> pd.DataFrame:
    out = df
    for f in filters:
        _require_columns(out, [f.col], "filter")
        op = _FILTER_OPS[f.op]
        out = out.loc[op(out[f.col], f.value)].copy()
    return out


def _apply_top_groups(df: pd.DataFrame, spec: TopGroupsSpec) -> pd.DataFrame:
    _require_columns(df, [spec.by, spec.value], "top_groups")
    if spec.agg == "size":
        totals = df.groupby(spec.by).size()
    else:
        totals = df.groupby(spec.by)[spec.value].agg(spec.agg)
    top = list(totals.sort_values(ascending=False).head(spec.n).index)
    return df[df[spec.by].isin(top)].copy()


def _apply_groupby(df: pd.DataFrame, spec: GroupBySpec) -> pd.DataFrame:
    _require_columns(df, list(spec.by), "groupby.by")
    size_keys = [k for k, fn in spec.agg.items() if fn == "size"]
    other = {k: fn for k, fn in spec.agg.items() if fn != "size"}
    if size_keys and other:
        raise ValueError(
            "groupby.agg cannot mix 'size' with other aggregations in one step; "
            "use size alone (e.g. {\"count\": \"size\"})"
        )
    if size_keys:
        if len(size_keys) != 1:
            raise ValueError("groupby.agg may have only one 'size' output column")
        name = size_keys[0]
        out = df.groupby(spec.by, dropna=False).size().reset_index(name=name)
        return out
    for col in other:
        # Allow renaming: if col not in df, treat key as output name only when
        # the source column equals the key (standard pandas named agg form).
        if col not in df.columns:
            raise ValueError(
                f"groupby.agg: unknown column {col!r}. "
                f"Available columns: {', '.join(map(str, df.columns))}"
            )
    out = df.groupby(spec.by, dropna=False).agg(other).reset_index()
    return out


def _apply_pivot(df: pd.DataFrame, spec: PivotSpec) -> pd.DataFrame:
    _require_columns(df, [spec.index, spec.columns, spec.values], "pivot")
    if spec.aggfunc == "size":
        # count of rows per cell
        pivot = df.pivot_table(
            index=spec.index,
            columns=spec.columns,
            values=spec.values,
            aggfunc="count",
            fill_value=0,
        )
    else:
        pivot = df.pivot_table(
            index=spec.index,
            columns=spec.columns,
            values=spec.values,
            aggfunc=spec.aggfunc,
            fill_value=0,
        )
    return pivot


def _apply_melt(df: pd.DataFrame, spec: MeltSpec) -> pd.DataFrame:
    _require_columns(df, list(spec.id_vars) + list(spec.value_vars), "melt")
    return df.melt(
        id_vars=spec.id_vars,
        value_vars=spec.value_vars,
        var_name=spec.var_name,
        value_name=spec.value_name,
    )


def _apply_sort(df: pd.DataFrame, spec: SortSpec) -> pd.DataFrame:
    by = [spec.by] if isinstance(spec.by, str) else list(spec.by)
    _require_columns(df, by, "sort")
    return df.sort_values(by, ascending=spec.ascending)


def apply_transforms(df: pd.DataFrame, transform: TransformSpec) -> pd.DataFrame:
    """Apply the ordered transform pipeline; returns a new DataFrame."""
    out = df
    if transform.filters:
        out = _apply_filters(out, transform.filters)
    if transform.top_groups is not None:
        out = _apply_top_groups(out, transform.top_groups)
    if transform.groupby is not None:
        out = _apply_groupby(out, transform.groupby)
    if transform.pivot is not None:
        out = _apply_pivot(out, transform.pivot)

    # Pivoted matrix (index + wide columns) until melted to long form.
    is_matrix = transform.pivot is not None and transform.melt is None

    if transform.melt is not None:
        if is_matrix or out.index.name is not None:
            out = out.reset_index()
        out = _apply_melt(out, transform.melt)
        is_matrix = False

    if transform.sort is not None:
        if is_matrix:
            out = out.sort_index(ascending=transform.sort.ascending)
        else:
            out = _apply_sort(out, transform.sort)

    if transform.top_n is not None:
        out = out.head(transform.top_n)
    return out


def _save(fig: plt.Figure, plots_dir: Path, name: str) -> str:
    path = plots_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot %s", path)
    return str(path)


def _y_list(y: Optional[Union[str, List[str]]]) -> List[str]:
    if y is None:
        return []
    return [y] if isinstance(y, str) else list(y)


def _draw_chart(df: pd.DataFrame, chart: ChartSpec, style: StyleSpec) -> plt.Figure:
    fig, ax = plt.subplots(figsize=tuple(style.figsize))

    if chart.kind == "heatmap":
        # Prefer pivoted matrix (index + numeric columns).
        matrix = df
        if not isinstance(matrix.index, pd.RangeIndex) or matrix.index.name is not None:
            pass
        else:
            # Long form: need x, y and a single values column in y or inferred
            if chart.x and chart.y and isinstance(chart.y, str):
                _require_columns(df, [chart.x, chart.y], "heatmap")
                # If a values column was melt'd, try common name; else count
                val_col = None
                for candidate in ("value", "count", "n", "forecast_units"):
                    if candidate in df.columns and candidate not in (chart.x, chart.y):
                        val_col = candidate
                        break
                if val_col is None:
                    raise ValueError(
                        "heatmap on a flat frame needs a numeric values column "
                        f"(tried value/count/n/forecast_units). Columns: "
                        f"{', '.join(map(str, df.columns))}"
                    )
                matrix = df.pivot_table(
                    index=chart.y, columns=chart.x, values=val_col, aggfunc="sum", fill_value=0
                )
        if matrix.empty:
            ax.text(0.5, 0.5, "No data", ha="center")
        elif _HAS_SNS:
            sns.heatmap(matrix, annot=True, fmt="g", cmap="Reds", ax=ax)
        else:  # pragma: no cover
            im = ax.imshow(matrix.values, aspect="auto", cmap="Reds")
            ax.set_xticks(range(len(matrix.columns)))
            ax.set_xticklabels(matrix.columns, rotation=45)
            ax.set_yticks(range(len(matrix.index)))
            ax.set_yticklabels(matrix.index)
            fig.colorbar(im, ax=ax)

    elif chart.kind == "hist":
        col = chart.x or (chart.y if isinstance(chart.y, str) else None)
        assert col is not None
        _require_columns(df, [col], "hist")
        ax.hist(df[col].dropna(), bins=chart.bins)

    elif chart.series is not None:
        _require_columns(df, [chart.x, chart.series], "chart")
        ycols = _y_list(chart.y)
        if len(ycols) != 1:
            raise ValueError("When chart.series is set, chart.y must be a single column")
        _require_columns(df, ycols, "chart.y")
        ycol = ycols[0]
        for key, g in df.groupby(chart.series, sort=False):
            g = g.sort_values(chart.x)
            if chart.kind == "line":
                ax.plot(g[chart.x], g[ycol], label=str(key), marker=".", ms=3)
            elif chart.kind == "area":
                ax.fill_between(g[chart.x], g[ycol], alpha=0.35, label=str(key))
                ax.plot(g[chart.x], g[ycol], marker=".", ms=3)
            elif chart.kind == "bar":
                # Grouped bars: offset by category position — simple overlay fallback
                ax.bar(g[chart.x].astype(str), g[ycol], label=str(key), alpha=0.7)
            elif chart.kind == "barh":
                ax.barh(g[chart.x].astype(str), g[ycol], label=str(key), alpha=0.7)
            elif chart.kind == "scatter":
                ax.scatter(g[chart.x], g[ycol], label=str(key), s=20)
            else:
                raise ValueError(f"chart.series not supported for kind={chart.kind!r}")
        if style.legend:
            ax.legend(fontsize=7)

    else:
        ycols = _y_list(chart.y)
        _require_columns(df, [chart.x] + ycols, "chart")
        # Multi-y without series: plot each y column
        for ycol in ycols:
            if chart.kind == "line":
                ax.plot(df[chart.x], df[ycol], label=ycol, marker=".", ms=3)
            elif chart.kind == "area":
                ax.fill_between(df[chart.x], df[ycol], alpha=0.4, label=ycol)
                ax.plot(df[chart.x], df[ycol], marker=".", ms=3)
            elif chart.kind == "bar":
                ax.bar(df[chart.x].astype(str), df[ycol], label=ycol)
            elif chart.kind == "barh":
                ax.barh(df[chart.x].astype(str), df[ycol], label=ycol)
            elif chart.kind == "scatter":
                ax.scatter(df[chart.x], df[ycol], label=ycol, s=20)
            else:
                raise ValueError(f"Unhandled chart kind {chart.kind!r}")
        if style.legend and len(ycols) > 1:
            ax.legend()

    if style.title:
        ax.set_title(style.title)
    if style.xlabel:
        ax.set_xlabel(style.xlabel)
    elif chart.x and chart.kind not in ("heatmap", "hist"):
        ax.set_xlabel(chart.x)
    if style.ylabel:
        ax.set_ylabel(style.ylabel)
    if style.tick_rotation:
        plt.setp(ax.get_xticklabels(), rotation=style.tick_rotation, ha="right")
    # Auto date formatting when x looks like datetimes
    if chart.x and chart.x in df.columns and pd.api.types.is_datetime64_any_dtype(df[chart.x]):
        fig.autofmt_xdate()
    return fig


def render(
    spec: PlotSpec | Dict[str, Any] | str,
    config: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
    plots_dir: Optional[Path | str] = None,
) -> str:
    """Render a PlotSpec to a PNG under ``plots_dir``; return the saved path.

    Parameters
    ----------
    spec:
        A ``PlotSpec``, a raw dict, or a path to a JSON file.
    config:
        Pipeline config (used for dataset paths and default plots_dir).
    df:
        Optional in-memory DataFrame; skips CSV load when provided.
    plots_dir:
        Override output directory (defaults to ``config['paths']['plots_dir']``).
    """
    if isinstance(spec, str):
        plot_spec = PlotSpec.from_json_file(spec)
    elif isinstance(spec, dict):
        plot_spec = PlotSpec.model_validate(spec)
    else:
        plot_spec = spec

    if df is None:
        data = load_dataset(config, plot_spec.dataset)
    else:
        data = df.copy()

    transformed = apply_transforms(data, plot_spec.transform)
    fig = _draw_chart(transformed, plot_spec.chart, plot_spec.style)

    out_dir = ensure_dir(plots_dir if plots_dir is not None else config["paths"]["plots_dir"])
    # Confine output to plots_dir (filename already validated by PlotSpec).
    return _save(fig, out_dir, plot_spec.output)


def render_builtin_specs(
    config: Dict[str, Any],
    data_by_dataset: Optional[Dict[str, pd.DataFrame]] = None,
    specs_dir: Optional[Path | str] = None,
) -> List[str]:
    """Render every JSON spec in ``plot_specs/``; guard per-plot failures."""
    from ..utils import PROJECT_ROOT

    directory = (
        Path(specs_dir)
        if specs_dir is not None
        else PROJECT_ROOT / "plot_specs"
    )
    if not directory.is_dir():
        logger.warning("Builtin plot_specs directory missing: %s", directory)
        return []

    paths: List[str] = []
    for spec_path in sorted(directory.glob("*.json")):
        try:
            plot_spec = PlotSpec.from_json_file(spec_path)
            frame = None
            if data_by_dataset is not None:
                frame = data_by_dataset.get(plot_spec.dataset)
            paths.append(render(plot_spec, config, df=frame))
        except Exception as exc:  # pragma: no cover - guarded like legacy code
            logger.warning("Plot generation failed for %s: %s", spec_path.name, exc)
    return paths
