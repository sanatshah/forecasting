"""Pydantic PlotSpec contract for declarative chart rendering.

Validation errors intentionally list valid options (datasets, ops, chart kinds)
so a cloud agent can self-correct without guessing.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from .datasets import DATASET_NAMES

FilterOp = Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "notin", "contains"]
AggFunc = Literal["sum", "mean", "count", "min", "max", "median", "nunique", "size"]
ChartKind = Literal["line", "bar", "barh", "area", "scatter", "hist", "heatmap"]
SortOrder = Literal["asc", "desc"]

FILTER_OPS: List[str] = list(FilterOp.__args__)  # type: ignore[attr-defined]
AGG_FUNCS: List[str] = list(AggFunc.__args__)  # type: ignore[attr-defined]
CHART_KINDS: List[str] = list(ChartKind.__args__)  # type: ignore[attr-defined]

_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,120}\.png$")


class FilterSpec(BaseModel):
    """Keep rows where ``col`` satisfies ``op`` against ``value``."""

    col: str
    op: FilterOp
    value: Any

    @field_validator("op", mode="before")
    @classmethod
    def _check_op(cls, v: Any) -> Any:
        if v not in FILTER_OPS:
            raise ValueError(
                f"Unknown filter op {v!r}. Valid ops: {', '.join(FILTER_OPS)}"
            )
        return v


class GroupBySpec(BaseModel):
    """Group by columns and aggregate.

    ``agg`` maps output column name -> aggregation function. Use ``size`` for
    a row-count (the key becomes the count column name).
    """

    by: List[str] = Field(..., min_length=1)
    agg: Dict[str, AggFunc] = Field(..., min_length=1)

    @field_validator("agg", mode="before")
    @classmethod
    def _check_agg(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            raise ValueError("groupby.agg must be an object mapping column -> func")
        bad = [f for f in v.values() if f not in AGG_FUNCS]
        if bad:
            raise ValueError(
                f"Unknown agg func(s) {bad}. Valid funcs: {', '.join(AGG_FUNCS)}"
            )
        return v


class PivotSpec(BaseModel):
    """Pivot to a wide matrix (index kept as the DataFrame index)."""

    index: str
    columns: str
    values: str
    aggfunc: AggFunc = "count"


class MeltSpec(BaseModel):
    """Wide-to-long melt for multi-metric line / bar charts."""

    id_vars: List[str] = Field(..., min_length=1)
    value_vars: List[str] = Field(..., min_length=1)
    var_name: str = "variable"
    value_name: str = "value"


class TopGroupsSpec(BaseModel):
    """Keep rows belonging to the top-N groups by an aggregated value."""

    by: str
    value: str
    n: int = Field(..., ge=1, le=100)
    agg: AggFunc = "sum"


class SortSpec(BaseModel):
    """Sort rows by one or more columns."""

    by: Union[str, List[str]]
    ascending: bool = True


class TransformSpec(BaseModel):
    """Ordered, safe transform pipeline applied before charting."""

    filters: List[FilterSpec] = Field(default_factory=list)
    top_groups: Optional[TopGroupsSpec] = None
    groupby: Optional[GroupBySpec] = None
    pivot: Optional[PivotSpec] = None
    melt: Optional[MeltSpec] = None
    sort: Optional[SortSpec] = None
    top_n: Optional[int] = Field(default=None, ge=1, le=10_000)


class ChartSpec(BaseModel):
    """Chart geometry. For heatmap after pivot, ``x``/``y`` are optional."""

    kind: ChartKind
    x: Optional[str] = None
    y: Optional[Union[str, List[str]]] = None
    series: Optional[str] = None  # hue / one line (or bar group) per value
    bins: int = Field(default=20, ge=2, le=200)  # hist only

    @field_validator("kind", mode="before")
    @classmethod
    def _check_kind(cls, v: Any) -> Any:
        if v not in CHART_KINDS:
            raise ValueError(
                f"Unknown chart kind {v!r}. Valid kinds: {', '.join(CHART_KINDS)}"
            )
        return v

    @model_validator(mode="after")
    def _require_xy(self) -> "ChartSpec":
        if self.kind == "heatmap":
            return self
        if self.kind == "hist":
            if self.x is None and self.y is None:
                raise ValueError("hist charts require x (or y) column")
            return self
        if self.x is None:
            raise ValueError(
                f"{self.kind} charts require chart.x "
                f"(valid kinds with optional x: heatmap)"
            )
        if self.y is None:
            raise ValueError(f"{self.kind} charts require chart.y")
        return self


class StyleSpec(BaseModel):
    """Optional visual styling."""

    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    figsize: List[float] = Field(default_factory=lambda: [10.0, 5.0])
    tick_rotation: float = 0.0
    legend: bool = True

    @field_validator("figsize")
    @classmethod
    def _check_figsize(cls, v: List[float]) -> List[float]:
        if len(v) != 2:
            raise ValueError("style.figsize must be [width, height]")
        if any(x <= 0 or x > 40 for x in v):
            raise ValueError("style.figsize values must be in (0, 40]")
        return v


class PlotSpec(BaseModel):
    """Full declarative plot document."""

    dataset: str
    transform: TransformSpec = Field(default_factory=TransformSpec)
    chart: ChartSpec
    style: StyleSpec = Field(default_factory=StyleSpec)
    output: str = Field(..., description="PNG filename under plots_dir")

    @field_validator("dataset", mode="before")
    @classmethod
    def _check_dataset(cls, v: Any) -> Any:
        if v not in DATASET_NAMES:
            raise ValueError(
                f"Unknown dataset {v!r}. Valid datasets: {', '.join(DATASET_NAMES)}"
            )
        return v

    @field_validator("output", mode="before")
    @classmethod
    def _check_output(cls, v: Any) -> Any:
        if not isinstance(v, str) or not _SAFE_FILENAME.match(v):
            raise ValueError(
                "output must be a simple PNG filename "
                "(letters/digits/._-, end with .png, max ~120 chars); "
                f"got {v!r}"
            )
        return v

    @classmethod
    def from_json_file(cls, path: str | bytes) -> "PlotSpec":
        """Load and validate a PlotSpec from a JSON file path."""
        from pathlib import Path

        text = Path(path).read_text(encoding="utf-8")
        return cls.model_validate_json(text)
