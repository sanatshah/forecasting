"""Declarative plot-spec engine for pipeline artifacts.

Cloud agents (and humans) discover datasets, write a validated JSON PlotSpec,
and render a PNG via a deterministic matplotlib backend — no generated code.
"""
from __future__ import annotations

from .datasets import DATASET_NAMES, describe_dataset, list_datasets, load_dataset
from .renderer import render
from .spec import PlotSpec

__all__ = [
    "DATASET_NAMES",
    "PlotSpec",
    "describe_dataset",
    "list_datasets",
    "load_dataset",
    "render",
]
