"""CLI for the declarative plot engine.

Usage (from retail_forecasting_optimization/):

    python -m src.plotting list-datasets
    python -m src.plotting describe forecasts
    python -m src.plotting render --spec plot_specs/actual_vs_forecast.json
    python -m src.plotting render --spec /tmp/my_spec.json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from ..utils import load_config
from .datasets import DATASET_NAMES, describe_dataset, list_datasets
from .renderer import render
from .spec import PlotSpec


def _cmd_list_datasets(config_path: Optional[str]) -> int:
    config = load_config(config_path)
    rows = list_datasets(config)
    print(json.dumps(rows, indent=2))
    missing = [r["name"] for r in rows if not r["exists"]]
    if missing:
        print(
            "\nMissing datasets: "
            + ", ".join(missing)
            + ". Run `python main.py --quick` first.",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_describe(config_path: Optional[str], name: str) -> int:
    config = load_config(config_path)
    try:
        info = describe_dataset(config, name)
    except (KeyError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(info, indent=2, default=str))
    return 0


def _cmd_render(config_path: Optional[str], spec_path: str) -> int:
    config = load_config(config_path)
    try:
        PlotSpec.from_json_file(spec_path)  # surface validation errors early
        path = render(spec_path, config)
    except Exception as exc:
        print(f"render failed: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.plotting",
        description="Discover datasets and render declarative PlotSpec JSON files.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to config.yaml (defaults to config/config.yaml).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-datasets", help="List registry datasets and whether CSVs exist.")

    p_desc = sub.add_parser("describe", help="Describe columns/dtypes/samples for a dataset.")
    p_desc.add_argument(
        "dataset",
        choices=DATASET_NAMES,
        help=f"Dataset name. Choices: {', '.join(DATASET_NAMES)}",
    )

    p_render = sub.add_parser("render", help="Validate and render a PlotSpec JSON file.")
    p_render.add_argument(
        "--spec",
        required=True,
        help="Path to a PlotSpec JSON file.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "list-datasets":
        return _cmd_list_datasets(args.config)
    if args.command == "describe":
        return _cmd_describe(args.config, args.dataset)
    if args.command == "render":
        return _cmd_render(args.config, args.spec)
    parser.error(f"Unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
