#!/usr/bin/env python3
"""Extract dashboard-sized JSON from retail forecasting pipeline CSVs.

Run from retail_forecasting_optimization/:

    ./.venv/bin/python ../.cursor/skills/pipeline-canvas-dashboard/scripts/extract_canvas_data.py action-breakdown
    ./.venv/bin/python ../.cursor/skills/pipeline-canvas-dashboard/scripts/extract_canvas_data.py sku-forecasts --top-skus 8 -o /tmp/sku.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Allow importing from retail_forecasting_optimization when run from that cwd.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC = _REPO_ROOT / "retail_forecasting_optimization"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.utils import load_config  # noqa: E402
from src.dashboard import aggregations  # noqa: E402

RECIPES = ("action-breakdown", "executive-summary", "sku-forecasts")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recipe",
        choices=RECIPES,
        help="Built-in aggregation recipe for canvas embedding.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Pipeline config path (relative to retail_forecasting_optimization/).",
    )
    parser.add_argument(
        "--top-skus",
        type=int,
        default=8,
        help="For sku-forecasts: number of SKUs by total forecast units.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write JSON to file instead of stdout.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.recipe == "action-breakdown":
        payload = aggregations.action_breakdown(config)
    elif args.recipe == "executive-summary":
        payload = aggregations.executive_summary(config)
    elif args.recipe == "sku-forecasts":
        payload = aggregations.sku_forecasts(config, args.top_skus)
    else:
        parser.error(f"Unknown recipe: {args.recipe}")
        return 2

    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
