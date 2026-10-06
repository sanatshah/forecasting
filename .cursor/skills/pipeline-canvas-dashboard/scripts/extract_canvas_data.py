#!/usr/bin/env python3
"""Extract dashboard-sized JSON from subscriber forecasting pipeline CSVs.

Run from retail_forecasting_optimization/:

    ./.venv/bin/python ../.cursor/skills/pipeline-canvas-dashboard/scripts/extract_canvas_data.py action-breakdown
    ./.venv/bin/python ../.cursor/skills/pipeline-canvas-dashboard/scripts/extract_canvas_data.py segment-forecasts -o /tmp/segments.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow importing from retail_forecasting_optimization when run from that cwd.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC = _REPO_ROOT / "retail_forecasting_optimization"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.utils import load_config  # noqa: E402
from src.dashboard import aggregations  # noqa: E402

RECIPES = ("action-breakdown", "executive-summary", "segment-forecasts", "okr", "segment-metrics")


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
        "--target",
        default=None,
        help="For segment-metrics: gross_adds, churned_subs or hours_watched.",
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
    elif args.recipe == "segment-forecasts":
        payload = aggregations.segment_forecasts(config)
    elif args.recipe == "okr":
        payload = aggregations.okr_summary(config)
    elif args.recipe == "segment-metrics":
        payload = aggregations.segment_metrics(config, args.target)
    else:
        parser.error(f"Unknown recipe: {args.recipe}")
        return 2

    text = json.dumps(payload, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
