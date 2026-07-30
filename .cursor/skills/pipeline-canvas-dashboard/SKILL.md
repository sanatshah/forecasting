---
name: pipeline-canvas-dashboard
description: Build interactive Cursor canvases from retail forecasting pipeline outputs (forecasts, recommendations, metrics). Use when the user asks for a dashboard, canvas, or live view of pipeline data, SKU explorers, risk breakdowns, or executive summaries fed by outputs/.
---

# Pipeline canvas dashboards

## Scope

Interactive **Cursor canvases** (`.canvas.tsx`) backed by CSV artifacts from `retail_forecasting_optimization/outputs/`.

- **Canvas rules**: read the **canvas** skill (`~/.cursor/skills-cursor/canvas/SKILL.md`) before writing any `.canvas.tsx`.
- **Pipeline data**: use **run-pipeline** if outputs are missing or stale.
- **Static PNG charts**: use **create-plot** instead — this skill is for live, interactive canvases beside the chat.

## Workflow checklist

```
Pipeline canvas dashboard:
- [ ] cd retail_forecasting_optimization
- [ ] Confirm datasets exist: python -m src.plotting list-datasets
- [ ] If missing/stale, run pipeline (run-pipeline skill; prefer --quick for smoke)
- [ ] Describe target dataset(s): python -m src.plotting describe <name>
- [ ] Pick dashboard pattern (see reference.md) or design from the prompt
- [ ] Extract embedded data: python .cursor/skills/pipeline-canvas-dashboard/scripts/extract_canvas_data.py <recipe>
- [ ] Write ~/.cursor/projects/<workspace>/canvases/<name>.canvas.tsx (canvas skill path rules)
- [ ] Embed extracted JSON as typed consts; no fetch(), no npm imports
- [ ] Label every chart/table with metric, units, legend, source path, snapshot date
- [ ] Verify Canvas TypeScript check reports no errors
- [ ] Link the canvas file in the chat response
```

Use `./.venv/bin/python` from `retail_forecasting_optimization/` (see run-pipeline skill).

## Data → canvas rules

Canvases **cannot load CSVs at runtime**. Always:

1. Read pipeline CSVs with Python (extract script or `load_dataset`).
2. Aggregate to dashboard-sized payloads (counts, top-N series, KPI totals).
3. Paste results as inline `const` objects in the `.canvas.tsx` file.

Include a `META` or `SOURCE` block naming the CSV path and snapshot date (max `date` in the dataset).

**Never render empty sections.** Omit charts/tables/stats with no rows. If the whole dashboard would be empty, do not create a canvas — say what's missing and run the pipeline.

## Extract script (preferred)

From `retail_forecasting_optimization/`:

```bash
# Built-in recipes (stdout JSON → paste into canvas)
./.venv/bin/python ../.cursor/skills/pipeline-canvas-dashboard/scripts/extract_canvas_data.py action-breakdown
./.venv/bin/python ../.cursor/skills/pipeline-canvas-dashboard/scripts/extract_canvas_data.py executive-summary
./.venv/bin/python ../.cursor/skills/pipeline-canvas-dashboard/scripts/extract_canvas_data.py sku-forecasts --top-skus 8

# Save to file for large payloads
./.venv/bin/python ../.cursor/skills/pipeline-canvas-dashboard/scripts/extract_canvas_data.py action-breakdown -o /tmp/action-breakdown.json
```

| Recipe | Primary dataset | Typical canvas |
|--------|-----------------|----------------|
| `action-breakdown` | recommendations | Stacked bar + UsageBar by department/action |
| `executive-summary` | metrics + recommendations | Stat row + risk/action KPIs |
| `sku-forecasts` | forecasts | Select + LineChart per SKU/location |

For custom aggregations, extend the script or use a one-off pandas snippet — still embed the result inline.

## Canvas location and naming

Write to the workspace managed directory only:

`~/.cursor/projects/<workspace>/canvases/<descriptive-name>.canvas.tsx`

Examples in this repo: `action-breakdown-by-department.canvas.tsx`, `sku-forecast-explorer.canvas.tsx`.

Optional interactive defaults: sibling `<name>.canvas.data.json` (e.g. `{ "selectedSku": "SKU0003" }`) with `useCanvasState`.

## Component mapping

| Dashboard need | `cursor/canvas` components |
|----------------|----------------------------|
| KPI strip | `Grid` + `Stat` |
| Time series | `LineChart` (`categories` = dates, `series` = metrics) |
| Category breakdown | `BarChart` (stacked or normalized) |
| Composition / mix | `UsageBar` + `Swatch` colors |
| Detail grid | `Table` with `headers`, `rows`, `columnAlign` |
| Filters | `Select`, `useCanvasState` |
| Section chrome | `Card` / `CardHeader` / `CardBody` — mix with open `Stack` sections |

Read `~/.cursor/skills-cursor/canvas/sdk/index.d.ts` for exact prop shapes.

## Existing examples

Study before inventing new layouts:

- `~/.cursor/projects/Users-sunny-Projects-forecasting/canvases/action-breakdown-by-department.canvas.tsx` — recommendations breakdown
- `~/.cursor/projects/Users-sunny-Projects-forecasting/canvases/sku-forecast-explorer.canvas.tsx` — SKU selector + forecast curves

## After delivery

1. Link the canvas with a markdown file path (canvas skill intro rules).
2. One sentence on what pipeline snapshot it reflects and how to refresh (re-run pipeline + re-extract + update consts).
3. Do not commit generated `outputs/` or canvas files unless the user asks.

## Additional resources

- Dataset columns and dashboard patterns: [reference.md](reference.md)
- PlotSpec PNG charts (not canvases): **create-plot** skill
