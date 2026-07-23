---
name: create-plot
description: Create a new plot from a natural-language prompt using the declarative PlotSpec engine. Use when the user asks to plot, chart, visualize, or graph pipeline outputs (forecasts, recommendations, metrics, holdout predictions, cleaned data).
---

# Create a plot from a prompt

## Scope

Declarative plot engine in `retail_forecasting_optimization/src/plotting/`.

The agent turns a user prompt into a validated JSON **PlotSpec**, then renders a PNG. Do **not** write ad-hoc matplotlib code.

Working directory: `retail_forecasting_optimization/`.

## Workflow checklist

```
Create plot:
- [ ] Confirm datasets exist (`python -m src.plotting list-datasets`)
- [ ] If missing, run pipeline (`python main.py --quick`) via run-pipeline skill
- [ ] `describe` the relevant dataset(s)
- [ ] Write a PlotSpec JSON (scratch under outputs/, not plot_specs/ unless permanent)
- [ ] `python -m src.plotting render --spec <path>`
- [ ] Open/view the PNG and iterate on validation or transform errors
```

## CLI

| Goal | Command |
|------|---------|
| List datasets | `python -m src.plotting list-datasets` |
| Inspect columns | `python -m src.plotting describe forecasts` |
| Render a spec | `python -m src.plotting render --spec path/to/spec.json` |

Use the project venv Python (see run-pipeline skill).

## PlotSpec shape

```json
{
  "dataset": "forecasts",
  "transform": {
    "filters": [{"col": "department", "op": "eq", "value": "Home"}],
    "top_groups": {"by": "series_id", "value": "forecast_units", "n": 5, "agg": "sum"},
    "groupby": {"by": ["date"], "agg": {"forecast_units": "sum"}},
    "pivot": {"index": "department", "columns": "risk_flag", "values": "forecast_units", "aggfunc": "count"},
    "melt": {"id_vars": ["date"], "value_vars": ["actual", "forecast"], "var_name": "metric", "value_name": "units"},
    "sort": {"by": "forecast_units", "ascending": false},
    "top_n": 20
  },
  "chart": {
    "kind": "line",
    "x": "date",
    "y": "forecast_units",
    "series": null
  },
  "style": {
    "title": "My chart",
    "xlabel": "Date",
    "ylabel": "Units",
    "figsize": [10, 5],
    "tick_rotation": 0,
    "legend": true
  },
  "output": "my_chart.png"
}
```

Omit unused transform keys (or leave `transform` as `{}`).

### Allowed values

| Field | Options |
|-------|---------|
| `dataset` | `forecasts`, `recommendations`, `metrics`, `holdout_predictions`, `cleaned` |
| `filter.op` | `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `notin`, `contains` |
| `agg` / `aggfunc` | `sum`, `mean`, `count`, `min`, `max`, `median`, `nunique`, `size` |
| `chart.kind` | `line`, `bar`, `barh`, `area`, `scatter`, `hist`, `heatmap` |
| `output` | Simple `name.png` only (written under `outputs/plots/`) |

- `groupby.agg` with `"size"` must be alone, e.g. `{"count": "size"}` for value_counts-style charts.
- `heatmap` usually follows a `pivot` (no `x`/`y` required).
- Multi-metric lines: `groupby` → `melt` → `chart.series` on the melted var column.
- Top-N series curves: `top_groups` then `chart.series` on the group key.

## Where to write specs

| Intent | Location |
|--------|----------|
| One-off / user prompt | `outputs/adhoc_<slug>.json` (gitignored via `outputs/*.csv` pattern does **not** cover json — prefer `outputs/plots/` sibling or delete after; do not commit unless asked) |
| Permanent pipeline chart | `plot_specs/<name>.json` (committed; rendered by `generate_all_plots`) |

Prefer `outputs/adhoc_<slug>.json` for prompt-driven charts.

## Examples

**WAPE by department (already a builtin):** see `plot_specs/wape_by_department.json`.

**Stockout recommendations by department:**

```json
{
  "dataset": "recommendations",
  "transform": {
    "filters": [{"col": "risk_flag", "op": "eq", "value": "STOCKOUT"}],
    "groupby": {"by": ["department"], "agg": {"count": "size"}},
    "sort": {"by": "count", "ascending": false}
  },
  "chart": {"kind": "barh", "x": "department", "y": "count"},
  "style": {
    "title": "Stockout recommendations by department",
    "xlabel": "Count",
    "legend": false
  },
  "output": "stockout_by_department.png"
}
```

## Failure handling

| Symptom | Action |
|---------|--------|
| Dataset missing | Run `python main.py --quick` (run-pipeline skill) |
| Validation error naming columns/ops | Fix the spec using the listed valid options; re-render |
| Empty / blank chart | `describe` the dataset; relax filters; check groupby columns |
| Want a chart kind not supported | Say so; do not fall back to free-form matplotlib |

## After render

- Print the saved PNG path from the CLI stdout.
- View the image to confirm it matches the prompt.
- Summarize what was plotted in one or two sentences.
