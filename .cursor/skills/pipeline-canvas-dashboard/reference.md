# Pipeline canvas reference

## Dataset registry

All paths resolve via `config/config.yaml`. Discover at runtime:

```bash
python -m src.plotting list-datasets
python -m src.plotting describe forecasts
```

| Name | CSV | Use in dashboards |
|------|-----|-------------------|
| `forecasts` | `outputs/forecasts.csv` | Daily forward units by series; SKU/location/channel curves |
| `recommendations` | `outputs/recommendations.csv` | Actions (HOLD, REPLENISH, REDUCE, TRANSFER), risk flags, inventory KPIs |
| `metrics` | `outputs/model_metrics.csv` | WAPE/MAPE by model, department, channel |
| `holdout_predictions` | `outputs/holdout_predictions.csv` | Actual vs forecast backtest lines |
| `cleaned` | `data/processed/cleaned.csv` | Historical demand for context charts |

## Recommendations columns (dashboard-relevant)

| Column | Role |
|--------|------|
| `date` | Snapshot date (usually one row per SKU-location) |
| `department`, `class`, `subclass` | Merch hierarchy filters |
| `sku_id`, `location_id`, `channel` | Series keys |
| `recommended_action` | HOLD, REPLENISH, REDUCE_EXPOSURE, TRANSFER |
| `risk_flag` | STOCKOUT, OVERSTOCK, BALANCED, etc. |
| `forecast_units`, `on_hand_units`, `weeks_of_supply` | Numeric KPIs |

Normalize `REDUCE_EXPOSURE` → `REDUCE` in display labels when matching existing canvases.

## Metrics columns

| Column | Role |
|--------|------|
| `model` | Model name |
| `level` | overall, department, channel |
| `group` | Segment name when level ≠ overall |
| `wape`, `mape`, `mae`, `rmse`, `bias` | Metric values per row |

## Forecasts columns

| Column | Role |
|--------|------|
| `date` | Forecast date |
| `series_id` or `sku_id` + `location_id` + `channel` | Series identity |
| `forecast_units` | Predicted demand |
| `department`, `class`, `subclass`, `lifecycle` | Filters and grouping |

## Dashboard patterns

### Executive summary strip

- **Stats**: best model + WAPE from `metrics` (level=overall); stockout/overstock counts from `recommendations` (`risk_flag`).
- **Source caption**: `outputs/model_metrics.csv` + `outputs/recommendations.csv` + snapshot date.

### Action breakdown by department

- **Groupby**: `department` × `recommended_action` (count rows).
- **Charts**: stacked `BarChart`; normalized mix per department; `UsageBar` per department.
- **Table**: department × action columns + totals row.

### SKU forecast explorer

- **Top SKUs**: by sum of `forecast_units` over horizon (default 8).
- **Per SKU**: aggregate daily totals + per location/channel series for `LineChart`.
- **Interaction**: `Select` for SKU via `useCanvasState`; optional `.canvas.data.json` default.
- **Cap series**: keep ≤5 location/channel lines per SKU for readability.

### Model accuracy comparison

- **Filter** `metrics` where `level='department'`; compare `wape` across `group` values.
- **BarChart**: departments on X, one series per model (or single best model).

### Risk heatmap alternative

For dense matrices, prefer **create-plot** (`chart.kind: heatmap`). Use canvas when the user wants filtering or narrative layout around a smaller table.

## Refresh workflow

1. `python main.py` or `python main.py --quick`
2. Re-run `extract_canvas_data.py <recipe>`
3. Update inline consts in the `.canvas.tsx` (or regenerate the file)
4. Confirm TypeScript check passes

## TypeScript embedding tips

- Use `as const` on string literal arrays (`DATES`, department names).
- Prefer plain objects/arrays over CSV-sized raw rows in the canvas file.
- Type extracted payloads with small local `type` aliases matching the JSON shape.
