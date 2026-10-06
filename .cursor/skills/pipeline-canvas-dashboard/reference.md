# Pipeline canvas reference

## Dataset registry

All paths resolve via `config/config.yaml`. Discover at runtime:

```bash
python -m src.plotting list-datasets
python -m src.plotting describe forecasts
```

| Name | CSV | Use in dashboards |
|------|-----|-------------------|
| `forecasts` | `outputs/forecasts.csv` | Daily forward flows and derived KPIs per segment |
| `recommendations` | `outputs/recommendations.csv` | Segment x horizon: risk flags, actions, price-change scenario |
| `okr_summary` | `outputs/okr_summary.csv` | Growth OKRs per horizon, baseline vs price change |
| `metrics` | `outputs/model_metrics.csv` | WAPE/MAPE by target, model, tier, channel, segment |
| `holdout_predictions` | `outputs/holdout_predictions.csv` | Actual vs forecast backtest lines per target |
| `cleaned` | `data/processed/cleaned.csv` | History for context charts |

## Recommendations columns (dashboard-relevant)

| Column | Role |
|--------|------|
| `date` | Snapshot date (last observed day) |
| `series_id`, `tier`, `acquisition_channel`, `distribution_partner` | Segment identity and filters |
| `forecast_horizon` | 7 / 14 / 28 days; filter to one (usually 28) |
| `forecast_gross_adds`, `forecast_churned_subs`, `forecast_net_adds`, `ending_paid_subs` | Baseline outlook |
| `hours_per_paid_sub_month`, `usage_change_pct` | Usage OKR and trend vs trailing |
| `forecast_churn_rate`, `trailing_churn_rate` | Daily churn rate, forecast vs trailing |
| `list_price`, `scenario_new_price`, `net_adds_delta`, `revenue_delta` | Price-change scenario |
| `risk_flag` | NEGATIVE_NET_ADDS, CHURN_SPIKE, TENTPOLE_CLIFF, USAGE_DECLINE, PRICE_SENSITIVE, OK |
| `recommended_action` | RETENTION_OFFER, ENGAGEMENT_PUSH, ANNUAL_PLAN_UPSELL, PROCEED_PRICE_CHANGE, HOLD_PRICE, MONITOR |
| `price_decision` | PROCEED_PRICE_CHANGE, HOLD_PRICE, NO_CHANGE_PLANNED |
| `reason_code`, `explanation` | Why |

## OKR summary columns

| Column | Role |
|--------|------|
| `forecast_horizon`, `scenario` | One row per horizon x (baseline, price_change) |
| `high_value_net_adds`, `high_value_paid_subs_end`, `high_value_gross_adds` | OKR 1: high-value subscriber growth |
| `hours_per_paid_sub_month`, `high_value_hours_per_paid_sub_month` | OKR 2: usage per paid sub |
| `total_net_adds`, `total_paid_subs_end`, `revenue` | Context |

## Metrics columns

| Column | Role |
|--------|------|
| `target` | gross_adds, churned_subs, hours_watched |
| `model` | Model name |
| `level` | overall, tier, acquisition_channel, distribution_partner, series_id, tentpole_period |
| `group` | Segment name when level ≠ overall |
| `wape`, `mape`, `mae`, `rmse`, `bias` | Metric values per row |

## Forecasts columns

| Column | Role |
|--------|------|
| `date`, `horizon_day` | Forecast date and step |
| `series_id`, `tier`, `acquisition_channel`, `distribution_partner` | Segment identity |
| `forecast_gross_adds`, `forecast_churned_subs`, `forecast_hours_watched` | Forecast targets |
| `model_<target>` | Model used per target |
| `opening_paid_subs`, `forecast_net_adds`, `forecast_paid_subs`, `forecast_hours_per_paid_sub` | Derived KPIs |
| `tentpole_flag`, `tentpole_name` | Content calendar overlay |

## Dashboard patterns

### Growth OKR strip

- **Stats**: high-value net adds and paid subs, hours per paid sub per month, revenue; baseline vs price change deltas from `okr_summary` at the primary horizon.
- **Source caption**: `outputs/okr_summary.csv` + snapshot date.

### Actions by tier

- **Groupby**: `tier` × `recommended_action` at one horizon (count rows).
- **Charts**: stacked `BarChart`; `UsageBar` per tier.

### Segment forecast explorer

- **Per segment**: daily net adds, paid subs and hours per paid sub from `segment-forecasts`.
- **Interaction**: `Select` for segment via `useCanvasState`.
- **Overlay**: mark tentpole days from `forecasts.tentpole_name`.

### Model accuracy comparison

- **Filter** `metrics` by `target` and `level='tier'` (or `acquisition_channel`); compare `wape` across `group`.

### Dense matrices

For tier × channel heatmaps, prefer **create-plot** (`chart.kind: heatmap`). Use a canvas when the user wants filtering or narrative layout.

## Refresh workflow

1. `python main.py` or `python main.py --quick`
2. Re-run `extract_canvas_data.py <recipe>`
3. Update inline consts in the `.canvas.tsx` (or regenerate the file)
4. Confirm TypeScript check passes

## TypeScript embedding tips

- Use `as const` on string literal arrays (`DATES`, tier names).
- Prefer plain objects/arrays over CSV-sized raw rows in the canvas file.
- Type extracted payloads with small local `type` aliases matching the JSON shape.
