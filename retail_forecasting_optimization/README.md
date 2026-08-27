# Retail Demand Forecasting & Optimization

A modular, production-oriented Python system that forecasts retail demand and
turns those forecasts into concrete inventory and markdown recommendations. It
is built to run locally on a sample dataset today and to scale to real retail
data at **UPC / SKU / Location / Channel / Day** grain later.

---

## 1. Project purpose

Retail demand is seasonal, promotional, intermittent, and sensitive to price,
markdowns, inventory, and holidays. This project provides an end-to-end
pipeline that:

1. Ingests time-series sales data.
2. Validates and cleans it, producing a data-quality report.
3. Engineers retail-specific, leakage-safe features.
4. Trains naive, statistical, and machine-learning forecasters.
5. Evaluates them with retail-appropriate metrics and selects the best.
6. Produces forward forecasts for 7, 14, and 28-day horizons.
7. Runs an optimization layer that recommends replenish / hold / transfer /
   reduce and markdown depth, respecting business guardrails.
8. Generates plots and plain-English explanations.

## 2. Retail use case

The system supports pricing, inventory, markdown, and replenishment decisions.
For each SKU/location/channel it answers:

- How much will we sell over the next 7/14/28 days?
- Are we heading for a stockout or an overstock?
- Should we mark down, and how deep, given margin guardrails and price
  elasticity?
- What is the expected sales and margin impact of the recommended action?

## 3. Setup instructions

```bash
cd retail_forecasting_optimization

# Windows / PowerShell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# macOS / Linux
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

`xgboost` and `lightgbm` are optional. If they cannot be installed on your
platform/Python version, the system automatically falls back to scikit-learn's
`HistGradientBoostingRegressor` (configurable in `config/config.yaml`).

## 4. How to run

```bash
# Full pipeline on the sample data
.\.venv\Scripts\python.exe main.py

# Fast smoke run on a few series
.\.venv\Scripts\python.exe main.py --quick

# Custom config
.\.venv\Scripts\python.exe main.py --config config/config.yaml

# Tests
.\.venv\Scripts\python.exe -m pytest -q
```

The run prints an executive summary and writes all artifacts to `outputs/`.

### Web dashboard (React)

A RetailStore-branded React dashboard reads live pipeline outputs via a FastAPI API.

**Prerequisites:** Run the pipeline first so `outputs/*.csv` exist.

```bash
# Terminal 1 — API (from retail_forecasting_optimization/)
./.venv/bin/pip install -r requirements.txt
./.venv/bin/uvicorn src.dashboard_api:app --reload --port 8000

# Terminal 2 — UI
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Vite proxies `/api` to the API on port 8000.

| Page | Content |
|------|---------|
| Overview | KPIs, risk distribution, actions by department |
| Recommendations | Filterable decision table with explanations |
| Forecasts | SKU selector, forward curves, and holdout actual vs predicted overlay |
| Accuracy | WAPE by department for the best model |

Production build: `cd frontend && npm run build` (output in `frontend/dist/`).

## 5. Input data schema

Daily rows at `date x sku_id x location_id x channel` grain:

| Column | Meaning |
|---|---|
| `date` | Calendar date |
| `sku_id`, `product_id` | SKU and parent product identifiers |
| `location_id`, `channel` | Store/DC id and sales channel (store/online) |
| `department`, `class`, `subclass` | Merchandise hierarchy |
| `units_sold` | Target: units sold that day |
| `sales_revenue` | Revenue that day |
| `regular_price`, `selling_price`, `markdown_pct` | Pricing |
| `promo_flag`, `promo_event_name` | Promotion indicator/name |
| `inventory_on_hand`, `inventory_in_transit` | Inventory position |
| `stockout_flag` | Stockout indicator |
| `holiday_flag` | Holiday indicator |
| `fiscal_week`, `fiscal_month`, `fiscal_quarter` | Fiscal calendar |
| `season`, `product_lifecycle_status` | Season, lifecycle (New/Core/End of Life) |

Missing columns are handled gracefully: validation records the gap and
downstream steps skip logic that needs the absent field.

## 6. Output files

All under `outputs/`:

- `forecasts.csv` - daily forward forecasts per series (28-day horizon), with a
  `horizon_day` index for rollups.
- `recommendations.csv` - the decision table (see schema below).
- `model_metrics.csv` - per-model, per-slice accuracy metrics.
- `holdout_predictions.csv` - holdout actual vs forecast for the best model.
- `data_quality_report.csv` - every validation finding and fix applied.
- `plots/` - PNGs from declarative specs in `plot_specs/*.json` (actual vs
  forecast, forecast by SKU/location, error/WAPE by department and channel,
  inventory risk heatmap, forecast distribution, recommendation summary) plus
  feature importance from explainability.
- `processed/cleaned.csv` (under `data/`) - the cleaned dataset.

Ad-hoc charts (no pipeline re-run) via the plot CLI:

```bash
python -m src.plotting list-datasets
python -m src.plotting describe forecasts
python -m src.plotting render --spec outputs/adhoc_example.json
```

Recommendation table columns: `date, sku_id, location_id, channel, department,
forecast_horizon, forecast_units, inventory_on_hand, weeks_of_supply,
risk_flag, recommended_action, recommended_markdown_pct, expected_sales,
expected_margin, objective_score, reason_code, explanation`.

## 7. Model methodology

- **Naive baselines**: last-7-day average, same-weekday-last-week, seasonal
  naive (weekly).
- **Statistical**: moving average, and optional SARIMAX per series (guarded and
  capped for runtime; falls back to a naive value on failure or when
  statsmodels is unavailable).
- **Machine learning**: a single **global** gradient-boosting regressor
  (LightGBM -> XGBoost -> sklearn HistGradientBoosting, first available) trained
  across all series on lag/rolling/price/promo/calendar/hierarchy/inventory
  features. Multi-step forecasts are produced **recursively**.
- **Chronos (optional)**: `ChronosForecaster` in `src/model_chronos.py`
  implements `AdvancedForecasterInterface` with Amazon Chronos-Bolt
  (`amazon/chronos-bolt-small` by default). It joins the WAPE backtest when
  `chronos-forecasting` and `torch` are installed; otherwise it is skipped.
  Explainability remains ML-based (Chronos has no feature importances).

Splitting is strictly **time-based**: the trailing `holdout_days` per series are
held out for evaluation, and the ML model is trained only on earlier rows.

## 8. Evaluation metrics

Computed overall and sliced by department, channel, location, SKU, lifecycle
status, and promo vs non-promo periods:

- **WAPE** (primary selection metric), **MAPE** (safe zero handling), **MAE**,
  **RMSE**, **Bias**, and a **forecast accuracy %** = clip(1 - WAPE, 0, 1).

The best model is chosen by lowest overall WAPE.

## 9. Optimization logic

For each series and horizon the engine computes **weeks of supply** from the
forecast and current inventory, then:

- **Inventory action**: REPLENISH (stockout risk), REDUCE_EXPOSURE (overstock),
  TRANSFER / HOLD in between, with `risk_flag` in
  {STOCKOUT, OVERSTOCK, WATCH_LOW, WATCH_HIGH, OK}.
- **Markdown / pricing**: searches candidate markdown depths and picks the one
  maximizing a transparent objective, using **constant-elasticity** demand
  response (configurable per department/class). Guardrails enforced: minimum
  margin over cost, maximum markdown depth, no negative price, selling price not
  above regular.
- **Reason codes**: HIGH_STOCK_LOW_DEMAND, LOW_STOCK_HIGH_DEMAND,
  PROMO_RESPONSE_STRONG/WEAK, STOCKOUT_RISK, OVERSTOCK_RISK, NORMAL_DEMAND.
- **Objective** (`objective_score`): maximize expected margin, penalize
  overstock and stockout distance from the healthy weeks-of-supply band. It is
  intentionally structured to be swapped for `scipy.optimize` or an LP later.

All thresholds, guardrails, elasticities, and weights live in
`config/config.yaml` - no business assumptions are hard-coded in source.

## 10. Known limitations

- The sample has no real future price/promo calendar, so forward covariates are
  carried forward from the last observed values (promo/holiday default to 0).
  Replace `build_future_frame` inputs with a real calendar in production.
- Cost is approximated from an assumed gross margin; wire in true unit cost when
  available for exact margin math.
- Elasticity is a simple constant-elasticity assumption; calibrate per
  department/class from historical price/volume when data allows.
- SARIMAX is capped to a few series for runtime on the sample.
- Recursive multi-step forecasting rebuilds features per step; for very large
  fleets, batch or vectorize inference.

## 11. Future enhancements

- Chronos-Bolt is wired via `AdvancedForecasterInterface`
  (`src/model_chronos.py`); optional follow-ups include Chronos-2 /
  TimesFM / PatchTST adapters and richer covariate support.
- Replace the greedy markdown search with a constrained optimizer
  (`scipy.optimize` / LP / MILP) across the assortment.
- Probabilistic forecasts (quantiles) to drive service-level safety stock.
- Hierarchical reconciliation across SKU -> product -> department rollups.
- Promotion-uplift modeling and cross-item cannibalization.
- Model registry, scheduled retraining, and a serving API.

## 12. Project structure

```
retail_forecasting_optimization/
  README.md
  requirements.txt
  config/config.yaml
  data/{sample_input.csv, processed/}
  notebooks/01_exploration.ipynb
  plot_specs/*.json          # builtin declarative charts
  src/{data_loader, data_validation, feature_engineering, model_baseline,
       model_ml, model_selection, forecasting_pipeline, optimization_engine,
       evaluation, visualization, explainability, utils}.py
  src/plotting/              # PlotSpec engine (datasets, spec, renderer, CLI)
  tests/{test_data_validation, test_feature_engineering, test_optimization_engine,
         test_plotting}.py
  outputs/{forecasts.csv, recommendations.csv, model_metrics.csv,
           holdout_predictions.csv, plots/}
  main.py
```
