# Peacock Subscriber Forecasting & Scenarios

A modular Python pipeline that forecasts Peacock subscriber flows and usage,
then turns those forecasts into segment-level retention, engagement and pricing
decisions. It rolls everything up into two Growth OKRs:

1. **High-value subscriber growth**: net adds and ending paid subs for
   full-price, directly billed Premium and Premium Plus.
2. **Usage per paid sub**: hours watched per paid subscriber per month.

It runs locally on a synthetic sample today and is shaped for real billing and
engagement data at **tier x acquisition channel x day** grain.

---

## 1. Why a demand-forecasting stack fits subscribers

Retail demand forecasting and subscriber forecasting are the same problem:
predict a flow over time from history, seasonality and price. The pipeline
started as a retail model and maps across like this:

| Retail | Peacock |
|---|---|
| Units sold | Net subscribers by tier: gross adds minus churn (Premium, Premium Plus, Ad Tier) |
| Store / SKU | Segment: tier x acquisition channel (direct, app store, MVPD partner, retail bundle) with distribution partner as an attribute |
| Seasonality and promos | Content calendar and sports. Tentpoles such as NFL, the Olympics and big Bravo premieres drive signups and churn the way holidays drive retail |
| Price | List price per tier; a planned increase is a scenario input |
| Basket size | Usage per paid sub (hours watched per paid sub) |
| Inventory on hand | Paid subscriber base at start of day (`paid_subs_bod`) |
| Replenish / markdown recommendations | Retention offer / engagement push / annual-plan upsell / proceed or hold a price change |

Gross adds, churn and hours are each forecast as a non-negative flow; net adds,
paid subs and hours per paid sub are derived, which keeps forecasts coherent
(paid subs = opening base + cumulative net adds).

## 2. Setup

```bash
cd retail_forecasting_optimization

# macOS / Linux
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# Windows / PowerShell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`xgboost` and `lightgbm` are optional. If they cannot be installed, the system
falls back to scikit-learn's `HistGradientBoostingRegressor`.

Regenerate the synthetic sample (and its content calendar) from the repo root:

```bash
python demo_prompts/IDE_demo_prompt/generate_dataset.py \
  --out retail_forecasting_optimization/data/sample_input.csv --seed 42
```

This writes `data/sample_input.csv` (12 segments, 2024-01-01 to 2026-09-30)
and `data/content_calendar.csv` (tentpoles and holidays through 2026-12-31).

## 3. How to run

```bash
./.venv/bin/python main.py            # full pipeline
./.venv/bin/python main.py --quick    # fast smoke run on a few segments
./.venv/bin/python -m pytest -q       # tests
```

The run prints an executive summary (best model per target, Growth OKRs
baseline vs price change, risk and action counts) and writes artifacts to
`outputs/`. Exit code `2` means data validation failed; see
`outputs/data_quality_report.csv`.

### Web dashboard (React)

A Peacock-themed React dashboard reads live pipeline outputs through a FastAPI
API. Run the pipeline first so `outputs/*.csv` exist.

```bash
# Terminal 1 - API (from retail_forecasting_optimization/)
./.venv/bin/uvicorn src.dashboard_api:app --reload --port 8000

# Terminal 2 - UI
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Vite proxies `/api` to port 8000.

| Page | Content |
|------|---------|
| Overview | Growth OKR KPIs, baseline vs price-change net adds, segment risk, actions by tier |
| Segments | Segment selector, forward net adds / paid subs / hours per sub, holdout overlay per target |
| Scenarios | Filterable decision table: risk, action, price decision, net-add and revenue deltas, explanations |
| Accuracy | WAPE by tier and acquisition channel for the best model, per target |

API endpoints: `/api/health`, `/api/summary`, `/api/okr`,
`/api/action-breakdown`, `/api/segment-forecasts`,
`/api/holdout-forecasts?segment=&target=`,
`/api/recommendations?tier=&risk=&action=&horizon=`,
`/api/metrics/segment?target=`.

## 4. Input data schema

`data/sample_input.csv`: daily rows at `date x tier x acquisition_channel`.

| Column | Meaning |
|---|---|
| `date` | Calendar date |
| `tier` | Premium, Premium Plus, Ad Tier |
| `acquisition_channel`, `distribution_partner` | direct / app_store / mvpd_partner / retail_bundle and the partner (Peacock, Apple/Google, Xfinity/Spectrum, Retail Bundle) |
| `gross_adds`, `churned_subs` | Targets: new paid subs and cancellations that day |
| `paid_subs_bod`, `paid_subs_eod` | Paid base at start / end of day (`eod = bod + adds - churn`) |
| `hours_watched` | Target: hours streamed that day |
| `daily_active_subs` | Subs who streamed that day (reporting only) |
| `list_price`, `effective_price`, `discount_pct` | Monthly price and promo discount |
| `promo_flag`, `promo_name` | Acquisition offer (Black Friday, summer sale) |
| `tentpole_flag`, `tentpole_name`, `tentpole_type`, `tentpole_intensity` | Content calendar: sports / reality / event and expected pull |
| `price_increase_flag` | 1 on the day a list-price increase takes effect |
| `holiday_flag`, `fiscal_week`, `fiscal_month`, `fiscal_quarter`, `season` | Calendar |

`data/content_calendar.csv` holds the tentpole and holiday columns by date and
extends past the history, so tentpoles are **known future covariates** in the
forecast. Retail never had a reliable future promo calendar; here it is the
main structural advantage.

## 5. Output files

All under `outputs/`:

- `forecasts.csv`: daily forward forecasts per segment (28 days) with
  `forecast_gross_adds`, `forecast_churned_subs`, `forecast_hours_watched`,
  the model used for each, and derived `forecast_net_adds`,
  `forecast_paid_subs`, `forecast_hours_per_paid_sub`.
- `recommendations.csv`: one row per segment per horizon (7/14/28) with the
  baseline outlook, the price-change scenario, risk flag, action, price
  decision, reason code and explanation.
- `okr_summary.csv`: Growth OKRs per horizon, baseline vs price change.
- `model_metrics.csv`: per-target, per-model accuracy by slice.
- `holdout_predictions.csv`: holdout actual vs forecast for the best model,
  per target.
- `data_quality_report.csv`: every validation finding and fix applied.
- `plots/`: PNGs from `plot_specs/*.json` (actual vs forecast, WAPE by tier
  and channel, churn error by segment, net adds heatmap, net adds forecast,
  paid subs by tier, usage per paid sub, recommendation summary, high-value
  net adds baseline vs price change) plus feature importance.

Ad-hoc charts without a pipeline re-run:

```bash
python -m src.plotting list-datasets
python -m src.plotting describe okr_summary
python -m src.plotting render --spec outputs/adhoc_example.json
```

## 6. Model methodology

Each target (`gross_adds`, `churned_subs`, `hours_watched`) runs through the
same comparison independently:

- **Naive baselines**: last-7-day average, same weekday last week, seasonal
  naive (weekly).
- **Statistical**: moving average and optional SARIMAX (capped for runtime).
- **Machine learning**: one **global** gradient-boosting regressor across all
  segments on lag/rolling target history, price, promo and tentpole timing and
  uplift, calendar, the opening paid base, trailing churn rate and hours per
  sub, and segment categoricals. Multi-step forecasts are **recursive**; each
  step sees the full remaining future calendar so `days_until_next_tentpole`
  matches training.
- **Chronos (optional)**: `ChronosForecaster` (`src/model_chronos.py`) joins
  the backtest when `chronos-forecasting` and `torch` are installed.

Splitting is strictly time-based: the trailing `holdout_days` per segment are
held out. Target-derived features use `shift(1)` so the current day never
leaks; sibling targets are never model inputs.

## 7. Evaluation

Computed per target, overall and sliced by tier, acquisition channel,
distribution partner, segment, and tentpole vs non-tentpole days: **WAPE**
(selection metric), **MAPE**, **MAE**, **RMSE**, **Bias**, and forecast
accuracy = clip(1 - WAPE, 0, 1). The best model is picked per target.

## 8. Scenario engine

`src/scenario_engine.py` replaces the retail inventory/markdown optimizer. For
each segment and horizon:

- **Risk flag** (first that applies): `NEGATIVE_NET_ADDS`, `CHURN_SPIKE`
  (forecast churn rate above 1.25x trailing), `TENTPOLE_CLIFF` (a big tentpole
  just ended with nothing comparable ahead), `USAGE_DECLINE` (hours per sub
  more than 8% below trailing), `PRICE_SENSITIVE` (the planned price change
  loses on the objective), `OK`.
- **Action**: `RETENTION_OFFER`, `ANNUAL_PLAN_UPSELL`, `ENGAGEMENT_PUSH`,
  `HOLD_PRICE`, `PROCEED_PRICE_CHANGE`, `MONITOR`.
- **Price-change scenario**: from `scenarios.price_changes` in config. Churn
  and gross adds respond with constant elasticity, `(new / old) ** e`, scaled
  by channel (partner-billed channels feel less of a list-price change) and
  applied only to the share of the horizon on or after the effective date.
- **Objective**: `revenue - churn_penalty_months x churned x price`. The
  price decision is `PROCEED_PRICE_CHANGE` when the scenario objective beats
  baseline, otherwise `HOLD_PRICE`.

`build_okr_summary` rolls segments into the two Growth OKRs. High-value means
`okr.high_value_tiers` sold through `okr.high_value_channels`.

All thresholds, elasticities and weights live in `config/config.yaml`.

## 9. Known limitations

- The sample is synthetic. Elasticities and tentpole intensities are
  assumptions; calibrate them from past price increases and event cohorts.
- Paid subs are an approximation: opening base plus forecast flows. Plan
  switches between tiers and reactivations are not modelled separately.
- The holdout backtest feeds the actual `paid_subs_bod` as a covariate, which
  is slightly optimistic versus a true forward run where the base is itself
  forecast.
- Revenue is list price x average paid subs; promo discounts, partner revenue
  share and ad revenue are not included.
- Recursive forecasting rebuilds features per step; batch or vectorize it for
  large segment counts.

## 10. Future enhancements

- Cohort-based churn (tenure curves) instead of a single hazard per segment.
- Probabilistic forecasts (quantiles) for OKR confidence ranges.
- Hierarchical reconciliation across segment, tier and total.
- Ad-tier revenue (impressions per hour) as a third OKR input.
- A constrained optimizer over offer budget and price across tiers.

## 11. Project structure

```
retail_forecasting_optimization/
  README.md
  requirements.txt
  config/config.yaml
  data/{sample_input.csv, content_calendar.csv, processed/}
  notebooks/01_exploration.ipynb
  plot_specs/*.json          # builtin declarative charts
  src/{data_loader, data_validation, feature_engineering, model_baseline,
       model_ml, model_chronos, model_selection, forecasting_pipeline,
       scenario_engine, evaluation, visualization, explainability, utils,
       dashboard_api}.py
  src/dashboard/             # API aggregations
  src/plotting/              # PlotSpec engine (datasets, spec, renderer, CLI)
  frontend/                  # React + Vite dashboard
  tests/
  outputs/{forecasts.csv, recommendations.csv, okr_summary.csv,
           model_metrics.csv, holdout_predictions.csv, plots/}
  main.py
```
