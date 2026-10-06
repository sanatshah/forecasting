---
name: run-pipeline
description: Runs the Peacock subscriber forecasting and scenario pipeline in retail_forecasting_optimization. Use when the user asks to run the pipeline, regenerate forecasts/recommendations/OKRs, smoke-test the system, or refresh outputs/.
---

# Run Subscriber Forecasting Pipeline

## Scope

End-to-end pipeline in `retail_forecasting_optimization/`:

load (history + content calendar) → validate → per target (features → train/select → forecast) → derive net adds / paid subs / hours per paid sub → scenarios + Growth OKRs → plots → executive summary

Targets: `gross_adds`, `churned_subs`, `hours_watched` (`data.targets` in config).

Entry point: `main.py`. Config: `config/config.yaml`.

## Prerequisites

1. **Working directory**: `cd retail_forecasting_optimization`
2. **Input data**: `data/sample_input.csv` and `data/content_calendar.csv`. If missing, generate both from the repo root:

```bash
python demo_prompts/IDE_demo_prompt/generate_dataset.py \
  --out retail_forecasting_optimization/data/sample_input.csv --seed 42
```

3. **Virtual env** (create once):

```bash
# macOS / Linux
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# Windows / PowerShell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`lightgbm` and `xgboost` are optional; the pipeline falls back to sklearn `HistGradientBoostingRegressor` if they are unavailable.

## Run commands

Use the venv Python for all commands below.

| Goal | Command |
|------|---------|
| Full pipeline (12 segments x 3 targets) | `python main.py` |
| Fast smoke test (4 segments) | `python main.py --quick` |
| Custom config | `python main.py --config path/to/config.yaml` |
| Unit tests | `python -m pytest -q` |

**macOS / Linux** — prefix with `./.venv/bin/python` if the venv is not activated.

**Windows / PowerShell** — prefix with `.\.venv\Scripts\python.exe`.

## Workflow checklist

Copy and track progress:

```
Pipeline run:
- [ ] cd retail_forecasting_optimization
- [ ] Confirm data/sample_input.csv and data/content_calendar.csv exist
- [ ] Run python main.py (or --quick for smoke test)
- [ ] Confirm exit code 0 and executive summary printed
- [ ] Verify outputs/ artifacts (see below)
```

## Expected outputs

Written under `outputs/` (paths from `config/config.yaml`):

| File | Purpose |
|------|---------|
| `forecasts.csv` | Daily forward forecasts per segment: each target + derived net adds, paid subs, hours per paid sub |
| `recommendations.csv` | Segment x horizon: risk flag, action, price-change scenario, reason, explanation |
| `okr_summary.csv` | Growth OKRs per horizon, baseline vs price change |
| `model_metrics.csv` | Per-target, per-model evaluation metrics (`target` column) |
| `holdout_predictions.csv` | Holdout actual vs forecast for the best model, per target |
| `data_quality_report.csv` | Validation findings |
| `plots/*.png` | Charts from `plot_specs/*.json` (declarative plot engine) |

Also writes `data/processed/cleaned.csv`.

## Ad-hoc plots (after a run)

To create a new chart from a prompt without re-running the full pipeline, use the **create-plot** skill:

```bash
python -m src.plotting list-datasets
python -m src.plotting describe okr_summary
python -m src.plotting render --spec outputs/adhoc_example.json
```

Builtin charts live in `plot_specs/*.json` and are regenerated at the end of every pipeline run.

## Success criteria

- Exit code **0**
- Console prints **EXECUTIVE SUMMARY** with best model per target, Growth OKRs (baseline vs price change), and risk/action counts
- `outputs/recommendations.csv`, `outputs/forecasts.csv` and `outputs/okr_summary.csv` updated (check mtime)

## Failure handling

| Symptom | Action |
|---------|--------|
| Exit code **2**, "Validation failed" | Read `outputs/data_quality_report.csv`; fix input data or relax thresholds in `config/config.yaml` under `validation:` |
| Missing `sample_input.csv` / `content_calendar.csv` | Generate dataset (see Prerequisites) or point `--config` at a config with valid `paths.input_csv` / `paths.content_calendar_csv` |
| Import errors | Re-run `pip install -r requirements.txt` in `.venv` |
| Slow full run | Use `python main.py --quick` first; tune `models.sarimax.max_series` or trim `data.targets` |

## After the run

When the user wants analysis (not just execution):

- Summarize executive summary stats (best model per target, high-value net adds, usage per paid sub, price-change deltas)
- Point to `outputs/recommendations.csv` for the segment decision table and `outputs/okr_summary.csv` for OKRs
- Point to `outputs/plots/okr_high_value_net_adds.png` and `outputs/plots/recommendation_summary.png` for visuals

Do **not** commit generated `outputs/` CSVs or plots unless the user asks — they are gitignored.

## Config knobs (common changes)

Edit `config/config.yaml` without touching source:

- `paths.input_csv` / `paths.content_calendar_csv` — alternate inputs
- `data.targets` — which flows to forecast
- `forecast.horizons` / `forecast.primary_horizon` — forecast and scenario horizons
- `scenarios.price_changes` — planned price increases (tier, effective date, new price)
- `scenarios.elasticity.*` — churn / acquisition elasticity by tier and channel scale
- `scenarios.risk.*` — churn spike, usage decline and tentpole cliff thresholds
- `okr.high_value_tiers` / `okr.high_value_channels` — what counts as high-value
- `models.ml_backend_preference` — force sklearn if boosting libs fail to install
