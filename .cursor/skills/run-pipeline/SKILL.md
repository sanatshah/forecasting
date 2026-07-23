---
name: run-pipeline
description: Runs the retail demand forecasting and optimization pipeline in retail_forecasting_optimization. Use when the user asks to run the pipeline, regenerate forecasts/recommendations, smoke-test the system, or refresh outputs/.
---

# Run Retail Forecasting Pipeline

## Scope

End-to-end pipeline in `retail_forecasting_optimization/`:

load → validate → features → train/select → forecast → optimize → plots → executive summary

Entry point: `main.py`. Config: `config/config.yaml`.

## Prerequisites

1. **Working directory**: `cd retail_forecasting_optimization`
2. **Input data**: `data/sample_input.csv` (default in config). If missing, generate from the repo root:

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
| Full pipeline | `python main.py` |
| Fast smoke test (4 series) | `python main.py --quick` |
| Custom config | `python main.py --config path/to/config.yaml` |
| Unit tests | `python -m pytest -q` |

**macOS / Linux** — prefix with `./.venv/bin/python` if the venv is not activated.

**Windows / PowerShell** — prefix with `.\.venv\Scripts\python.exe`.

## Workflow checklist

Copy and track progress:

```
Pipeline run:
- [ ] cd retail_forecasting_optimization
- [ ] Confirm data/sample_input.csv exists
- [ ] Run python main.py (or --quick for smoke test)
- [ ] Confirm exit code 0 and executive summary printed
- [ ] Verify outputs/ artifacts (see below)
```

## Expected outputs

Written under `outputs/` (paths from `config/config.yaml`):

| File | Purpose |
|------|---------|
| `forecasts.csv` | Daily forward forecasts per series |
| `recommendations.csv` | Inventory/markdown recommendations |
| `model_metrics.csv` | Per-model evaluation metrics |
| `data_quality_report.csv` | Validation findings |
| `plots/*.png` | Actual vs forecast, risk heatmap, recommendation summary, etc. |

Also writes `data/processed/cleaned.csv`.

## Success criteria

- Exit code **0**
- Console prints **EXECUTIVE SUMMARY** with best model, WAPE, stockout/overstock counts
- `outputs/recommendations.csv` and `outputs/forecasts.csv` updated (check mtime)

## Failure handling

| Symptom | Action |
|---------|--------|
| Exit code **2**, "Validation failed" | Read `outputs/data_quality_report.csv`; fix input data or relax thresholds in `config/config.yaml` under `validation:` |
| Missing `sample_input.csv` | Generate dataset (see Prerequisites) or point `--config` at a config with a valid `paths.input_csv` |
| Import errors | Re-run `pip install -r requirements.txt` in `.venv` |
| Slow full run | Use `python main.py --quick` first; tune `models.sarimax.max_series` in config |

## After the run

When the user wants analysis (not just execution):

- Summarize executive summary stats (best model, WAPE, risk counts)
- Point to `outputs/recommendations.csv` for decision table
- Point to `outputs/plots/recommendation_summary.png` for visual overview

Do **not** commit generated `outputs/` CSVs or plots unless the user asks — they are gitignored.

## Config knobs (common changes)

Edit `config/config.yaml` without touching source:

- `paths.input_csv` — alternate input file
- `forecast.horizons` / `forecast.primary_horizon` — forecast and optimization horizons
- `optimization.*` — weeks-of-supply thresholds, markdown guardrails, elasticity
- `models.ml_backend_preference` — force sklearn if boosting libs fail to install
