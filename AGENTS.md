# AGENTS.md

## Cursor Cloud specific instructions

### What this repo is

The only runnable product in this repository is the **Peacock Subscriber
Forecasting & Scenarios** pipeline under `retail_forecasting_optimization/` (the
directory name is a holdover from the retail model it was adapted from). It is
primarily a **Python batch/CLI pipeline** that forecasts gross adds, churn and
hours watched per segment (tier x acquisition channel), derives net adds, paid
subs and usage per paid sub, and runs a price-change scenario engine. An optional
**Peacock-themed React dashboard** (`frontend/` + FastAPI in
`src/dashboard_api.py`) reads live `outputs/` CSVs. `demo_prompts/` holds a
synthetic dataset generator and prompt text; it is not a service.

### Environment

- Python 3.12 with a virtualenv at `retail_forecasting_optimization/.venv`
  (created by the startup update script; the system package `python3.12-venv` is
  baked into the VM snapshot).
- All commands run from `retail_forecasting_optimization/` using the venv Python:
  `./.venv/bin/python`. Dependencies are in `requirements.txt`.
- `xgboost`/`lightgbm` are optional; the pipeline auto-falls back to sklearn
  `HistGradientBoostingRegressor` (both install fine on Python 3.12 here).

### Run / test commands

Run these from `retail_forecasting_optimization/` (see `README.md` and the
`run-pipeline` skill for full details):

- Tests: `./.venv/bin/python -m pytest -q`
- Fast smoke run (4 segments, all 3 targets): `./.venv/bin/python main.py --quick`
- Full pipeline (12 segments x 3 targets): `./.venv/bin/python main.py`
- Dashboard API: `./.venv/bin/uvicorn src.dashboard_api:app --reload --port 8000`
- Dashboard UI: `cd frontend && npm install && npm run dev` (see README)

There is **no linter configured** (no ruff/flake8/pylint config or dependency), so
there is no lint step to run.

### Gotchas

- Exit code `2` means data validation failed — inspect
  `outputs/data_quality_report.csv` (not a crash).
- Outputs (`outputs/*.csv`, `outputs/plots/*.png`, `data/processed/cleaned.csv`)
  are gitignored; do not commit them unless asked.
- Input data `data/sample_input.csv` and `data/content_calendar.csv` ship in the
  repo. If either is missing, regenerate both from repo root:
  `python demo_prompts/IDE_demo_prompt/generate_dataset.py --out retail_forecasting_optimization/data/sample_input.csv --seed 42`.
- The pipeline loops over `data.targets`; per-target artifacts carry a `target`
  column (`model_metrics.csv`, `holdout_predictions.csv`).
