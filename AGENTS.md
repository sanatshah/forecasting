# AGENTS.md

## Cursor Cloud specific instructions

### What this repo is

The only runnable product in this repository is the **Retail Demand Forecasting &
Optimization** pipeline under `retail_forecasting_optimization/`. It is a
single-language **Python batch/CLI pipeline** (no web server, database, frontend,
or long-running services), despite workspace rules that mention FastAPI/React/
Spanner/BigQuery — none of that code exists here. `demo_prompts/` holds a synthetic
dataset generator and prompt text; it is not a service.

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
- Fast smoke run (4 series, ~10s): `./.venv/bin/python main.py --quick`
- Full pipeline (~70s): `./.venv/bin/python main.py`

There is **no linter configured** (no ruff/flake8/pylint config or dependency), so
there is no lint step to run.

### Gotchas

- Exit code `2` means data validation failed — inspect
  `outputs/data_quality_report.csv` (not a crash).
- Outputs (`outputs/*.csv`, `outputs/plots/*.png`, `data/processed/cleaned.csv`)
  are gitignored; do not commit them unless asked.
- Input data `data/sample_input.csv` already ships in the repo. If ever missing,
  regenerate from repo root:
  `python demo_prompts/IDE_demo_prompt/generate_dataset.py --out retail_forecasting_optimization/data/sample_input.csv --seed 42`.
