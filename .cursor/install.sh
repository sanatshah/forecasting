#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the Retail Demand Forecasting &
# Optimization project. Safe to run repeatedly and against cached state.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$REPO_ROOT/retail_forecasting_optimization"
cd "$APP_DIR"

# System packages needed to build the venv and native wheels (statsmodels).
# No-op when the base image/snapshot already provides them.
if ! dpkg -s python3.12-venv >/dev/null 2>&1 \
  || ! dpkg -s python3.12-dev >/dev/null 2>&1 \
  || ! dpkg -s build-essential >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y python3.12-venv python3.12-dev build-essential
fi

# Python virtualenv + dependencies.
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

# Frontend dependencies (dashboard UI).
( cd frontend && npm install )

# Generate the pipeline outputs the dashboard API serves (outputs/*.csv).
# Deterministic on the shipped sample data; overwrites outputs/ each run.
./.venv/bin/python main.py

echo "install.sh: environment ready"
