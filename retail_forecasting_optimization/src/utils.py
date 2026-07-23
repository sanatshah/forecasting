"""Shared utilities: configuration loading, logging, and small helpers.

Keeping these in one place avoids duplicating boilerplate across modules and
ensures every module reads the *same* configuration object.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

# Project root = parent of the ``src`` directory that contains this file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

_LOGGER_CONFIGURED = False


def get_logger(name: str = "retail_forecasting") -> logging.Logger:
    """Return a module logger, configuring the root handler once.

    A single stream handler with a consistent format is attached the first
    time this is called so that all modules share the same log formatting.
    """
    global _LOGGER_CONFIGURED
    if not _LOGGER_CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        # Avoid duplicate handlers if reconfigured in notebooks.
        if not root.handlers:
            root.addHandler(handler)
        _LOGGER_CONFIGURED = True
    return logging.getLogger(name)


def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load the YAML configuration file into a plain dict.

    Parameters
    ----------
    path:
        Optional explicit path. Defaults to ``config/config.yaml`` at the
        project root.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {cfg_path} did not parse to a mapping.")
    return config


def resolve_path(relative_or_absolute: str | os.PathLike) -> Path:
    """Resolve a config path relative to the project root when not absolute."""
    p = Path(relative_or_absolute)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create a directory (and parents) if needed and return it as ``Path``."""
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_global_seed(seed: int = 42) -> None:
    """Seed numpy (and the ``PYTHONHASHSEED`` env var) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def safe_divide(
    numerator: np.ndarray | float,
    denominator: np.ndarray | float,
    fill: float = 0.0,
) -> np.ndarray:
    """Element-wise divide that returns ``fill`` where the denominator is 0.

    Used throughout metrics and optimization to avoid divide-by-zero warnings
    and NaNs when demand or price is zero.
    """
    numerator = np.asarray(numerator, dtype="float64")
    denominator = np.asarray(denominator, dtype="float64")
    out = np.full(np.broadcast(numerator, denominator).shape, fill, dtype="float64")
    mask = denominator != 0
    np.divide(numerator, denominator, out=out, where=mask)
    return out
