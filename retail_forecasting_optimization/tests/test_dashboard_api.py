"""Tests for the dashboard FastAPI layer."""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.dashboard import aggregations
from src.dashboard_api import app
from src.utils import load_config

client = TestClient(app)


@pytest.fixture(scope="module")
def has_outputs() -> bool:
    try:
        r = client.get("/api/summary")
        return r.status_code == 200
    except Exception:
        return False


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_summary(has_outputs: bool):
    r = client.get("/api/summary")
    if not has_outputs:
        assert r.status_code == 404
        return
    body = r.json()
    assert "bestModel" in body
    assert "riskCounts" in body
    assert "meta" in body


def test_action_breakdown(has_outputs: bool):
    r = client.get("/api/action-breakdown")
    if not has_outputs:
        assert r.status_code == 404
        return
    body = r.json()
    assert "departments" in body
    assert "breakdown" in body


def test_recommendations(has_outputs: bool):
    r = client.get("/api/recommendations")
    if not has_outputs:
        assert r.status_code == 404
        return
    body = r.json()
    assert "rows" in body
    assert "filters" in body


def test_department_metrics(has_outputs: bool):
    r = client.get("/api/metrics/department")
    if not has_outputs:
        assert r.status_code == 404
        return
    body = r.json()
    assert "departments" in body
    assert "meta" in body


@pytest.fixture()
def holdout_csv(tmp_path):
    """Minimal holdout_predictions.csv for aggregation tests."""
    path = tmp_path / "holdout_predictions.csv"
    df = pd.DataFrame(
        {
            "date": ["2025-12-01", "2025-12-01", "2025-12-02", "2025-12-02"],
            "sku_id": ["SKU9001", "SKU9001", "SKU9001", "SKU9002"],
            "location_id": ["LOC01", "LOC02", "LOC01", "LOC01"],
            "channel": ["store", "online", "store", "store"],
            "department": ["Womens", "Womens", "Womens", "Home"],
            "class": ["Dresses", "Dresses", "Dresses", "Bedding"],
            "subclass": ["Casual", "Casual", "Casual", "Sheets"],
            "product_lifecycle_status": ["Core", "Core", "Core", "Core"],
            "series_id": ["s1", "s2", "s1", "s3"],
            "actual": [10.0, 5.0, 12.0, 8.0],
            "forecast": [9.0, 6.0, 11.0, 7.0],
        }
    )
    df.to_csv(path, index=False)
    return path


def test_holdout_forecasts_requires_sku_id():
    r = client.get("/api/holdout-forecasts")
    assert r.status_code == 422


def test_holdout_forecasts_aggregation(holdout_csv, config):
    cfg = dict(config)
    cfg["paths"] = dict(config["paths"])
    cfg["paths"]["holdout_predictions_csv"] = str(holdout_csv)

    result = aggregations.holdout_forecasts(cfg, "SKU9001")
    assert result["meta"]["skuId"] == "SKU9001"
    assert result["dates"] == ["2025-12-01", "2025-12-02"]
    assert result["actuals"] == [15.0, 12.0]
    assert result["predictions"] == [15.0, 11.0]
    assert result["metrics"]["mae"] == 0.5
    assert result["metrics"]["wape"] == round(1.0 / 27.0, 6)
    assert result["meta"]["snapshotDate"] == "2025-12-02"
    assert result["meta"]["holdoutStart"] == "2025-12-01"
    assert result["meta"]["holdoutEnd"] == "2025-12-02"


def test_holdout_forecasts_api_success(holdout_csv, config, monkeypatch):
    cfg = dict(config)
    cfg["paths"] = dict(config["paths"])
    cfg["paths"]["holdout_predictions_csv"] = str(holdout_csv)
    monkeypatch.setattr("src.dashboard_api._config", cfg)

    r = client.get("/api/holdout-forecasts?sku_id=SKU9001")
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["skuId"] == "SKU9001"
    assert body["dates"] == ["2025-12-01", "2025-12-02"]
    assert body["actuals"] == [15.0, 12.0]
    assert body["predictions"] == [15.0, 11.0]
    assert body["metrics"]["mae"] == 0.5
    assert body["metrics"]["wape"] == round(1.0 / 27.0, 6)


def test_holdout_forecasts_unknown_sku(holdout_csv, config, monkeypatch):
    cfg = dict(config)
    cfg["paths"] = dict(config["paths"])
    cfg["paths"]["holdout_predictions_csv"] = str(holdout_csv)
    monkeypatch.setattr("src.dashboard_api._config", cfg)

    r = client.get("/api/holdout-forecasts?sku_id=UNKNOWN-SKU")
    assert r.status_code == 404


def test_holdout_forecasts_missing_file(config, tmp_path, monkeypatch):
    cfg = load_config()
    cfg = dict(cfg)
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["holdout_predictions_csv"] = str(tmp_path / "missing_holdout.csv")
    monkeypatch.setattr("src.dashboard_api._config", cfg)

    r = client.get("/api/holdout-forecasts?sku_id=SKU9001")
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["message"] == "Pipeline outputs not found. Run the pipeline first."


def test_holdout_forecasts_api(has_outputs: bool):
    if not has_outputs:
        r = client.get("/api/holdout-forecasts?sku_id=SKU0001")
        assert r.status_code == 404
        return

    forecasts = client.get("/api/sku-forecasts?top_skus=1")
    assert forecasts.status_code == 200
    sku_id = forecasts.json()["skus"][0]["skuId"]

    r = client.get(f"/api/holdout-forecasts?sku_id={sku_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["skuId"] == sku_id
    assert len(body["dates"]) == len(body["actuals"]) == len(body["predictions"])
    assert body["dates"] == sorted(body["dates"])
    assert "mae" in body["metrics"]
    assert body["metrics"]["wape"] is None or isinstance(body["metrics"]["wape"], float)
