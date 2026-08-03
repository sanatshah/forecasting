"""Tests for the dashboard FastAPI layer."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.dashboard import aggregations
from src.dashboard_api import app, get_config

client = TestClient(app)


@pytest.fixture(scope="module")
def has_outputs() -> bool:
    try:
        r = client.get("/api/summary")
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def has_holdout() -> bool:
    try:
        r = client.get("/api/holdout-forecasts")
        if r.status_code != 200:
            return False
        return bool(r.json().get("meta", {}).get("available"))
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


def test_holdout_forecasts_shape(has_holdout: bool):
    r = client.get("/api/holdout-forecasts")
    assert r.status_code == 200
    body = r.json()
    assert "meta" in body
    assert "dates" in body
    assert "actuals" in body
    assert "predictions" in body
    assert "metrics" in body
    assert "skuIds" in body
    assert "available" in body["meta"]

    if not has_holdout:
        assert body["meta"]["available"] is False
        assert body["dates"] == []
        return

    assert body["meta"]["available"] is True
    assert len(body["dates"]) == len(body["actuals"]) == len(body["predictions"])
    assert len(body["dates"]) > 0
    assert body["metrics"] is not None
    assert "mae" in body["metrics"]
    assert "wape" in body["metrics"]
    assert body["meta"]["snapshotDate"] == body["dates"][-1]


def test_holdout_forecasts_sku_filter(has_holdout: bool):
    if not has_holdout:
        pytest.skip("holdout_predictions.csv not available")
    listed = client.get("/api/holdout-forecasts").json()
    sku = listed["skuIds"][0]
    r = client.get(f"/api/holdout-forecasts?sku_id={sku}")
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["skuId"] == sku
    assert body["meta"]["available"] is True
    assert len(body["dates"]) > 0


def test_holdout_forecasts_unknown_sku(has_holdout: bool):
    if not has_holdout:
        pytest.skip("holdout_predictions.csv not available")
    r = client.get("/api/holdout-forecasts?sku_id=__missing_sku__")
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["available"] is True
    assert body["dates"] == []
    assert body["metrics"] is None


def test_holdout_forecasts_missing_file(tmp_path, monkeypatch):
    cfg = dict(get_config())
    cfg["paths"] = dict(cfg["paths"])
    missing = tmp_path / "holdout_predictions.csv"
    cfg["paths"]["holdout_predictions_csv"] = str(missing)

    body = aggregations.holdout_forecasts(cfg, sku_id="SKU0001")
    assert body["meta"]["available"] is False
    assert body["dates"] == []
    assert body["metrics"] is None
    assert "not found" in (body["meta"]["message"] or "").lower()


def test_holdout_forecasts_aggregation_and_metrics(tmp_path):
    cfg = dict(get_config())
    cfg["paths"] = dict(cfg["paths"])
    path = tmp_path / "holdout_predictions.csv"
    cfg["paths"]["holdout_predictions_csv"] = str(path)

    frame = pd.DataFrame(
        {
            "date": ["2025-12-01", "2025-12-01", "2025-12-02", "2025-12-02"],
            "sku_id": ["SKU_A", "SKU_A", "SKU_A", "SKU_B"],
            "location_id": ["L1", "L2", "L1", "L1"],
            "channel": ["store", "store", "store", "store"],
            "actual": [10.0, 5.0, 20.0, 100.0],
            "forecast": [8.0, 7.0, 16.0, 90.0],
        }
    )
    frame.to_csv(path, index=False)

    body = aggregations.holdout_forecasts(cfg, sku_id="SKU_A")
    assert body["meta"]["available"] is True
    assert body["dates"] == ["2025-12-01", "2025-12-02"]
    # Day 1 aggregates across locations: actual 15, forecast 15.
    assert body["actuals"] == [15.0, 20.0]
    assert body["predictions"] == [15.0, 16.0]
    assert body["metrics"] is not None
    # |15-15| + |20-16| / (15+20) = 4/35
    assert body["metrics"]["wape"] == pytest.approx(4 / 35, rel=1e-4)
    assert body["metrics"]["mae"] == pytest.approx(2.0, rel=1e-4)
    assert "SKU_A" in body["skuIds"]
    assert "SKU_B" in body["skuIds"]
    assert Path(body["meta"]["source"]).name == "holdout_predictions.csv"
