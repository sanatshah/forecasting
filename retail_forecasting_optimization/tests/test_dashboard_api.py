"""Tests for the dashboard FastAPI layer."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.dashboard_api import app

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
