"""Tests for the dashboard FastAPI layer over synthetic pipeline outputs."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.dashboard import aggregations
from src.dashboard_api import app

client = TestClient(app)

SEGMENTS = [
    ("Premium|direct", "Premium", "direct", "Peacock", 50_000),
    ("Ad Tier|app_store", "Ad Tier", "app_store", "Apple/Google", 30_000),
]


def _rec_row(sid: str, tier: str, channel: str, partner: str, opening: int, h: int) -> Dict[str, Any]:
    negative = tier == "Premium"
    net = (-20.0 if negative else 15.0) * h
    return {
        "date": "2026-09-30",
        "forecast_start": "2026-10-01",
        "forecast_end": (pd.Timestamp("2026-10-01") + pd.Timedelta(days=h - 1)).strftime("%Y-%m-%d"),
        "series_id": sid,
        "tier": tier,
        "acquisition_channel": channel,
        "distribution_partner": partner,
        "forecast_horizon": h,
        "opening_paid_subs": opening,
        "forecast_gross_adds": 100.0 * h,
        "forecast_churned_subs": 100.0 * h - net,
        "forecast_net_adds": net,
        "ending_paid_subs": opening + net,
        "avg_paid_subs": opening + net / 2,
        "forecast_hours_watched": opening * h * 1.0,
        "hours_per_paid_sub_month": 30.0,
        "forecast_churn_rate": 0.003,
        "trailing_churn_rate": 0.002,
        "usage_change_pct": -0.01,
        "tentpole_days": 4,
        "list_price": 10.99 if negative else 4.99,
        "scenario_new_price": 12.99 if negative else 4.99,
        "scenario_gross_adds": 100.0 * h,
        "scenario_churned_subs": 100.0 * h - net,
        "scenario_net_adds": net - (5.0 if negative else 0.0),
        "scenario_ending_paid_subs": opening + net,
        "net_adds_delta": -5.0 if negative else 0.0,
        "baseline_revenue": 1000.0,
        "scenario_revenue": 1100.0 if negative else 1000.0,
        "revenue_delta": 100.0 if negative else 0.0,
        "risk_flag": "NEGATIVE_NET_ADDS" if negative else "OK",
        "recommended_action": "RETENTION_OFFER" if negative else "MONITOR",
        "price_decision": "PROCEED_PRICE_CHANGE" if negative else "NO_CHANGE_PLANNED",
        "objective_score": 900.0,
        "reason_code": "ADDS_BELOW_CHURN" if negative else "STABLE_GROWTH",
        "explanation": f"{tier} via {channel}.",
    }


def _metric_rows() -> List[Dict[str, Any]]:
    rows = []
    for target in ["gross_adds", "churned_subs"]:
        for model, base in [("global_ml", 0.10), ("seasonal_naive", 0.20)]:
            for level, group, bump in [
                ("overall", "__overall__", 0.0),
                ("tier", "Premium", 0.01),
                ("tier", "Ad Tier", 0.03),
                ("acquisition_channel", "direct", 0.02),
            ]:
                w = base + bump + (0.05 if target == "churned_subs" else 0.0)
                rows.append(
                    {"model": model, "target": target, "level": level, "group": group,
                     "wape": w, "mape": w, "mae": 1.0, "rmse": 1.5, "bias": 0.1,
                     "forecast_accuracy": 1 - w, "n": 28}
                )
    return rows


@pytest.fixture()
def outputs_config(tmp_path, config, monkeypatch):
    """Write a minimal set of pipeline outputs and point the API at them."""
    dates = pd.date_range("2026-10-01", periods=3, freq="D")
    forecasts = []
    for sid, tier, channel, partner, opening in SEGMENTS:
        net = -20.0 if tier == "Premium" else 15.0
        for i, d in enumerate(dates):
            paid = opening + net * (i + 1)
            forecasts.append(
                {"series_id": sid, "date": d, "tier": tier, "acquisition_channel": channel,
                 "distribution_partner": partner, "opening_paid_subs": opening,
                 "forecast_gross_adds": 100.0, "forecast_churned_subs": 100.0 - net,
                 "forecast_net_adds": net, "forecast_paid_subs": paid,
                 "forecast_hours_watched": paid * 1.0, "forecast_hours_per_paid_sub": 1.0}
            )

    recs = [_rec_row(*seg, h) for seg in SEGMENTS for h in (7, 28)]

    okr = []
    for h in (7, 28):
        for scenario, hv_net in [("baseline", -20.0 * h), ("price_change", -20.0 * h - 5)]:
            okr.append(
                {"forecast_horizon": h, "scenario": scenario, "forecast_start": "2026-10-01",
                 "forecast_end": "2026-10-28", "high_value_net_adds": hv_net,
                 "high_value_paid_subs_end": 50_000 + hv_net, "high_value_gross_adds": 100.0 * h,
                 "total_net_adds": hv_net + 15.0 * h, "total_paid_subs_end": 80_000.0,
                 "hours_per_paid_sub_month": 30.0, "high_value_hours_per_paid_sub_month": 31.0,
                 "revenue": 2000.0 if scenario == "baseline" else 2100.0}
            )

    holdout = []
    for target, scale in [("gross_adds", 1.0), ("churned_subs", 0.5)]:
        for sid, tier, channel, _, _ in SEGMENTS:
            for d, actual, forecast in [("2026-09-29", 10.0, 9.0), ("2026-09-30", 12.0, 11.0)]:
                holdout.append(
                    {"date": d, "target": target, "series_id": sid, "tier": tier,
                     "acquisition_channel": channel, "actual": actual * scale,
                     "forecast": forecast * scale}
                )

    frames = {
        "forecasts_csv": pd.DataFrame(forecasts),
        "recommendations_csv": pd.DataFrame(recs),
        "okr_summary_csv": pd.DataFrame(okr),
        "metrics_csv": pd.DataFrame(_metric_rows()),
        "holdout_predictions_csv": pd.DataFrame(holdout),
    }
    cfg = dict(config)
    cfg["paths"] = dict(config["paths"])
    for key, df in frames.items():
        path = tmp_path / f"{key}.csv"
        df.to_csv(path, index=False)
        cfg["paths"][key] = str(path)
    monkeypatch.setattr("src.dashboard_api._config", cfg)
    return cfg


@pytest.fixture()
def missing_config(tmp_path, config, monkeypatch):
    cfg = dict(config)
    cfg["paths"] = {k: str(tmp_path / f"missing_{k}.csv") for k in config["paths"]}
    monkeypatch.setattr("src.dashboard_api._config", cfg)
    return cfg


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_summary(outputs_config):
    r = client.get("/api/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["bestModel"] == "global_ml"
    assert body["bestModels"]["churned_subs"]["model"] == "global_ml"
    assert body["bestModels"]["gross_adds"]["wape"] == pytest.approx(0.10)
    assert body["okr"]["scenario"] == "baseline"
    assert body["okr"]["horizon"] == 28
    assert body["okrScenario"]["highValueNetAdds"] == pytest.approx(-565.0)
    assert body["riskCounts"] == {"NEGATIVE_NET_ADDS": 1, "OK": 1}
    assert body["meta"]["segmentCount"] == 2


def test_okr(outputs_config):
    r = client.get("/api/okr")
    assert r.status_code == 200
    body = r.json()
    assert len(body["rows"]) == 4
    assert body["meta"]["highValueTiers"] == ["Premium", "Premium Plus"]
    assert body["meta"]["priceChanges"][0]["tier"] == "Premium"


def test_action_breakdown(outputs_config):
    r = client.get("/api/action-breakdown")
    assert r.status_code == 200
    body = r.json()
    assert body["tiers"] == ["Ad Tier", "Premium"]
    assert body["breakdown"]["Premium"]["RETENTION_OFFER"] == 1
    assert sum(body["totals"].values()) == body["meta"]["grandTotal"] == 2


def test_segment_forecasts(outputs_config):
    r = client.get("/api/segment-forecasts")
    assert r.status_code == 200
    body = r.json()
    assert len(body["dates"]) == 3
    assert [s["segmentId"] for s in body["segments"]] == ["Premium|direct", "Ad Tier|app_store"]
    premium = body["segments"][0]
    assert premium["netAdds"] == [-20.0, -20.0, -20.0]
    assert premium["totalNetAdds"] == pytest.approx(-60.0)
    assert premium["endingPaidSubs"] == pytest.approx(50_000 - 60)
    assert len(premium["hoursPerPaidSub"]) == 3


def test_holdout_forecasts_aggregation(outputs_config):
    result = aggregations.holdout_forecasts(outputs_config, "Premium|direct", "churned_subs")
    assert result["meta"]["target"] == "churned_subs"
    assert result["meta"]["tier"] == "Premium"
    assert result["dates"] == ["2026-09-29", "2026-09-30"]
    assert result["actuals"] == [5.0, 6.0]
    assert result["predictions"] == [4.5, 5.5]
    assert result["meta"]["snapshotDate"] == "2026-09-30"


def test_holdout_forecasts_defaults_to_gross_adds(outputs_config):
    r = client.get("/api/holdout-forecasts", params={"segment": "Ad Tier|app_store"})
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["target"] == "gross_adds"
    assert body["actuals"] == [10.0, 12.0]


def test_holdout_forecasts_requires_segment():
    r = client.get("/api/holdout-forecasts")
    assert r.status_code == 422


def test_holdout_forecasts_unknown_segment(outputs_config):
    r = client.get("/api/holdout-forecasts", params={"segment": "Nope|direct"})
    assert r.status_code == 404


def test_holdout_forecasts_missing_file(missing_config):
    r = client.get("/api/holdout-forecasts", params={"segment": "Premium|direct"})
    assert r.status_code == 404
    assert r.json()["detail"]["message"] == "Pipeline outputs not found. Run the pipeline first."


def test_recommendations_default_horizon(outputs_config):
    r = client.get("/api/recommendations")
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["horizon"] == 28
    assert len(body["rows"]) == 2
    assert body["filters"]["horizons"] == [7, 28]
    row = next(x for x in body["rows"] if x["tier"] == "Premium")
    assert row["priceDecision"] == "PROCEED_PRICE_CHANGE"
    assert row["scenarioPrice"] == pytest.approx(12.99)


def test_recommendations_filters(outputs_config):
    r = client.get("/api/recommendations", params={"tier": "Ad Tier", "horizon": 7})
    assert r.status_code == 200
    body = r.json()
    assert [x["segmentId"] for x in body["rows"]] == ["Ad Tier|app_store"]
    assert body["rows"][0]["forecastHorizon"] == 7

    r = client.get("/api/recommendations", params={"risk": "NEGATIVE_NET_ADDS"})
    assert {x["riskFlag"] for x in r.json()["rows"]} == {"NEGATIVE_NET_ADDS"}


def test_metrics_segment_default_target(outputs_config):
    r = client.get("/api/metrics/segment")
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["target"] == "gross_adds"
    assert body["meta"]["model"] == "global_ml"
    assert [t["group"] for t in body["tiers"]] == ["Premium", "Ad Tier"]
    assert [c["group"] for c in body["channels"]] == ["direct"]


def test_metrics_segment_other_target(outputs_config):
    r = client.get("/api/metrics/segment", params={"target": "churned_subs"})
    assert r.status_code == 200
    assert r.json()["tiers"][0]["wape"] == pytest.approx(0.16)


def test_metrics_segment_unknown_target(outputs_config):
    r = client.get("/api/metrics/segment", params={"target": "nope"})
    assert r.status_code == 404


@pytest.mark.parametrize(
    "path",
    ["/api/summary", "/api/okr", "/api/action-breakdown", "/api/segment-forecasts",
     "/api/recommendations", "/api/metrics/segment"],
)
def test_missing_outputs_return_404_with_hint(missing_config, path):
    r = client.get(path)
    assert r.status_code == 404
    assert "main.py --quick" in r.json()["detail"]["hint"]
