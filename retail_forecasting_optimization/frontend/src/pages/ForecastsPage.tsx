import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import type {
  ForecastViewMode,
  HoldoutForecastsResponse,
  SkuForecastsResponse,
} from "../types";
import { ErrorState, KpiCard, LoadingState, Panel } from "../components/ui";

const SERIES_COLORS = ["#e21a2c", "#1a1a1a", "#1565c0", "#7b1fa2", "#2e7d32"];

function formatWape(wape: number | null | undefined): string {
  if (wape == null || Number.isNaN(wape)) return "—";
  return `${(wape * 100).toFixed(1)}%`;
}

export function ForecastsPage() {
  const [data, setData] = useState<SkuForecastsResponse | null>(null);
  const [holdout, setHoldout] = useState<HoldoutForecastsResponse | null>(null);
  const [selectedSku, setSelectedSku] = useState("");
  const [viewMode, setViewMode] = useState<ForecastViewMode>("forward");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    api
      .skuForecasts(8)
      .then((res) => {
        setData(res);
        if (res.skus.length > 0) {
          setSelectedSku(res.skus[0].skuId);
        }
      })
      .catch(setError)
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selectedSku) {
      setHoldout(null);
      return;
    }
    let cancelled = false;
    api
      .holdoutForecasts(selectedSku)
      .then((res) => {
        if (!cancelled) setHoldout(res);
      })
      .catch(() => {
        // Graceful fallback: keep forward-only view when holdout API fails.
        if (!cancelled) {
          setHoldout({
            meta: {
              recipe: "holdout-forecasts",
              available: false,
              source: null,
              skuId: selectedSku,
              snapshotDate: null,
              holdoutStart: null,
              holdoutEnd: null,
              message: "Holdout data unavailable.",
            },
            dates: [],
            actuals: [],
            predictions: [],
            metrics: null,
            skuIds: [],
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedSku]);

  const sku = useMemo(
    () => data?.skus.find((s) => s.skuId === selectedSku),
    [data, selectedSku],
  );

  const holdoutAvailable =
    Boolean(holdout?.meta.available) && (holdout?.dates.length ?? 0) > 0;

  useEffect(() => {
    if (!holdoutAvailable && viewMode !== "forward") {
      setViewMode("forward");
    }
  }, [holdoutAvailable, viewMode]);

  const forwardChartData = useMemo(() => {
    if (!data || !sku) return [];
    return data.dates.map((date, i) => {
      const point: Record<string, string | number> = {
        date: date.slice(5),
        fullDate: date,
        Total: sku.aggregate[i] ?? 0,
      };
      for (const s of sku.series) {
        point[`${s.location} / ${s.channel}`] = s.data[i] ?? 0;
      }
      return point;
    });
  }, [data, sku]);

  const holdoutChartData = useMemo(() => {
    if (!holdoutAvailable || !holdout) return [];
    return holdout.dates.map((date, i) => ({
      date: date.slice(5),
      fullDate: date,
      Actual: holdout.actuals[i] ?? 0,
      Predicted: holdout.predictions[i] ?? 0,
    }));
  }, [holdout, holdoutAvailable]);

  const combinedChartData = useMemo(() => {
    if (!data || !sku || !holdoutAvailable || !holdout) return [];
    const byDate = new Map<string, Record<string, string | number | null>>();

    for (let i = 0; i < holdout.dates.length; i++) {
      const date = holdout.dates[i];
      byDate.set(date, {
        date: date.slice(5),
        fullDate: date,
        Actual: holdout.actuals[i] ?? null,
        Forecast: null,
      });
    }
    for (let i = 0; i < data.dates.length; i++) {
      const date = data.dates[i];
      const existing = byDate.get(date) ?? {
        date: date.slice(5),
        fullDate: date,
        Actual: null,
        Forecast: null,
      };
      existing.Forecast = sku.aggregate[i] ?? null;
      byDate.set(date, existing);
    }

    return Array.from(byDate.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([, point]) => point);
  }, [data, sku, holdout, holdoutAvailable]);

  const snapshotLabel = holdout?.meta.snapshotDate
    ? holdout.meta.snapshotDate.slice(5)
    : null;

  if (loading) return <LoadingState />;
  if (error) return <ErrorState error={error} />;
  if (!data || !sku) return null;

  const subtitle =
    viewMode === "holdout" && holdoutAvailable
      ? `Holdout actual vs predicted — ${holdout?.meta.holdoutStart} to ${holdout?.meta.holdoutEnd}`
      : viewMode === "combined" && holdoutAvailable
        ? `Holdout actuals + forward forecast (divider at ${holdout?.meta.snapshotDate})`
        : `Forward demand curves — ${data.meta.horizonDays}-day horizon (${data.meta.forecastStart} to ${data.meta.forecastEnd})`;

  return (
    <>
      <h1 className="page-title">SKU Forecasts</h1>
      <p className="page-subtitle">{subtitle}</p>

      <div className="filter-bar">
        <div className="filter-group">
          <label htmlFor="sku-select">SKU</label>
          <select
            id="sku-select"
            value={selectedSku}
            onChange={(e) => setSelectedSku(e.target.value)}
          >
            {data.skus.map((s) => (
              <option key={s.skuId} value={s.skuId}>
                {s.skuId} — {s.department} ({s.totalUnits.toLocaleString()} units)
              </option>
            ))}
          </select>
        </div>

        <div className="filter-group">
          <span className="filter-label">View</span>
          <div className="view-toggle" role="group" aria-label="Forecast view mode">
            <button
              type="button"
              className={viewMode === "forward" ? "active" : ""}
              onClick={() => setViewMode("forward")}
            >
              Forward
            </button>
            <button
              type="button"
              className={viewMode === "holdout" ? "active" : ""}
              onClick={() => setViewMode("holdout")}
              disabled={!holdoutAvailable}
              title={
                holdoutAvailable
                  ? "Holdout actual vs predicted"
                  : (holdout?.meta.message ?? "Holdout data unavailable")
              }
            >
              Holdout
            </button>
            <button
              type="button"
              className={viewMode === "combined" ? "active" : ""}
              onClick={() => setViewMode("combined")}
              disabled={!holdoutAvailable}
              title={
                holdoutAvailable
                  ? "Holdout actuals with forward forecast"
                  : (holdout?.meta.message ?? "Holdout data unavailable")
              }
            >
              Combined
            </button>
          </div>
        </div>
      </div>

      <div className="kpi-grid">
        <KpiCard label="Total Units" value={sku.totalUnits.toLocaleString()} />
        <KpiCard label="Avg Daily" value={sku.avgDaily} unit="units" />
        <KpiCard label="Department" value={sku.department} />
        <KpiCard label="Lifecycle" value={sku.lifecycle || "—"} />
        {holdoutAvailable && holdout?.metrics && (
          <KpiCard
            label="Holdout WAPE"
            value={formatWape(holdout.metrics.wape)}
          />
        )}
      </div>

      {viewMode === "forward" && (
        <Panel
          title={`Forecast: ${sku.skuId}`}
          caption={`${sku.class} / ${sku.subclass} — top location/channel series`}
        >
          <ResponsiveContainer width="100%" height={360}>
            <LineChart data={forwardChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e8e8e8" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="Total"
                stroke="#e21a2c"
                strokeWidth={2.5}
                dot={false}
              />
              {sku.series.map((s, i) => (
                <Line
                  key={`${s.location}-${s.channel}`}
                  type="monotone"
                  dataKey={`${s.location} / ${s.channel}`}
                  stroke={SERIES_COLORS[(i + 1) % SERIES_COLORS.length]}
                  strokeWidth={1.5}
                  dot={false}
                  strokeDasharray="4 2"
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </Panel>
      )}

      {viewMode === "holdout" && holdoutAvailable && (
        <Panel
          title={`Holdout: ${sku.skuId}`}
          caption={`Actual vs predicted — WAPE ${formatWape(holdout?.metrics?.wape)} · MAE ${
            holdout?.metrics?.mae?.toFixed(2) ?? "—"
          }`}
        >
          <ResponsiveContainer width="100%" height={360}>
            <LineChart data={holdoutChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e8e8e8" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="Actual"
                stroke="#1a1a1a"
                strokeWidth={2.5}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="Predicted"
                stroke="#e21a2c"
                strokeWidth={2}
                dot={false}
                strokeDasharray="6 3"
              />
            </LineChart>
          </ResponsiveContainer>
        </Panel>
      )}

      {viewMode === "combined" && holdoutAvailable && (
        <Panel
          title={`Combined: ${sku.skuId}`}
          caption="Solid = holdout actuals · Dashed = forward forecast · Vertical line = snapshot date"
        >
          <ResponsiveContainer width="100%" height={360}>
            <LineChart data={combinedChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e8e8e8" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              {snapshotLabel && (
                <ReferenceLine
                  x={snapshotLabel}
                  stroke="#5c5c5c"
                  strokeDasharray="2 2"
                  label={{
                    value: "Snapshot",
                    position: "insideTopRight",
                    fill: "#5c5c5c",
                    fontSize: 11,
                  }}
                />
              )}
              <Line
                type="monotone"
                dataKey="Actual"
                stroke="#1a1a1a"
                strokeWidth={2.5}
                dot={false}
                connectNulls={false}
              />
              <Line
                type="monotone"
                dataKey="Forecast"
                stroke="#e21a2c"
                strokeWidth={2}
                dot={false}
                strokeDasharray="6 3"
                connectNulls={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </Panel>
      )}
    </>
  );
}
