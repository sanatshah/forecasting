import { useCallback, useEffect, useMemo, useState } from "react";
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
import { api, ApiClientError } from "../api/client";
import type {
  ForecastViewMode,
  HoldoutForecastsResponse,
  SkuForecastsResponse,
} from "../types";
import { EmptyState, ErrorState, KpiCard, LoadingState, Panel } from "../components/ui";

const SERIES_COLORS = ["#e21a2c", "#1a1a1a", "#1565c0", "#7b1fa2", "#2e7d32"];
const VIEW_MODES: { id: ForecastViewMode; label: string }[] = [
  { id: "forward", label: "Forward" },
  { id: "holdout", label: "Holdout" },
  { id: "combined", label: "Combined" },
];

const SNAPSHOT_LINE_LABEL = {
  value: "Snapshot",
  position: "insideTopRight" as const,
  fill: "#5c5c5c",
  fontSize: 11,
};

function formatDateLabel(date: string) {
  return date.slice(5);
}

export function ForecastsPage() {
  const [data, setData] = useState<SkuForecastsResponse | null>(null);
  const [holdout, setHoldout] = useState<HoldoutForecastsResponse | null>(null);
  const [selectedSku, setSelectedSku] = useState("");
  const [viewMode, setViewMode] = useState<ForecastViewMode>("forward");
  const [userSetMode, setUserSetMode] = useState(false);
  const [loading, setLoading] = useState(true);
  const [holdoutLoading, setHoldoutLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [holdoutUnavailable, setHoldoutUnavailable] = useState<string | null>(null);

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
      setHoldoutUnavailable(null);
      return;
    }

    // Drop prior SKU holdout immediately so charts/KPIs cannot mix identities.
    setUserSetMode(false);
    setHoldout(null);
    setHoldoutUnavailable(null);
    setHoldoutLoading(true);

    let cancelled = false;
    const skuForRequest = selectedSku;

    api
      .holdoutForecasts(skuForRequest)
      .then((res) => {
        if (cancelled || res.meta.skuId !== skuForRequest) return;
        setHoldout(res);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setHoldout(null);
        if (err instanceof ApiClientError && err.status === 404) {
          const detail =
            typeof err.detail === "object"
              ? (err.detail.message ?? "Holdout data unavailable")
              : String(err.detail);
          setHoldoutUnavailable(detail);
          setViewMode("forward");
          return;
        }
        setHoldoutUnavailable("Failed to load holdout data");
        setViewMode("forward");
      })
      .finally(() => {
        if (!cancelled) setHoldoutLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedSku]);

  useEffect(() => {
    if (holdout && !userSetMode && !holdoutUnavailable) {
      setViewMode("combined");
    }
  }, [holdout, userSetMode, holdoutUnavailable]);

  const handleViewModeChange = useCallback((mode: ForecastViewMode) => {
    setUserSetMode(true);
    setViewMode(mode);
  }, []);

  const sku = useMemo(
    () => data?.skus.find((s) => s.skuId === selectedSku),
    [data, selectedSku],
  );

  const holdoutAvailable =
    holdout != null &&
    !holdoutUnavailable &&
    holdout.meta.skuId === selectedSku;

  const forwardChartData = useMemo(() => {
    if (!data || !sku) return [];
    return data.dates.map((date, i) => {
      const point: Record<string, string | number | null> = {
        date: formatDateLabel(date),
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
    if (!holdout) return [];
    return holdout.dates.map((date, i) => ({
      date: formatDateLabel(date),
      fullDate: date,
      Actual: holdout.actuals[i] ?? 0,
      Predicted: holdout.predictions[i] ?? 0,
    }));
  }, [holdout]);

  const combinedChartData = useMemo(() => {
    if (!data || !sku || !holdout) {
      return { points: [] as Record<string, string | number | null>[], snapshotDate: null as string | null };
    }

    const points: Record<string, string | number | null>[] = [];
    const snapshotDate = holdout.meta.snapshotDate;

    for (let i = 0; i < holdout.dates.length; i++) {
      points.push({
        date: formatDateLabel(holdout.dates[i]),
        fullDate: holdout.dates[i],
        Actual: holdout.actuals[i] ?? 0,
        "Holdout Predicted": holdout.predictions[i] ?? 0,
        "Forward Forecast": null,
      });
    }

    for (let i = 0; i < data.dates.length; i++) {
      points.push({
        date: formatDateLabel(data.dates[i]),
        fullDate: data.dates[i],
        Actual: null,
        "Holdout Predicted": null,
        "Forward Forecast": sku.aggregate[i] ?? 0,
      });
    }

    points.sort((a, b) => String(a.fullDate).localeCompare(String(b.fullDate)));

    return { points, snapshotDate };
  }, [data, sku, holdout]);

  if (loading) return <LoadingState />;
  if (error) return <ErrorState error={error} />;
  if (!data || data.skus.length === 0) {
    return (
      <EmptyState
        title="No forecast data"
        message="Run the pipeline to generate forward forecasts."
        command="cd retail_forecasting_optimization && ./.venv/bin/python main.py --quick"
      />
    );
  }
  if (!sku) return null;

  const snapshotLabel = holdout?.meta.snapshotDate ?? data.meta.snapshotDate;
  const holdoutWape =
    holdout?.metrics.wape != null ? `${(holdout.metrics.wape * 100).toFixed(1)}%` : "—";

  const subtitle =
    viewMode === "holdout" && holdout
      ? `Holdout evaluation — ${holdout.meta.holdoutStart} to ${holdout.meta.holdoutEnd} (snapshot ${snapshotLabel})`
      : viewMode === "combined" && holdout
        ? `Actual vs predicted holdout + forward horizon (snapshot ${snapshotLabel})`
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
          <div className="segmented-control" role="group" aria-label="Forecast view mode">
            {VIEW_MODES.map(({ id, label }) => {
              const disabled =
                (id === "holdout" || id === "combined") &&
                (!holdoutAvailable || holdoutLoading);
              return (
                <button
                  key={id}
                  type="button"
                  className={`segment-btn${viewMode === id ? " segment-btn-active" : ""}`}
                  disabled={disabled}
                  aria-pressed={viewMode === id}
                  onClick={() => handleViewModeChange(id)}
                >
                  {label}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {holdoutUnavailable && (
        <p className="holdout-notice" role="status">
          Holdout overlay unavailable: {holdoutUnavailable}. Showing forward forecasts only.
        </p>
      )}

      <div className="kpi-grid">
        <KpiCard label="Total Units" value={sku.totalUnits.toLocaleString()} />
        <KpiCard label="Avg Daily" value={sku.avgDaily} unit="units" />
        {holdoutAvailable && (
          <KpiCard label="Holdout WAPE" value={holdoutWape} />
        )}
        <KpiCard label="Department" value={sku.department} />
        <KpiCard label="Lifecycle" value={sku.lifecycle || "—"} />
      </div>

      <Panel
        title={`Forecast: ${sku.skuId}`}
        caption={
          viewMode === "forward"
            ? `${sku.class} / ${sku.subclass} — top location/channel series`
            : viewMode === "holdout"
              ? `${sku.class} / ${sku.subclass} — holdout actual vs predicted`
              : `${sku.class} / ${sku.subclass} — holdout actuals and forward forecast`
        }
      >
        {viewMode === "forward" && (
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
        )}

        {viewMode !== "forward" && holdoutLoading && <LoadingState />}

        {viewMode === "holdout" && holdoutAvailable && !holdoutLoading && (
          <ResponsiveContainer width="100%" height={360}>
            <LineChart data={holdoutChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e8e8e8" />
              <XAxis dataKey="fullDate" tick={{ fontSize: 11 }} tickFormatter={formatDateLabel} />
              <YAxis />
              <Tooltip labelFormatter={formatDateLabel} />
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
                strokeDasharray="4 2"
              />
            </LineChart>
          </ResponsiveContainer>
        )}

        {viewMode === "combined" &&
          holdoutAvailable &&
          !holdoutLoading &&
          combinedChartData.points.length > 0 && (
          <ResponsiveContainer width="100%" height={360}>
            <LineChart data={combinedChartData.points}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e8e8e8" />
              <XAxis dataKey="fullDate" tick={{ fontSize: 11 }} tickFormatter={formatDateLabel} />
              <YAxis />
              <Tooltip labelFormatter={formatDateLabel} />
              <Legend />
              {combinedChartData.snapshotDate && (
                <ReferenceLine
                  x={combinedChartData.snapshotDate}
                  stroke="#9a9a9a"
                  strokeDasharray="6 4"
                  label={SNAPSHOT_LINE_LABEL}
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
                dataKey="Holdout Predicted"
                stroke="#e21a2c"
                strokeWidth={2}
                dot={false}
                strokeDasharray="4 2"
                connectNulls={false}
              />
              <Line
                type="monotone"
                dataKey="Forward Forecast"
                stroke="#1565c0"
                strokeWidth={2}
                dot={false}
                strokeDasharray="4 2"
                connectNulls={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </Panel>
    </>
  );
}
