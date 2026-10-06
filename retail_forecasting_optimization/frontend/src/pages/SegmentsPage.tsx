import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
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
  ForecastTarget,
  ForecastViewMode,
  HoldoutForecastsResponse,
  SegmentForecast,
  SegmentForecastsResponse,
} from "../types";
import {
  CHART_AXIS,
  CHART_GRID,
  EmptyState,
  ErrorState,
  formatCount,
  formatSigned,
  KpiCard,
  LoadingState,
  Panel,
  TIER_COLORS,
  TOOLTIP_STYLE,
} from "../components/ui";

type ForwardMetric = "netAdds" | "paidSubs" | "hoursPerPaidSub" | "grossAdds" | "churnedSubs";

const FORWARD_METRICS: { id: ForwardMetric; label: string }[] = [
  { id: "netAdds", label: "Net adds" },
  { id: "paidSubs", label: "Paid subs" },
  { id: "hoursPerPaidSub", label: "Hours / paid sub" },
  { id: "grossAdds", label: "Gross adds" },
  { id: "churnedSubs", label: "Churn" },
];

const HOLDOUT_TARGETS: { id: ForecastTarget; label: string }[] = [
  { id: "gross_adds", label: "Gross adds" },
  { id: "churned_subs", label: "Churn" },
  { id: "hours_watched", label: "Hours watched" },
];

const VIEW_MODES: { id: ForecastViewMode; label: string }[] = [
  { id: "forward", label: "Forward" },
  { id: "holdout", label: "Holdout" },
];

function formatDateLabel(date: string) {
  return date.slice(5);
}

function formatTooltipValue(value: number) {
  return Math.abs(value) >= 100 ? formatCount(value) : value.toFixed(3);
}

function segmentLabel(s: SegmentForecast) {
  return `${s.tier} · ${s.channel}`;
}

export function SegmentsPage() {
  const [data, setData] = useState<SegmentForecastsResponse | null>(null);
  const [holdout, setHoldout] = useState<HoldoutForecastsResponse | null>(null);
  const [selected, setSelected] = useState("");
  const [metric, setMetric] = useState<ForwardMetric>("netAdds");
  const [target, setTarget] = useState<ForecastTarget>("gross_adds");
  const [viewMode, setViewMode] = useState<ForecastViewMode>("forward");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);
  const [holdoutUnavailable, setHoldoutUnavailable] = useState<string | null>(null);

  useEffect(() => {
    api
      .segmentForecasts()
      .then((res) => {
        setData(res);
        if (res.segments.length > 0) setSelected(res.segments[0].segmentId);
      })
      .catch(setError)
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setHoldout(null);
    setHoldoutUnavailable(null);
    api
      .holdoutForecasts(selected, target)
      .then(setHoldout)
      .catch((err: unknown) => {
        const detail =
          err instanceof ApiClientError && typeof err.detail === "object"
            ? (err.detail.message ?? "Holdout data unavailable")
            : "Failed to load holdout data";
        setHoldoutUnavailable(detail);
        setViewMode("forward");
      });
  }, [selected, target]);

  const handleViewModeChange = useCallback((mode: ForecastViewMode) => setViewMode(mode), []);

  const segment = useMemo(
    () => data?.segments.find((s) => s.segmentId === selected),
    [data, selected],
  );

  const forwardChartData = useMemo(() => {
    if (!data || !segment) return [];
    return data.dates.map((date, i) => ({
      date: formatDateLabel(date),
      value: segment[metric][i] ?? 0,
    }));
  }, [data, segment, metric]);

  const holdoutChartData = useMemo(() => {
    if (!holdout) return [];
    return holdout.dates.map((date, i) => ({
      date: formatDateLabel(date),
      Actual: holdout.actuals[i] ?? 0,
      Predicted: holdout.predictions[i] ?? 0,
    }));
  }, [holdout]);

  if (loading) return <LoadingState />;
  if (error) return <ErrorState error={error} />;
  if (!data || data.segments.length === 0) {
    return (
      <EmptyState
        title="No forecast data"
        message="Run the pipeline to generate segment forecasts."
        command="cd retail_forecasting_optimization && ./.venv/bin/python main.py --quick"
      />
    );
  }
  if (!segment) return null;

  const color = TIER_COLORS[segment.tier] ?? "#0089D0";
  const avgHours =
    segment.hoursPerPaidSub.reduce((a, b) => a + b, 0) / Math.max(segment.hoursPerPaidSub.length, 1);
  const holdoutAvailable = holdout != null && !holdoutUnavailable;
  const holdoutWape =
    holdout?.metrics.wape != null ? `${(holdout.metrics.wape * 100).toFixed(1)}%` : "—";
  const metricLabel = FORWARD_METRICS.find((m) => m.id === metric)?.label ?? metric;
  const targetLabel = HOLDOUT_TARGETS.find((t) => t.id === target)?.label ?? target;

  return (
    <>
      <h1 className="page-title">Segment Forecasts</h1>
      <p className="page-subtitle">
        {viewMode === "forward"
          ? `${data.meta.horizonDays}-day outlook (${data.meta.forecastStart} to ${data.meta.forecastEnd}); net adds = gross adds − churn`
          : `Holdout backtest ${holdout?.meta.holdoutStart ?? ""} to ${holdout?.meta.holdoutEnd ?? ""}`}
      </p>

      <div className="filter-bar">
        <div className="filter-group">
          <label htmlFor="segment-select">Segment</label>
          <select id="segment-select" value={selected} onChange={(e) => setSelected(e.target.value)}>
            {data.segments.map((s) => (
              <option key={s.segmentId} value={s.segmentId}>
                {segmentLabel(s)} ({formatCount(s.endingPaidSubs)} subs)
              </option>
            ))}
          </select>
        </div>

        {viewMode === "forward" ? (
          <div className="filter-group">
            <label htmlFor="metric-select">Metric</label>
            <select
              id="metric-select"
              value={metric}
              onChange={(e) => setMetric(e.target.value as ForwardMetric)}
            >
              {FORWARD_METRICS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>
        ) : (
          <div className="filter-group">
            <label htmlFor="target-select">Target</label>
            <select
              id="target-select"
              value={target}
              onChange={(e) => setTarget(e.target.value as ForecastTarget)}
            >
              {HOLDOUT_TARGETS.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.label}
                </option>
              ))}
            </select>
          </div>
        )}

        <div className="filter-group">
          <span className="filter-label">View</span>
          <div className="segmented-control" role="group" aria-label="Forecast view mode">
            {VIEW_MODES.map(({ id, label }) => (
              <button
                key={id}
                type="button"
                className={`segment-btn${viewMode === id ? " segment-btn-active" : ""}`}
                disabled={id === "holdout" && !holdoutAvailable}
                aria-pressed={viewMode === id}
                onClick={() => handleViewModeChange(id)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {holdoutUnavailable && (
        <p className="holdout-notice" role="status">
          Holdout overlay unavailable: {holdoutUnavailable}. Showing forward forecasts only.
        </p>
      )}

      <div className="kpi-grid">
        <KpiCard label="Opening paid subs" value={formatCount(segment.openingPaidSubs)} />
        <KpiCard label="Ending paid subs" value={formatCount(segment.endingPaidSubs)} />
        <KpiCard label="Net adds" value={formatSigned(segment.totalNetAdds)} unit="horizon" />
        <KpiCard label="Hours / paid sub" value={avgHours.toFixed(3)} unit="per day" />
        {holdoutAvailable && <KpiCard label={`${targetLabel} WAPE`} value={holdoutWape} />}
        <KpiCard label="Partner" value={segment.partner || "—"} />
      </div>

      <Panel
        title={`${segmentLabel(segment)} — ${viewMode === "forward" ? metricLabel : targetLabel}`}
        caption={
          viewMode === "forward"
            ? "Recursive multi-step forecast with the content calendar known in advance"
            : "Holdout actual vs best-model prediction"
        }
      >
        {viewMode === "forward" ? (
          <ResponsiveContainer width="100%" height={360}>
            <AreaChart data={forwardChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
              <XAxis dataKey="date" tick={CHART_AXIS} />
              <YAxis tick={CHART_AXIS} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={formatTooltipValue} />
              {metric === "netAdds" && <ReferenceLine y={0} stroke="#5d5d6c" />}
              <Area
                type="monotone"
                dataKey="value"
                name={metricLabel}
                stroke={color}
                fill={color}
                fillOpacity={0.25}
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          holdout && (
            <ResponsiveContainer width="100%" height={360}>
              <LineChart data={holdoutChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                <XAxis dataKey="date" tick={CHART_AXIS} />
                <YAxis tick={CHART_AXIS} />
                <Tooltip contentStyle={TOOLTIP_STYLE} formatter={formatTooltipValue} />
                <Legend />
                <Line type="monotone" dataKey="Actual" stroke="#f4f4f6" strokeWidth={2.5} dot={false} />
                <Line
                  type="monotone"
                  dataKey="Predicted"
                  stroke={color}
                  strokeWidth={2}
                  dot={false}
                  strokeDasharray="4 2"
                />
              </LineChart>
            </ResponsiveContainer>
          )
        )}
      </Panel>
    </>
  );
}
