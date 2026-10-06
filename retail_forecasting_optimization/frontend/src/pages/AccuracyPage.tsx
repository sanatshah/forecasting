import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import type { SegmentMetric, SegmentMetricsResponse } from "../types";
import {
  CHART_AXIS,
  CHART_GRID,
  ErrorState,
  formatLabel,
  LoadingState,
  Panel,
  TOOLTIP_STYLE,
} from "../components/ui";

const TARGET_LABELS: Record<string, string> = {
  gross_adds: "Gross adds",
  churned_subs: "Churn",
  hours_watched: "Hours watched",
};

function formatWapeTooltip(value: number) {
  return [`${value}%`, "WAPE"];
}

function toChartData(rows: SegmentMetric[]) {
  return rows.map((r) => ({ group: r.group, wapePct: +(r.wape * 100).toFixed(1) }));
}

function MetricTable({ title, rows }: { title: string; rows: SegmentMetric[] }) {
  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>{title}</th>
          <th>WAPE</th>
          <th>MAPE</th>
          <th>MAE</th>
          <th>RMSE</th>
          <th>Bias</th>
          <th>N</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.group}>
            <td>{formatLabel(r.group)}</td>
            <td>{(r.wape * 100).toFixed(1)}%</td>
            <td>{(r.mape * 100).toFixed(1)}%</td>
            <td>{r.mae.toFixed(2)}</td>
            <td>{r.rmse.toFixed(2)}</td>
            <td>{r.bias.toFixed(2)}</td>
            <td>{r.n.toLocaleString()}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function AccuracyPage() {
  const [data, setData] = useState<SegmentMetricsResponse | null>(null);
  const [target, setTarget] = useState<string | undefined>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    setLoading(true);
    api
      .segmentMetrics(target)
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [target]);

  if (loading && !data) return <LoadingState />;
  if (error) return <ErrorState error={error} />;
  if (!data) return null;

  const modelLabel = data.meta.model.replace(/_/g, " ");
  const targetLabel = TARGET_LABELS[data.meta.target] ?? data.meta.target;

  return (
    <>
      <h1 className="page-title">Forecast Accuracy</h1>
      <p className="page-subtitle">
        {targetLabel} holdout WAPE by tier and acquisition channel — {modelLabel} (snapshot{" "}
        {data.meta.snapshotDate})
      </p>

      <div className="filter-bar">
        <div className="filter-group">
          <label htmlFor="target-filter">Target</label>
          <select
            id="target-filter"
            value={data.meta.target}
            onChange={(e) => setTarget(e.target.value)}
          >
            {data.meta.targets.map((t) => (
              <option key={t} value={t}>
                {TARGET_LABELS[t] ?? t}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="chart-grid">
        <Panel title="WAPE by tier" caption="Weighted Absolute Percentage Error (lower is better)">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={toChartData(data.tiers)} margin={{ bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
              <XAxis dataKey="group" tick={CHART_AXIS} />
              <YAxis unit="%" tick={CHART_AXIS} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={formatWapeTooltip} />
              <Bar dataKey="wapePct" fill="#0089D0" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="WAPE by acquisition channel" caption="Same model and target">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={toChartData(data.channels)} margin={{ bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
              <XAxis dataKey="group" tick={CHART_AXIS} />
              <YAxis unit="%" tick={CHART_AXIS} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={formatWapeTooltip} />
              <Bar dataKey="wapePct" fill="#6460AA" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      <div style={{ marginTop: "1.5rem" }}>
        <Panel title="Detail" caption={`Model: ${modelLabel}`}>
          <div className="data-table-wrap">
            <MetricTable title="Tier" rows={data.tiers} />
            <MetricTable title="Channel" rows={data.channels} />
          </div>
        </Panel>
      </div>
    </>
  );
}
