import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import type { ActionBreakdownResponse, SummaryResponse } from "../types";
import { ErrorState, KpiCard, LoadingState, Panel } from "../components/ui";

const ACTION_COLORS: Record<string, string> = {
  HOLD: "#9a9a9a",
  REPLENISH: "#1565c0",
  REDUCE: "#e21a2c",
  TRANSFER: "#7b1fa2",
};

const RISK_COLORS: Record<string, string> = {
  STOCKOUT: "#e21a2c",
  OVERSTOCK: "#e65100",
  WATCH_HIGH: "#f57f17",
  WATCH_LOW: "#fbc02d",
  OK: "#2e7d32",
};

function formatModel(name: string) {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatPieLabel({ name, value }: { name: string; value: number }) {
  return `${name}: ${value}`;
}

export function OverviewPage() {
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [breakdown, setBreakdown] = useState<ActionBreakdownResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    Promise.all([api.summary(), api.actionBreakdown()])
      .then(([s, b]) => {
        setSummary(s);
        setBreakdown(b);
      })
      .catch(setError)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingState />;
  if (error) return <ErrorState error={error} />;
  if (!summary || !breakdown) return null;

  const riskData = Object.entries(summary.riskCounts).map(([name, value]) => ({
    name,
    value,
    fill: RISK_COLORS[name] ?? "#9a9a9a",
  }));

  const stackedData = breakdown.departments.map((dept) => {
    const row: Record<string, string | number> = { department: dept };
    for (const action of breakdown.actions) {
      row[action] = breakdown.breakdown[dept]?.[action] ?? 0;
    }
    return row;
  });

  return (
    <>
      <h1 className="page-title">Executive Overview</h1>
      <p className="page-subtitle">
        Demand forecast performance and inventory risk summary for snapshot{" "}
        {summary.meta.snapshotDate}
      </p>

      <div className="kpi-grid">
        <KpiCard label="Best Model" value={formatModel(summary.bestModel)} />
        <KpiCard
          label="WAPE"
          value={summary.bestWape != null ? `${(summary.bestWape * 100).toFixed(1)}%` : "—"}
        />
        <KpiCard label="Recommendations" value={summary.meta.recommendationCount} />
        <KpiCard label="Stockout Risk" value={summary.riskCounts.STOCKOUT ?? 0} />
        <KpiCard label="Overstock Risk" value={summary.riskCounts.OVERSTOCK ?? 0} />
        <KpiCard
          label="Watch Items"
          value={(summary.riskCounts.WATCH_HIGH ?? 0) + (summary.riskCounts.WATCH_LOW ?? 0)}
        />
      </div>

      <div className="chart-grid">
        <Panel title="Risk Distribution" caption="SKU-location recommendations by risk flag">
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Tooltip
                formatter={(value: number, name: string) => [value, name]}
                labelFormatter={(label) => String(label)}
              />
              <Legend
                formatter={(value: string) => value.replace(/_/g, " ")}
                wrapperStyle={{ fontSize: 12 }}
              />
              <Pie
                data={riskData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={95}
                label={formatPieLabel}
                labelLine={{ stroke: "#666", strokeWidth: 1 }}
              >
                {riskData.map((entry) => (
                  <Cell key={entry.name} fill={entry.fill} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </Panel>

        <Panel
          title="Actions by Department"
          caption="Recommended inventory actions across merchandise departments"
        >
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={stackedData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e8e8e8" />
              <XAxis dataKey="department" tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              {breakdown.actions.map((action) => (
                <Bar
                  key={action}
                  dataKey={action}
                  stackId="a"
                  fill={ACTION_COLORS[action] ?? "#9a9a9a"}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </Panel>
      </div>
    </>
  );
}
