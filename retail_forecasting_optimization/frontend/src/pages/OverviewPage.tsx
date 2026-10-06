import { useEffect, useMemo, useState } from "react";
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
import type { ActionBreakdownResponse, OkrResponse, SummaryResponse } from "../types";
import {
  CHART_AXIS,
  CHART_GRID,
  ErrorState,
  formatCount,
  formatCurrency,
  formatLabel,
  formatSigned,
  KpiCard,
  LoadingState,
  Panel,
  TOOLTIP_STYLE,
} from "../components/ui";

const ACTION_COLORS: Record<string, string> = {
  RETENTION_OFFER: "#CC004C",
  ENGAGEMENT_PUSH: "#FCB711",
  ANNUAL_PLAN_UPSELL: "#6460AA",
  PROCEED_PRICE_CHANGE: "#0DB14B",
  HOLD_PRICE: "#0089D0",
  MONITOR: "#5d5d6c",
};

const RISK_COLORS: Record<string, string> = {
  NEGATIVE_NET_ADDS: "#CC004C",
  CHURN_SPIKE: "#F37021",
  TENTPOLE_CLIFF: "#6460AA",
  USAGE_DECLINE: "#FCB711",
  PRICE_SENSITIVE: "#0089D0",
  OK: "#0DB14B",
};

const TARGET_LABELS: Record<string, string> = {
  gross_adds: "Gross adds",
  churned_subs: "Churn",
  hours_watched: "Hours watched",
};

function formatModel(name: string) {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatPieLabel({ name, value }: { name: string; value: number }) {
  return `${formatLabel(name)}: ${value}`;
}

function formatTooltipCount(value: number, name: string) {
  return [formatCount(value), formatLabel(name)];
}

function formatHorizonTick(value: number) {
  return `${value}d`;
}

function deltaTone(value: number): "up" | "down" | "flat" {
  if (value > 0) return "up";
  if (value < 0) return "down";
  return "flat";
}

export function OverviewPage() {
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [breakdown, setBreakdown] = useState<ActionBreakdownResponse | null>(null);
  const [okr, setOkr] = useState<OkrResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    Promise.all([api.summary(), api.actionBreakdown(), api.okr()])
      .then(([s, b, o]) => {
        setSummary(s);
        setBreakdown(b);
        setOkr(o);
      })
      .catch(setError)
      .finally(() => setLoading(false));
  }, []);

  const scenarioChartData = useMemo(() => {
    if (!okr) return [];
    const byHorizon = new Map<number, Record<string, number>>();
    for (const row of okr.rows) {
      const point = byHorizon.get(row.horizon) ?? { horizon: row.horizon };
      point[row.scenario] = Math.round(row.highValueNetAdds);
      byHorizon.set(row.horizon, point);
    }
    return [...byHorizon.values()].sort((a, b) => a.horizon - b.horizon);
  }, [okr]);

  if (loading) return <LoadingState />;
  if (error) return <ErrorState error={error} />;
  if (!summary || !breakdown || !okr) return null;

  const base = summary.okr;
  const scen = summary.okrScenario;
  const netAddsDelta = base && scen ? scen.highValueNetAdds - base.highValueNetAdds : 0;
  const revenueDelta = base && scen ? scen.revenue - base.revenue : 0;

  const riskData = Object.entries(summary.riskCounts).map(([name, value]) => ({
    name,
    value,
    fill: RISK_COLORS[name] ?? "#5d5d6c",
  }));

  const stackedData = breakdown.tiers.map((tier) => {
    const row: Record<string, string | number> = { tier };
    for (const action of breakdown.actions) {
      row[action] = breakdown.breakdown[tier]?.[action] ?? 0;
    }
    return row;
  });

  const priceChangeText = okr.meta.priceChanges
    .map((p) => `${p.tier} → $${p.new_price} on ${p.effective_date}`)
    .join("; ");

  return (
    <>
      <h1 className="page-title">Growth OKRs</h1>
      <p className="page-subtitle">
        High-value subscriber growth ({okr.meta.highValueTiers.join(" + ")} via{" "}
        {okr.meta.highValueChannels.join(" / ")}) and usage per paid sub — next{" "}
        {summary.meta.horizon} days from snapshot {summary.meta.snapshotDate}
      </p>

      <div className="kpi-grid">
        <KpiCard
          label="High-value net adds"
          value={base ? formatSigned(base.highValueNetAdds) : "—"}
          delta={scen ? `${formatSigned(netAddsDelta)} with price change` : undefined}
          deltaTone={deltaTone(netAddsDelta)}
        />
        <KpiCard
          label="High-value paid subs"
          value={base ? formatCount(base.highValuePaidSubsEnd) : "—"}
          unit="end"
        />
        <KpiCard
          label="Hours / paid sub"
          value={base ? base.highValueHoursPerPaidSubMonth.toFixed(1) : "—"}
          unit="per month"
        />
        <KpiCard
          label="Total net adds"
          value={base ? formatSigned(base.totalNetAdds) : "—"}
          unit="all tiers"
        />
        <KpiCard
          label="Revenue"
          value={base ? formatCurrency(base.revenue) : "—"}
          delta={scen ? `${revenueDelta >= 0 ? "+" : ""}${formatCurrency(revenueDelta)} with price change` : undefined}
          deltaTone={deltaTone(revenueDelta)}
        />
        {Object.entries(summary.bestModels).map(([target, m]) => (
          <KpiCard
            key={target}
            label={`${TARGET_LABELS[target] ?? target} model`}
            value={formatModel(m.model)}
            unit={`${(m.wape * 100).toFixed(1)}% WAPE`}
          />
        ))}
      </div>

      <div className="chart-grid">
        <Panel
          title="High-value net adds: baseline vs price change"
          caption={priceChangeText || "No planned price change in config"}
        >
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={scenarioChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
              <XAxis dataKey="horizon" tick={CHART_AXIS} tickFormatter={formatHorizonTick} />
              <YAxis tick={CHART_AXIS} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={formatTooltipCount} />
              <Legend formatter={formatLabel} wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="baseline" fill="#0089D0" radius={[4, 4, 0, 0]} />
              <Bar dataKey="price_change" fill="#F37021" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Panel>

        <Panel title="Segment risk" caption="Tier x channel segments by risk flag">
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={formatTooltipCount} />
              <Legend formatter={formatLabel} wrapperStyle={{ fontSize: 12 }} />
              <Pie
                data={riskData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={90}
                label={formatPieLabel}
                labelLine={{ stroke: "#5d5d6c", strokeWidth: 1 }}
              >
                {riskData.map((entry) => (
                  <Cell key={entry.name} fill={entry.fill} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </Panel>

        <Panel title="Actions by tier" caption="Recommended retention, engagement and pricing actions">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={stackedData}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
              <XAxis dataKey="tier" tick={CHART_AXIS} />
              <YAxis tick={CHART_AXIS} allowDecimals={false} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={formatTooltipCount} />
              <Legend formatter={formatLabel} wrapperStyle={{ fontSize: 12 }} />
              {breakdown.actions.map((action) => (
                <Bar
                  key={action}
                  dataKey={action}
                  stackId="a"
                  fill={ACTION_COLORS[action] ?? "#5d5d6c"}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </Panel>
      </div>
    </>
  );
}
