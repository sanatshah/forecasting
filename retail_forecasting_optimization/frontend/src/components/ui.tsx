import type { ReactNode } from "react";
import { ApiClientError } from "../api/client";

interface EmptyStateProps {
  title: string;
  message: string;
  command?: string;
}

export function EmptyState({ title, message, command }: EmptyStateProps) {
  return (
    <div className="state-box">
      <h2>{title}</h2>
      <p>{message}</p>
      {command && <code>{command}</code>}
    </div>
  );
}

export function LoadingState() {
  return <div className="loading">Loading dashboard data…</div>;
}

interface ErrorStateProps {
  error: unknown;
}

export function ErrorState({ error }: ErrorStateProps) {
  if (error instanceof ApiClientError && error.status === 404) {
    const detail =
      typeof error.detail === "object" ? error.detail : { message: String(error.detail) };
    return (
      <EmptyState
        title="No pipeline data"
        message={detail.message ?? "Run the forecasting pipeline to generate dashboard data."}
        command={
          detail.hint ??
          "cd retail_forecasting_optimization && ./.venv/bin/python main.py --quick"
        }
      />
    );
  }

  const message = error instanceof Error ? error.message : "An unexpected error occurred.";
  return (
    <EmptyState
      title="Something went wrong"
      message={message}
      command="Ensure the API is running: ./.venv/bin/uvicorn src.dashboard_api:app --reload --port 8000"
    />
  );
}

interface KpiCardProps {
  label: string;
  value: string | number;
  unit?: string;
  delta?: string;
  deltaTone?: "up" | "down" | "flat";
}

export function KpiCard({ label, value, unit, delta, deltaTone = "flat" }: KpiCardProps) {
  return (
    <div className="kpi-card">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">
        {value}
        {unit && <small> {unit}</small>}
      </div>
      {delta && <div className={`kpi-delta kpi-delta-${deltaTone}`}>{delta}</div>}
    </div>
  );
}

interface PanelProps {
  title: string;
  caption?: string;
  children: ReactNode;
}

export function Panel({ title, caption, children }: PanelProps) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h3>{title}</h3>
        {caption && <div className="panel-caption">{caption}</div>}
      </div>
      <div className="panel-body">{children}</div>
    </section>
  );
}

interface BadgeProps {
  kind: "risk" | "action" | "price";
  value: string;
}

export function Badge({ kind, value }: BadgeProps) {
  const normalized = value.replace(/ /g, "_");
  return <span className={`badge badge-${kind}-${normalized}`}>{value.replace(/_/g, " ")}</span>;
}

export function formatCount(value: number): string {
  return Math.round(value).toLocaleString();
}

export function formatSigned(value: number, digits = 0): string {
  const rounded = Number(value.toFixed(digits));
  const text = rounded.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
  return rounded > 0 ? `+${text}` : text;
}

export function formatCurrency(value: number): string {
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(1)}K`;
  return `${sign}$${abs.toFixed(0)}`;
}

export function formatLabel(value: string): string {
  return value.replace(/_/g, " ");
}

export const TIER_COLORS: Record<string, string> = {
  Premium: "#0089D0",
  "Premium Plus": "#6460AA",
  "Ad Tier": "#FCB711",
};

export const CHART_GRID = "rgba(255, 255, 255, 0.08)";
export const CHART_AXIS = { fill: "#a8a8b3", fontSize: 11 };
export const TOOLTIP_STYLE = {
  background: "#1c1c24",
  border: "1px solid #33333f",
  borderRadius: 6,
  color: "#f4f4f6",
};
