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
}

export function KpiCard({ label, value, unit }: KpiCardProps) {
  return (
    <div className="kpi-card">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">
        {value}
        {unit && <small> {unit}</small>}
      </div>
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
  kind: "risk" | "action";
  value: string;
}

export function Badge({ kind, value }: BadgeProps) {
  const normalized = value.replace(/ /g, "_");
  return <span className={`badge badge-${kind}-${normalized}`}>{value}</span>;
}
