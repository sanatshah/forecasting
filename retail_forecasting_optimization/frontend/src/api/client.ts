import type {
  ActionBreakdownResponse,
  ApiError,
  HoldoutForecastsResponse,
  OkrResponse,
  RecommendationsResponse,
  SegmentForecastsResponse,
  SegmentMetricsResponse,
  SummaryResponse,
} from "../types";

export class ApiClientError extends Error {
  status: number;
  detail: ApiError | string;

  constructor(status: number, detail: ApiError | string) {
    super(typeof detail === "string" ? detail : detail.message ?? "API error");
    this.status = status;
    this.detail = detail;
  }
}

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ message: res.statusText }));
    const detail = body.detail ?? body;
    throw new ApiClientError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => fetchJson<{ status: string }>("/api/health"),
  summary: () => fetchJson<SummaryResponse>("/api/summary"),
  okr: () => fetchJson<OkrResponse>("/api/okr"),
  actionBreakdown: () => fetchJson<ActionBreakdownResponse>("/api/action-breakdown"),
  segmentForecasts: () => fetchJson<SegmentForecastsResponse>("/api/segment-forecasts"),
  holdoutForecasts: (segment: string, target: string) =>
    fetchJson<HoldoutForecastsResponse>(
      `/api/holdout-forecasts?segment=${encodeURIComponent(segment)}&target=${encodeURIComponent(target)}`,
    ),
  recommendations: (params?: {
    risk?: string;
    action?: string;
    tier?: string;
    horizon?: number;
  }) => {
    const qs = new URLSearchParams();
    if (params?.risk) qs.set("risk", params.risk);
    if (params?.action) qs.set("action", params.action);
    if (params?.tier) qs.set("tier", params.tier);
    if (params?.horizon) qs.set("horizon", String(params.horizon));
    const q = qs.toString();
    return fetchJson<RecommendationsResponse>(
      `/api/recommendations${q ? `?${q}` : ""}`,
    );
  },
  segmentMetrics: (target?: string) =>
    fetchJson<SegmentMetricsResponse>(
      `/api/metrics/segment${target ? `?target=${encodeURIComponent(target)}` : ""}`,
    ),
};
