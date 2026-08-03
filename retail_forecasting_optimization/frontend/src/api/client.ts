import type {
  ActionBreakdownResponse,
  ApiError,
  DepartmentMetricsResponse,
  RecommendationsResponse,
  SkuForecastsResponse,
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
  actionBreakdown: () => fetchJson<ActionBreakdownResponse>("/api/action-breakdown"),
  skuForecasts: (topSkus = 8) =>
    fetchJson<SkuForecastsResponse>(`/api/sku-forecasts?top_skus=${topSkus}`),
  recommendations: (params?: {
    risk?: string;
    action?: string;
    department?: string;
  }) => {
    const qs = new URLSearchParams();
    if (params?.risk) qs.set("risk", params.risk);
    if (params?.action) qs.set("action", params.action);
    if (params?.department) qs.set("department", params.department);
    const q = qs.toString();
    return fetchJson<RecommendationsResponse>(
      `/api/recommendations${q ? `?${q}` : ""}`,
    );
  },
  departmentMetrics: () =>
    fetchJson<DepartmentMetricsResponse>("/api/metrics/department"),
};
