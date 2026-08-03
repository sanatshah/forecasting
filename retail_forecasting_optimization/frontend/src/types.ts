export interface ApiError {
  message?: string;
  hint?: string;
  error?: string;
}

export interface SummaryResponse {
  meta: {
    snapshotDate: string;
    recommendationCount: number;
  };
  bestModel: string;
  bestWape: number | null;
  riskCounts: Record<string, number>;
  actionCounts: Record<string, number>;
}

export interface ActionBreakdownResponse {
  meta: { snapshotDate: string; grandTotal: number };
  departments: string[];
  actions: string[];
  breakdown: Record<string, Record<string, number>>;
  totals: Record<string, number>;
}

export interface SkuForecastSeries {
  location: string;
  channel: string;
  data: number[];
  total: number;
}

export interface SkuForecast {
  skuId: string;
  department: string;
  class: string;
  subclass: string;
  lifecycle: string;
  aggregate: number[];
  totalUnits: number;
  avgDaily: number;
  series: SkuForecastSeries[];
}

export interface SkuForecastsResponse {
  meta: {
    snapshotDate: string;
    horizonDays: number;
    forecastStart: string | null;
    forecastEnd: string | null;
  };
  dates: string[];
  skus: SkuForecast[];
}

export interface RecommendationRow {
  date: string;
  skuId: string;
  locationId: string;
  channel: string;
  department: string;
  forecastHorizon: number;
  forecastUnits: number;
  inventoryOnHand: number;
  weeksOfSupply: number;
  riskFlag: string;
  recommendedAction: string;
  recommendedMarkdownPct: number;
  expectedSales: number;
  expectedMargin: number;
  objectiveScore: number;
  reasonCode: string;
  explanation: string;
}

export interface RecommendationsResponse {
  meta: {
    snapshotDate: string;
    totalMatching: number;
    returned: number;
  };
  filters: {
    departments: string[];
    riskFlags: string[];
    actions: string[];
  };
  rows: RecommendationRow[];
}

export interface DepartmentMetric {
  department: string;
  wape: number;
  mape: number;
  mae: number;
  rmse: number;
  bias: number;
  n: number;
}

export interface DepartmentMetricsResponse {
  meta: { snapshotDate: string; model: string };
  departments: DepartmentMetric[];
}

export interface HoldoutMetrics {
  mae: number | null;
  wape: number | null;
  n: number;
}

export interface HoldoutForecastsResponse {
  meta: {
    recipe: string;
    available: boolean;
    source: string | null;
    skuId: string | null;
    snapshotDate: string | null;
    holdoutStart: string | null;
    holdoutEnd: string | null;
    message: string | null;
  };
  dates: string[];
  actuals: number[];
  predictions: number[];
  metrics: HoldoutMetrics | null;
  skuIds: string[];
}

export type ForecastViewMode = "forward" | "holdout" | "combined";
