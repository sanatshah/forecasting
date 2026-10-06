export interface ApiError {
  message?: string;
  hint?: string;
  error?: string;
}

export interface OkrRecord {
  horizon: number;
  scenario: "baseline" | "price_change" | string;
  forecastStart: string | null;
  forecastEnd: string | null;
  highValueNetAdds: number;
  highValuePaidSubsEnd: number;
  highValueGrossAdds: number;
  totalNetAdds: number;
  totalPaidSubsEnd: number;
  hoursPerPaidSubMonth: number;
  highValueHoursPerPaidSubMonth: number;
  revenue: number;
}

export interface SummaryResponse {
  meta: {
    snapshotDate: string;
    horizon: number;
    segmentCount: number;
  };
  bestModel: string;
  bestWape: number | null;
  bestModels: Record<string, { model: string; wape: number }>;
  okr: OkrRecord | null;
  okrScenario: OkrRecord | null;
  riskCounts: Record<string, number>;
  actionCounts: Record<string, number>;
}

export interface PriceChange {
  tier: string;
  new_price: number;
  effective_date: string;
}

export interface OkrResponse {
  meta: {
    highValueTiers: string[];
    highValueChannels: string[];
    priceChanges: PriceChange[];
  };
  rows: OkrRecord[];
}

export interface ActionBreakdownResponse {
  meta: { snapshotDate: string; horizon: number; grandTotal: number };
  tiers: string[];
  actions: string[];
  breakdown: Record<string, Record<string, number>>;
  totals: Record<string, number>;
}

export interface SegmentForecast {
  segmentId: string;
  tier: string;
  channel: string;
  partner: string;
  openingPaidSubs: number;
  grossAdds: number[];
  churnedSubs: number[];
  netAdds: number[];
  paidSubs: number[];
  hoursPerPaidSub: number[];
  totalNetAdds: number;
  endingPaidSubs: number;
}

export interface SegmentForecastsResponse {
  meta: {
    snapshotDate: string;
    horizonDays: number;
    forecastStart: string | null;
    forecastEnd: string | null;
  };
  dates: string[];
  segments: SegmentForecast[];
}

export type ForecastTarget = "gross_adds" | "churned_subs" | "hours_watched";

export interface HoldoutForecastsResponse {
  meta: {
    segmentId: string;
    target: string;
    snapshotDate: string | null;
    holdoutStart: string | null;
    holdoutEnd: string | null;
    horizonDays: number;
    tier: string;
    channel: string;
    source?: string;
  };
  dates: string[];
  actuals: number[];
  predictions: number[];
  metrics: {
    mae: number;
    wape: number | null;
  };
}

export type ForecastViewMode = "forward" | "holdout";

export interface ScenarioRow {
  date: string;
  segmentId: string;
  tier: string;
  channel: string;
  partner: string;
  forecastHorizon: number;
  openingPaidSubs: number;
  grossAdds: number;
  churnedSubs: number;
  netAdds: number;
  endingPaidSubs: number;
  hoursPerPaidSubMonth: number;
  churnRate: number;
  trailingChurnRate: number;
  usageChangePct: number;
  listPrice: number;
  scenarioPrice: number;
  scenarioNetAdds: number;
  netAddsDelta: number;
  baselineRevenue: number;
  scenarioRevenue: number;
  revenueDelta: number;
  riskFlag: string;
  recommendedAction: string;
  priceDecision: string;
  objectiveScore: number;
  reasonCode: string;
  explanation: string;
}

export interface RecommendationsResponse {
  meta: {
    snapshotDate: string;
    horizon: number;
    totalMatching: number;
    returned: number;
  };
  filters: {
    tiers: string[];
    riskFlags: string[];
    actions: string[];
    horizons: number[];
  };
  rows: ScenarioRow[];
}

export interface SegmentMetric {
  group: string;
  wape: number;
  mape: number;
  mae: number;
  rmse: number;
  bias: number;
  n: number;
}

export interface SegmentMetricsResponse {
  meta: { snapshotDate: string; model: string; target: string; targets: string[] };
  tiers: SegmentMetric[];
  channels: SegmentMetric[];
}
