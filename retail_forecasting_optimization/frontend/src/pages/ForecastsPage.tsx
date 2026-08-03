import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import type { SkuForecastsResponse } from "../types";
import { ErrorState, KpiCard, LoadingState, Panel } from "../components/ui";

const SERIES_COLORS = ["#e21a2c", "#1a1a1a", "#1565c0", "#7b1fa2", "#2e7d32"];

export function ForecastsPage() {
  const [data, setData] = useState<SkuForecastsResponse | null>(null);
  const [selectedSku, setSelectedSku] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    api
      .skuForecasts(8)
      .then((res) => {
        setData(res);
        if (res.skus.length > 0) {
          setSelectedSku(res.skus[0].skuId);
        }
      })
      .catch(setError)
      .finally(() => setLoading(false));
  }, []);

  const sku = useMemo(
    () => data?.skus.find((s) => s.skuId === selectedSku),
    [data, selectedSku],
  );

  const chartData = useMemo(() => {
    if (!data || !sku) return [];
    return data.dates.map((date, i) => {
      const point: Record<string, string | number> = {
        date: date.slice(5),
        Total: sku.aggregate[i] ?? 0,
      };
      for (const s of sku.series) {
        point[`${s.location} / ${s.channel}`] = s.data[i] ?? 0;
      }
      return point;
    });
  }, [data, sku]);

  if (loading) return <LoadingState />;
  if (error) return <ErrorState error={error} />;
  if (!data || !sku) return null;

  return (
    <>
      <h1 className="page-title">SKU Forecasts</h1>
      <p className="page-subtitle">
        Forward demand curves — {data.meta.horizonDays}-day horizon (
        {data.meta.forecastStart} to {data.meta.forecastEnd})
      </p>

      <div className="filter-bar">
        <div className="filter-group">
          <label htmlFor="sku-select">SKU</label>
          <select
            id="sku-select"
            value={selectedSku}
            onChange={(e) => setSelectedSku(e.target.value)}
          >
            {data.skus.map((s) => (
              <option key={s.skuId} value={s.skuId}>
                {s.skuId} — {s.department} ({s.totalUnits.toLocaleString()} units)
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="kpi-grid">
        <KpiCard label="Total Units" value={sku.totalUnits.toLocaleString()} />
        <KpiCard label="Avg Daily" value={sku.avgDaily} unit="units" />
        <KpiCard label="Department" value={sku.department} />
        <KpiCard label="Lifecycle" value={sku.lifecycle || "—"} />
      </div>

      <Panel
        title={`Forecast: ${sku.skuId}`}
        caption={`${sku.class} / ${sku.subclass} — top location/channel series`}
      >
        <ResponsiveContainer width="100%" height={360}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e8e8e8" />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="Total"
              stroke="#e21a2c"
              strokeWidth={2.5}
              dot={false}
            />
            {sku.series.map((s, i) => (
              <Line
                key={`${s.location}-${s.channel}`}
                type="monotone"
                dataKey={`${s.location} / ${s.channel}`}
                stroke={SERIES_COLORS[(i + 1) % SERIES_COLORS.length]}
                strokeWidth={1.5}
                dot={false}
                strokeDasharray="4 2"
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Panel>
    </>
  );
}
