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
import type { DepartmentMetricsResponse } from "../types";
import { ErrorState, LoadingState, Panel } from "../components/ui";

export function AccuracyPage() {
  const [data, setData] = useState<DepartmentMetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    api
      .departmentMetrics()
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingState />;
  if (error) return <ErrorState error={error} />;
  if (!data) return null;

  const chartData = data.departments.map((d) => ({
    department: d.department,
    wapePct: +(d.wape * 100).toFixed(1),
    mapePct: +(d.mape * 100).toFixed(1),
    n: d.n,
  }));

  const modelLabel = data.meta.model.replace(/_/g, " ");

  return (
    <>
      <h1 className="page-title">Forecast Accuracy</h1>
      <p className="page-subtitle">
        WAPE by department — {modelLabel} (snapshot {data.meta.snapshotDate})
      </p>

      <Panel title="WAPE by Department" caption="Weighted Absolute Percentage Error (lower is better)">
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={chartData} margin={{ bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e8e8e8" />
            <XAxis dataKey="department" tick={{ fontSize: 11 }} />
            <YAxis unit="%" />
            <Tooltip formatter={(v: number) => [`${v}%`, "WAPE"]} />
            <Bar dataKey="wapePct" fill="#e21a2c" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Panel>

      <div style={{ marginTop: "1.5rem" }}>
        <Panel title="Department Detail" caption={`Model: ${modelLabel}`}>
          <div className="data-table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Department</th>
                  <th>WAPE</th>
                  <th>MAPE</th>
                  <th>MAE</th>
                  <th>RMSE</th>
                  <th>Bias</th>
                  <th>N</th>
                </tr>
              </thead>
              <tbody>
                {data.departments.map((d) => (
                  <tr key={d.department}>
                    <td>{d.department}</td>
                    <td>{(d.wape * 100).toFixed(1)}%</td>
                    <td>{(d.mape * 100).toFixed(1)}%</td>
                    <td>{d.mae.toFixed(2)}</td>
                    <td>{d.rmse.toFixed(2)}</td>
                    <td>{d.bias.toFixed(2)}</td>
                    <td>{d.n.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>
    </>
  );
}
