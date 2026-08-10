import { Fragment, useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { RecommendationsResponse } from "../types";
import { Badge, ErrorState, LoadingState, Panel } from "../components/ui";

export function RecommendationsPage() {
  const [data, setData] = useState<RecommendationsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);
  const [risk, setRisk] = useState("");
  const [action, setAction] = useState("");
  const [department, setDepartment] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);

  const resetFilters = useCallback(() => {
    setDepartment("");
    setRisk("");
    setAction("");
  }, []);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api
      .recommendations({
        risk: risk || undefined,
        action: action || undefined,
        department: department || undefined,
      })
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [risk, action, department]);

  useEffect(() => {
    load();
  }, [load]);

  if (loading && !data) return <LoadingState />;
  if (error) return <ErrorState error={error} />;
  if (!data) return null;

  return (
    <>
      <h1 className="page-title">Recommendations</h1>
      <p className="page-subtitle">
        Inventory and markdown decisions — {data.meta.returned} of {data.meta.totalMatching}{" "}
        matching rows
      </p>

      <div className="filter-bar">
        <div className="filter-group">
          <label htmlFor="dept-filter">Department</label>
          <select
            id="dept-filter"
            value={department}
            onChange={(e) => setDepartment(e.target.value)}
          >
            <option value="">All departments</option>
            {data.filters.departments.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </div>
        <div className="filter-group">
          <label htmlFor="risk-filter">Risk</label>
          <select id="risk-filter" value={risk} onChange={(e) => setRisk(e.target.value)}>
            <option value="">All risks</option>
            {data.filters.riskFlags.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </div>
        <div className="filter-group">
          <label htmlFor="action-filter">Action</label>
          <select id="action-filter" value={action} onChange={(e) => setAction(e.target.value)}>
            <option value="">All actions</option>
            {data.filters.actions.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </div>
        <button type="button" className="filter-reset-btn" onClick={resetFilters}>
          Reset
        </button>
      </div>

      <Panel title="Decision Table" caption={`Snapshot ${data.meta.snapshotDate}`}>
        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>SKU</th>
                <th>Location</th>
                <th>Department</th>
                <th>Risk</th>
                <th>Action</th>
                <th>Forecast</th>
                <th>On Hand</th>
                <th>WOS</th>
                <th>Margin</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row) => {
                const key = `${row.skuId}-${row.locationId}-${row.channel}`;
                const isOpen = expanded === key;
                return (
                  <Fragment key={key}>
                    <tr>
                      <td>{row.skuId}</td>
                      <td>
                        {row.locationId}
                        <br />
                        <small>{row.channel}</small>
                      </td>
                      <td>{row.department}</td>
                      <td>
                        <Badge kind="risk" value={row.riskFlag} />
                      </td>
                      <td>
                        <Badge kind="action" value={row.recommendedAction} />
                      </td>
                      <td>{row.forecastUnits.toFixed(0)}</td>
                      <td>{row.inventoryOnHand.toFixed(0)}</td>
                      <td>{row.weeksOfSupply.toFixed(1)}</td>
                      <td>${row.expectedMargin.toFixed(0)}</td>
                      <td>
                        {row.explanation && (
                          <button
                            type="button"
                            className="expand-btn"
                            onClick={() => setExpanded(isOpen ? null : key)}
                          >
                            {isOpen ? "Hide" : "Why?"}
                          </button>
                        )}
                      </td>
                    </tr>
                    {isOpen && row.explanation && (
                      <tr className="explanation-row">
                        <td colSpan={10}>
                          <strong>{row.reasonCode}</strong> — {row.explanation}
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>
    </>
  );
}
