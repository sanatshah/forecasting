import { Fragment, useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { RecommendationsResponse } from "../types";
import {
  Badge,
  ErrorState,
  formatCount,
  formatCurrency,
  formatLabel,
  formatSigned,
  LoadingState,
  Panel,
} from "../components/ui";

function signClass(value: number) {
  if (value > 0) return "num-positive";
  if (value < 0) return "num-negative";
  return undefined;
}

export function ScenariosPage() {
  const [data, setData] = useState<RecommendationsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);
  const [risk, setRisk] = useState("");
  const [action, setAction] = useState("");
  const [tier, setTier] = useState("");
  const [horizon, setHorizon] = useState<number | undefined>();
  const [expanded, setExpanded] = useState<string | null>(null);

  const resetFilters = useCallback(() => {
    setTier("");
    setRisk("");
    setAction("");
    setHorizon(undefined);
  }, []);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api
      .recommendations({
        risk: risk || undefined,
        action: action || undefined,
        tier: tier || undefined,
        horizon,
      })
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [risk, action, tier, horizon]);

  useEffect(() => {
    load();
  }, [load]);

  if (loading && !data) return <LoadingState />;
  if (error) return <ErrorState error={error} />;
  if (!data) return null;

  return (
    <>
      <h1 className="page-title">Scenarios</h1>
      <p className="page-subtitle">
        Segment risk flags, retention actions and the planned price-increase scenario over{" "}
        {data.meta.horizon} days — {data.meta.returned} of {data.meta.totalMatching} segments
      </p>

      <div className="filter-bar">
        <div className="filter-group">
          <label htmlFor="tier-filter">Tier</label>
          <select id="tier-filter" value={tier} onChange={(e) => setTier(e.target.value)}>
            <option value="">All tiers</option>
            {data.filters.tiers.map((t) => (
              <option key={t} value={t}>
                {t}
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
                {formatLabel(r)}
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
                {formatLabel(a)}
              </option>
            ))}
          </select>
        </div>
        <div className="filter-group">
          <label htmlFor="horizon-filter">Horizon</label>
          <select
            id="horizon-filter"
            value={horizon ?? data.meta.horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
          >
            {data.filters.horizons.map((h) => (
              <option key={h} value={h}>
                {h} days
              </option>
            ))}
          </select>
        </div>
        <button type="button" className="filter-reset-btn" onClick={resetFilters}>
          Reset
        </button>
      </div>

      <Panel
        title="Segment decision table"
        caption={`Snapshot ${data.meta.snapshotDate}. Scenario columns apply the configured price change with tier churn/acquisition elasticities.`}
      >
        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Segment</th>
                <th>Risk</th>
                <th>Action</th>
                <th>Net adds</th>
                <th>Ending subs</th>
                <th>Churn rate</th>
                <th>Hrs / sub / mo</th>
                <th>Price</th>
                <th>Δ Net adds</th>
                <th>Δ Revenue</th>
                <th>Price decision</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row) => {
                const key = `${row.segmentId}-${row.forecastHorizon}`;
                const isOpen = expanded === key;
                const priceChanged = row.scenarioPrice !== row.listPrice;
                const priceActive = row.priceDecision !== "NO_CHANGE_PLANNED";
                return (
                  <Fragment key={key}>
                    <tr>
                      <td>
                        {row.tier}
                        <br />
                        <small>
                          {row.channel} · {row.partner}
                        </small>
                      </td>
                      <td>
                        <Badge kind="risk" value={row.riskFlag} />
                      </td>
                      <td>
                        <Badge kind="action" value={row.recommendedAction} />
                      </td>
                      <td className={signClass(row.netAdds)}>{formatSigned(row.netAdds)}</td>
                      <td>{formatCount(row.endingPaidSubs)}</td>
                      <td>
                        {(row.churnRate * 100).toFixed(2)}%
                        <br />
                        <small>trailing {(row.trailingChurnRate * 100).toFixed(2)}%</small>
                      </td>
                      <td>
                        {row.hoursPerPaidSubMonth.toFixed(1)}
                        <br />
                        <small className={signClass(row.usageChangePct)}>
                          {formatSigned(row.usageChangePct * 100, 1)}%
                        </small>
                      </td>
                      <td>
                        ${row.listPrice.toFixed(2)}
                        {priceChanged && (
                          <>
                            <br />
                            <small>→ ${row.scenarioPrice.toFixed(2)}</small>
                          </>
                        )}
                      </td>
                      <td className={signClass(row.netAddsDelta)}>
                        {priceActive ? formatSigned(row.netAddsDelta) : "—"}
                      </td>
                      <td className={signClass(row.revenueDelta)}>
                        {priceActive
                          ? `${row.revenueDelta >= 0 ? "+" : ""}${formatCurrency(row.revenueDelta)}`
                          : "—"}
                      </td>
                      <td>
                        <Badge kind="price" value={row.priceDecision} />
                      </td>
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
                        <td colSpan={12}>
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
