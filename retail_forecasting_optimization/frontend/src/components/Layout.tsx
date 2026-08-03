import { NavLink, Outlet } from "react-router-dom";

interface LayoutProps {
  snapshotDate?: string;
}

function StarIcon() {
  return (
    <svg className="brand-star" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 2l2.9 6.9 7.5.6-5.7 4.9 1.7 7.3L12 18.5 5.6 21.7l1.7-7.3L1.6 9.5l7.5-.6L12 2z" />
    </svg>
  );
}

export function Layout({ snapshotDate }: LayoutProps) {
  return (
    <>
      <header className="app-header">
        <div className="header-inner">
          <div className="brand">
            <StarIcon />
            <div>
              <div className="brand-title">Macy&apos;s</div>
              <div className="brand-subtitle">Demand Planning</div>
            </div>
          </div>
          {snapshotDate && (
            <div className="snapshot-badge">Snapshot: {snapshotDate}</div>
          )}
        </div>
      </header>
      <nav className="app-nav">
        <div className="nav-inner">
          <NavLink to="/" end className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}>
            Overview
          </NavLink>
          <NavLink
            to="/recommendations"
            className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
          >
            Recommendations
          </NavLink>
          <NavLink
            to="/forecasts"
            className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
          >
            Forecasts
          </NavLink>
          <NavLink
            to="/accuracy"
            className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
          >
            Accuracy
          </NavLink>
        </div>
      </nav>
      <main className="app-main">
        <Outlet />
      </main>
    </>
  );
}
