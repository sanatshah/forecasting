import { NavLink, Outlet } from "react-router-dom";

interface LayoutProps {
  snapshotDate?: string;
}

function StoreIcon() {
  return (
    <svg className="brand-icon" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 10v10h16V10M3 10l2-6h14l2 6M8 20v-6h4v6M3 10a3 3 0 0 0 6 0 3 3 0 0 0 6 0 3 3 0 0 0 6 0" />
    </svg>
  );
}

const navClassName = ({ isActive }: { isActive: boolean }) =>
  `nav-link${isActive ? " active" : ""}`;

export function Layout({ snapshotDate }: LayoutProps) {
  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <header className="sidebar-brand">
          <div className="brand">
            <StoreIcon />
            <div>
              <div className="brand-title">RetailStore</div>
              <div className="brand-subtitle">Demand Planning</div>
            </div>
          </div>
          {snapshotDate && (
            <div className="snapshot-badge">Snapshot: {snapshotDate}</div>
          )}
        </header>
        <nav className="sidebar-nav" aria-label="Main navigation">
          <NavLink to="/" end className={navClassName}>
            Overview
          </NavLink>
          <NavLink to="/recommendations" className={navClassName}>
            Recommendations
          </NavLink>
          <NavLink to="/forecasts" className={navClassName}>
            Forecasts
          </NavLink>
          <NavLink to="/accuracy" className={navClassName}>
            Accuracy
          </NavLink>
        </nav>
      </aside>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}
