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

const navClassName = ({ isActive }: { isActive: boolean }) =>
  `nav-link${isActive ? " active" : ""}`;

export function Layout({ snapshotDate }: LayoutProps) {
  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <header className="sidebar-brand">
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
