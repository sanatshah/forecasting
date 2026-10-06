import { NavLink, Outlet } from "react-router-dom";

interface LayoutProps {
  snapshotDate?: string;
}

const FEATHER_COLORS = ["#FCB711", "#F37021", "#CC004C", "#6460AA", "#0089D0", "#0DB14B"];

function FeatherIcon() {
  return (
    <svg className="brand-icon" viewBox="0 0 24 24" aria-hidden="true">
      {FEATHER_COLORS.map((color, i) => (
        <circle key={color} cx={4 + i * 3.2} cy={i % 2 === 0 ? 9 : 15} r={2.4} fill={color} />
      ))}
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
            <FeatherIcon />
            <div>
              <div className="brand-title">Peacock</div>
              <div className="brand-subtitle">Subscriber Planning</div>
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
          <NavLink to="/segments" className={navClassName}>
            Segments
          </NavLink>
          <NavLink to="/scenarios" className={navClassName}>
            Scenarios
          </NavLink>
          <NavLink to="/accuracy" className={navClassName}>
            Accuracy
          </NavLink>
        </nav>
        <div className="feather-bar" aria-hidden="true" />
      </aside>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}
