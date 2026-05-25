import { NavLink, Outlet } from "react-router-dom";
import { FormEvent, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { useAppState } from "../context/AppStateContext";

const navItems = [
  { to: "/signal-generator", label: "Signal generator", index: "01" },
];

export function Layout() {
  const {
    error,
    status,
    constants,
    loadBackendSummary,
  } = useAppState();
  const [userName, setUserName] = useState("");
  const [sessionName, setSessionName] = useState<string | null>(null);

  const today = useMemo(
    () =>
      new Date().toLocaleDateString(undefined, {
        weekday: "long",
        month: "short",
        day: "numeric",
      }),
    []
  );

  const dacRate = constants?.DAC_SR ?? status?.dac_sr ?? null;
  const bufferLength = constants?.BUF_LEN ?? status?.buf_len ?? null;
  const amplitudeLimit = constants?.DAC_AMP ?? null;
  const hardwareLabel = status
    ? status.hardware_initialized
      ? "hardware ready"
      : "hardware not initialized"
    : null;
  const sessionFacts = [
    dacRate ? `${(dacRate / 1e9).toFixed(2)} GSPS DAC` : null,
    status || constants ? "hardware link" : null,
    hardwareLabel,
    bufferLength ? `buffer ${bufferLength.toLocaleString()}` : null,
    amplitudeLimit ? `amp ${amplitudeLimit.toLocaleString()}` : null,
  ].filter((value): value is string => Boolean(value));

  useEffect(() => {
    if (!sessionName) {
      return;
    }

    void loadBackendSummary();
  }, [sessionName, loadBackendSummary]);

  function handleLogin(event: FormEvent) {
    event.preventDefault();
    const normalizedUser = userName.trim();

    if (!normalizedUser) {
      return;
    }

    setSessionName(normalizedUser);
  }

  if (!sessionName) {
    return (
      <main className="auth-shell">
        <section className="auth-card">
          <p className="subtitle auth-copy">RFSoC hardware control</p>
          <div className="auth-meta-grid">
            <div className="meta-chip">
              <code>{api.backendUrl}</code>
            </div>
            <div className="meta-chip">
              <strong>overlay controller</strong>
            </div>
          </div>
          <form className="auth-form" onSubmit={handleLogin}>
            <label className="field">
              <span className="field-label">User name</span>
              <input
                value={userName}
                placeholder="e.g. Dr. Ada Maxwell"
                onChange={(e) => setUserName(e.target.value)}
              />
            </label>
            <button type="submit" disabled={!userName.trim()}>
              Open workspace
            </button>
          </form>
        </section>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <section className="panel session-panel">
          <div className="session-head">
            <h2 className="session-title">{sessionName}</h2>
            <p className="meta-note">{today}</p>
          </div>
          <div className="meta-stack">
            <div className="backend-endpoint">{api.backendUrl}</div>
            {sessionFacts.length > 0 && (
              <div className="session-facts">
                {sessionFacts.map((fact) => (
                  <div key={fact} className="session-fact">
                    {fact}
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>

        <nav className="panel nav-panel sidebar-nav">
          <p className="section-kicker">Navigation</p>
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                isActive ? "nav-link active" : "nav-link"
              }
            >
              <span className="nav-index">{item.index}</span>
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>
      </aside>

      <section className="workspace">
        {error && <div className="error">{error}</div>}
        <Outlet />
      </section>
    </main>
  );
}
