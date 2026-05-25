import { useAppState } from "../context/AppStateContext";

export function StatusPanel() {
  const { status, constants } = useAppState();
  const snapshot = {
    ...(status ? { status } : {}),
    ...(constants ? { constants } : {}),
  };

  const statusItems = status
    ? [
        ["Hardware init", status.hardware_initialized ? "ready" : "not initialized"],
        ["Buffer length", status.buf_len.toLocaleString()],
        ["DAC sample rate", `${(status.dac_sr / 1e9).toFixed(2)} GSPS`],
      ]
    : [];

  const constantItems = constants
    ? [
        ["DAC_AMP", constants.DAC_AMP.toLocaleString()],
        ["BUF_LEN", constants.BUF_LEN.toLocaleString()],
      ]
    : [];

  return (
    <section className="panel status-shell">
      <div className="section-heading">
        <p className="prompt-line">Telemetry</p>
        <h2>Telemetry</h2>
      </div>

      {!status && !constants && (
        <div className="empty-state">
          No telemetry available.
        </div>
      )}

      <div className="telemetry-grid">
        {statusItems.length > 0 && (
          <article className="telemetry-card">
            <p className="command-kicker">status</p>
            <div className="readout-list">
              {statusItems.map(([label, value]) => (
                <div key={label} className="readout-row">
                  <span>{label}</span>
                  <strong>{value}</strong>
                </div>
              ))}
            </div>
          </article>
        )}

        {constantItems.length > 0 && (
          <article className="telemetry-card">
            <p className="command-kicker">constants</p>
            <div className="readout-list">
              {constantItems.map(([label, value]) => (
                <div key={label} className="readout-row">
                  <span>{label}</span>
                  <strong>{value}</strong>
                </div>
              ))}
            </div>
          </article>
        )}
      </div>

      {Object.keys(snapshot).length > 0 && (
        <div className="raw-console monospace">
          <div className="console-bar">
            <span>telemetry json</span>
            <span>snapshot</span>
          </div>
          <pre>{JSON.stringify(snapshot, null, 2)}</pre>
        </div>
      )}
    </section>
  );
}
