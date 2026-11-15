import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import {
  MODE_LABEL,
  STATE_LABEL,
  capabilityLabels,
  formatDuration,
  type DeviceInfo,
  type DeviceStatus,
  type OutputStatus
} from "../device/types";

const POLL_INTERVAL_MS = 1000;

interface DevicePanelProps {
  /** Called once the board has reported what its bitstream offers. */
  onInfo?: (info: DeviceInfo) => void;
}

/**
 * What the board is doing right now.
 *
 * Experiments are run from Python (the gRPC device server); this panel only watches: it shows
 * the state of both outputs whoever set them up, and offers one control, "Stop all outputs".
 */
export function DevicePanel({ onInfo }: DevicePanelProps) {
  const [info, setInfo] = useState<DeviceInfo | null>(null);
  const [status, setStatus] = useState<DeviceStatus | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [stopping, setStopping] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api
      .getDeviceInfo()
      .then((result) => {
        if (!cancelled) {
          setInfo(result);
          onInfo?.(result);
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setProblem(error instanceof Error ? error.message : String(error));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [onInfo]);

  useEffect(() => {
    let cancelled = false;
    const poll = () => {
      api
        .getDeviceStatus()
        .then((result) => {
          if (!cancelled) {
            setStatus(result);
            setProblem(null);
          }
        })
        .catch((error: unknown) => {
          if (!cancelled) {
            setProblem(error instanceof Error ? error.message : String(error));
          }
        });
    };
    poll();
    const timer = window.setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const stopAll = useCallback(() => {
    setStopping(true);
    api
      .stopAllOutputs()
      .then(setStatus)
      .catch((error: unknown) => setProblem(error instanceof Error ? error.message : String(error)))
      .finally(() => setStopping(false));
  }, []);

  const sampleRate = info?.dac_sample_rate_hz ?? 9.8304e9;
  const anyActive = Boolean(status?.outputs.some((output) => output.state !== "idle" && output.state !== "done"));

  return (
    <section className="panel section-panel">
      <div className="section-heading">
        <p className="prompt-line">Device</p>
        <h2>Outputs</h2>
      </div>

      {problem && <p className="live-error">Board not reachable: {problem}</p>}

      <div className="dac-output-grid">
        {(status?.outputs ?? []).map((output) => (
          <OutputCard key={output.channel} output={output} sampleRate={sampleRate} />
        ))}
      </div>

      <div className="device-actions">
        <button type="button" className="dac-toggle enabled" disabled={stopping || !anyActive} onClick={stopAll}>
          Stop all outputs
        </button>
        <span className="meta-note">
          Experiments are set up and started from Python; this page shows their state, updated every second.
        </span>
      </div>

      {info && (
        <div className="readout-list device-system">
          <div className="readout-row">
            <span>bitstream offers</span>
            <strong>{capabilityLabels(info.capabilities).join(", ")}</strong>
          </div>
          <div className="readout-row">
            <span>reference clock</span>
            <strong>{info.clock_source || "unknown"}</strong>
          </div>
          <div className="readout-row">
            <span>bitstream file</span>
            <strong>{info.bitfile || "unknown"}</strong>
          </div>
        </div>
      )}
    </section>
  );
}

function OutputCard({ output, sampleRate }: { output: OutputStatus; sampleRate: number }) {
  const active = output.state === "running" || output.state === "holding";
  return (
    <article className={active ? "module-card module-card-emphasis" : "module-card"}>
      <div className="readout-row">
        <span>{output.channel.toUpperCase()}</span>
        <strong>{STATE_LABEL[output.state]}</strong>
      </div>
      <div className="readout-row">
        <span>playback</span>
        <strong>{MODE_LABEL[output.mode]}</strong>
      </div>
      {output.mode !== "none" && (
        <div className="readout-row">
          <span>loaded</span>
          <strong>
            {output.loaded_samples.toLocaleString()} samples ({formatDuration(output.loaded_samples, sampleRate)} per pass)
          </strong>
        </div>
      )}
      {output.mode === "sequence" && (
        <div className="readout-row">
          <span>current block</span>
          <strong>{output.block}</strong>
        </div>
      )}
      {output.buffer_low_water !== null && (
        <div className="readout-row">
          <span>DDR4 buffer low mark</span>
          <strong>{output.buffer_low_water} of 4096</strong>
        </div>
      )}
      {output.warnings.map((warning) => (
        <p key={warning} className="live-error">
          {warning}
        </p>
      ))}
    </article>
  );
}
