import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { drawSpectrum, drawTrace } from "../live/livePlot";
import {
  formatFrequency,
  liveSocketUrl,
  parseLiveMessage,
  type AdcInput,
  type LiveFrame,
  type LiveInfo,
  type LiveSettings,
  type LiveSettingsPatch
} from "../live/protocol";

type LinkState = "idle" | "connecting" | "live" | "closed";

interface Readout {
  peakHz: number;
  peakDb: number;
  rmsDb: number;
  clipping: boolean;
  averaged: number;
  captureMs: number;
  fps: number;
}

const AVERAGING_CHOICES = [1, 2, 4, 8, 16, 32, 64];
const FPS_CHOICES = [5, 10, 15, 20, 30];
const READOUT_INTERVAL_MS = 250;

/**
 * Live spectrum and time trace of one ADC input.
 *
 * The board captures and computes only while this panel is connected. Frames are drawn on a
 * canvas as they arrive; the numeric readouts update a few times per second.
 */
export function LiveAdcPanel() {
  const [link, setLink] = useState<LinkState>("idle");
  const [message, setMessage] = useState<string | null>(null);
  const [info, setInfo] = useState<LiveInfo | null>(null);
  const [settings, setSettings] = useState<LiveSettings | null>(null);
  const [readout, setReadout] = useState<Readout | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const frameRef = useRef<LiveFrame | null>(null);
  const nyquistRef = useRef(2457.6e6);
  const drawPendingRef = useRef(false);
  const lastReadoutRef = useRef(0);
  const frameTimesRef = useRef<number[]>([]);
  const spectrumCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const traceCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const draw = useCallback(() => {
    drawPendingRef.current = false;
    if (spectrumCanvasRef.current) {
      drawSpectrum(spectrumCanvasRef.current, frameRef.current, { nyquistHz: nyquistRef.current });
    }
    if (traceCanvasRef.current) {
      drawTrace(traceCanvasRef.current, frameRef.current);
    }
  }, []);

  const scheduleDraw = useCallback(() => {
    if (!drawPendingRef.current) {
      drawPendingRef.current = true;
      window.requestAnimationFrame(draw);
    }
  }, [draw]);

  const handleFrame = useCallback(
    (frame: LiveFrame) => {
      frameRef.current = frame;
      scheduleDraw();

      const now = performance.now();
      const times = frameTimesRef.current;
      times.push(now);
      while (times.length > 1 && now - times[0] > 2000) {
        times.shift();
      }
      if (now - lastReadoutRef.current >= READOUT_INTERVAL_MS) {
        lastReadoutRef.current = now;
        const span = times.length > 1 ? (times[times.length - 1] - times[0]) / 1000 : 0;
        setReadout({
          peakHz: frame.peak_hz,
          peakDb: frame.peak_db,
          rmsDb: frame.rms_db,
          clipping: frame.clipping,
          averaged: frame.averaged,
          captureMs: frame.capture_ms,
          fps: span > 0 ? (times.length - 1) / span : 0
        });
      }
    },
    [scheduleDraw]
  );

  const disconnect = useCallback(() => {
    const socket = socketRef.current;
    socketRef.current = null;
    if (socket) {
      socket.onclose = null;
      socket.close();
    }
    setLink("idle");
  }, []);

  const connect = useCallback(() => {
    if (socketRef.current) {
      return;
    }
    setMessage(null);
    setLink("connecting");
    frameTimesRef.current = [];
    const socket = new WebSocket(liveSocketUrl(api.backendUrl, window.location));
    socketRef.current = socket;

    socket.onmessage = (event: MessageEvent) => {
      const parsed = typeof event.data === "string" ? parseLiveMessage(event.data) : null;
      if (!parsed) {
        return;
      }
      if (parsed.type === "frame") {
        handleFrame(parsed);
      } else if (parsed.type === "info") {
        nyquistRef.current = parsed.nyquist_hz;
        setInfo(parsed);
        setSettings(parsed.settings);
        setLink("live");
      } else if (parsed.type === "settings") {
        setSettings({
          adc: parsed.adc,
          nfft: parsed.nfft,
          averaging: parsed.averaging,
          max_hold: parsed.max_hold,
          fps: parsed.fps,
          running: parsed.running
        });
      } else {
        setMessage(parsed.message);
      }
    };
    socket.onerror = () => {
      setMessage("Could not reach the live view on the board.");
    };
    socket.onclose = () => {
      socketRef.current = null;
      setLink("closed");
    };
  }, [handleFrame]);

  const send = useCallback((patch: LiveSettingsPatch) => {
    const socket = socketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(patch));
    }
  }, []);

  // close the connection when the panel goes away, so the board stops capturing
  useEffect(() => disconnect, [disconnect]);

  // keep the canvases sharp when the window is resized
  useEffect(() => {
    draw();
    window.addEventListener("resize", scheduleDraw);
    return () => window.removeEventListener("resize", scheduleDraw);
  }, [draw, scheduleDraw]);

  const live = link === "live" && settings !== null;
  const inputs: AdcInput[] = info?.inputs ?? ["A", "B", "C", "D"];
  const nfftChoices = info?.nfft_choices ?? [4096];
  const binHz = info && settings ? info.sample_rate_hz / settings.nfft : null;

  return (
    <section className="panel section-panel">
      <div className="section-heading">
        <p className="prompt-line">ADC input</p>
        <h2>Live view</h2>
      </div>

      <div className="live-toolbar">
        {link === "idle" || link === "closed" ? (
          <button type="button" onClick={connect}>
            Start live view
          </button>
        ) : (
          <button type="button" onClick={disconnect}>
            Stop
          </button>
        )}
        {live && settings && (
          <button type="button" className="mode-button active" onClick={() => send({ running: !settings.running })}>
            {settings.running ? "Pause" : "Resume"}
          </button>
        )}
        <span className="meta-note">
          {link === "idle" && "Not connected. The board only captures while the live view is open."}
          {link === "connecting" && "Connecting..."}
          {link === "live" && (settings?.running ? "Live" : "Paused")}
          {link === "closed" && "Disconnected."}
        </span>
      </div>
      {message && <p className="live-error">{message}</p>}

      {live && settings && (
        <div className="live-controls">
          <div className="field">
            <span className="field-label">Input</span>
            <div className="live-input-grid">
              {inputs.map((input) => (
                <button
                  key={input}
                  type="button"
                  className={settings.adc === input ? "mode-button active" : "mode-button"}
                  onClick={() => send({ adc: input })}
                >
                  ADC {input}
                </button>
              ))}
            </div>
          </div>
          <label className="field">
            <span className="field-label">FFT size</span>
            <select value={settings.nfft} onChange={(e) => send({ nfft: Number(e.target.value) })}>
              {nfftChoices.map((n) => (
                <option key={n} value={n}>
                  {n.toLocaleString()}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span className="field-label">Averaging (frames)</span>
            <select value={settings.averaging} onChange={(e) => send({ averaging: Number(e.target.value) })}>
              {AVERAGING_CHOICES.map((n) => (
                <option key={n} value={n}>
                  {n === 1 ? "off" : n}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span className="field-label">Frames per second</span>
            <select value={settings.fps} onChange={(e) => send({ fps: Number(e.target.value) })}>
              {FPS_CHOICES.filter((n) => n <= (info?.max_fps ?? 30)).map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <div className="field">
            <span className="field-label">Max hold</span>
            <div className="live-input-grid">
              <button
                type="button"
                className={settings.max_hold ? "mode-button active" : "mode-button"}
                onClick={() => send({ max_hold: !settings.max_hold })}
              >
                {settings.max_hold ? "On" : "Off"}
              </button>
              <button
                type="button"
                className="mode-button"
                disabled={!settings.max_hold}
                onClick={() => send({ reset_hold: true })}
              >
                Clear
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="live-plots">
        <article className="plot-card">
          <canvas ref={spectrumCanvasRef} className="live-canvas live-canvas-spectrum" />
        </article>
        <article className="plot-card">
          <canvas ref={traceCanvasRef} className="live-canvas live-canvas-trace" />
        </article>
      </div>

      {live && readout && (
        <div className="live-readouts">
          <div className="readout-row">
            <span>peak</span>
            <strong>
              {formatFrequency(readout.peakHz)} at {readout.peakDb.toFixed(1)} dBFS
            </strong>
          </div>
          <div className="readout-row">
            <span>total level</span>
            <strong className={readout.clipping ? "live-clipping" : undefined}>
              {readout.rmsDb.toFixed(1)} dBFS{readout.clipping ? " - clipping" : ""}
            </strong>
          </div>
          <div className="readout-row">
            <span>resolution</span>
            <strong>{binHz ? `${formatFrequency(binHz)} per bin` : "-"}</strong>
          </div>
          <div className="readout-row">
            <span>update</span>
            <strong>
              {readout.fps.toFixed(1)} frames/s, {readout.captureMs.toFixed(1)} ms per frame on the board
            </strong>
          </div>
        </div>
      )}

      <p className="subtitle fft-note">
        Each frame is a short snapshot (at most 53 µs) of the selected input, so this shows continuous
        or repeating signals; rare events are usually missed. Frequencies above{" "}
        {formatFrequency(nyquistRef.current)} fold back into the display.
      </p>
    </section>
  );
}
