// Messages of the live ADC view (WebSocket /adc/live). Mirrors backend/services/live.py.

export type AdcInput = "A" | "B" | "C" | "D";

export interface LiveSettings {
  adc: AdcInput;
  nfft: number;
  averaging: number;
  max_hold: boolean;
  fps: number;
  running: boolean;
  trigger: TriggerMode;
}

/** "free": snapshots at any time. "analog": each record starts at an analog trigger. */
export type TriggerMode = "free" | "analog";
export type AnalogTriggerType = "envelope" | "rising" | "falling";

/** Settings and status of the analog trigger (GET/POST /adc/trigger, and in every frame). */
export interface AnalogTriggerInfo {
  enabled: boolean;
  adc: AdcInput;
  mode: AnalogTriggerType;
  threshold: number;
  off_threshold: number;
  release_s: number;
  active: boolean;
  peak: number;
  peak_dbfs: number;
  events: number;
}

export interface LiveInfo {
  type: "info";
  sample_rate_hz: number;
  nyquist_hz: number;
  capacity_samples: number;
  inputs: AdcInput[];
  nfft_choices: number[];
  max_fps: number;
  trigger_modes: TriggerMode[];
  settings: LiveSettings;
}

export interface LiveFrame {
  type: "frame";
  seq: number;
  adc: AdcInput;
  nfft: number;
  segments: number;
  averaged: number;
  f_start_hz: number;
  f_step_hz: number;
  spectrum_db: number[];
  max_hold_db: number[] | null;
  peak_hz: number;
  peak_db: number;
  rms_db: number;
  clipping: boolean;
  trace: number[];
  trace_dt_s: number;
  /** null: free running; true: starts at a trigger; false: no trigger came, taken anyway */
  triggered: boolean | null;
  trigger: AnalogTriggerInfo | null;
  capture_ms: number;
}

export interface LiveSettingsMessage extends LiveSettings {
  type: "settings";
}

export interface LiveError {
  type: "error";
  message: string;
}

export type LiveMessage = LiveInfo | LiveFrame | LiveSettingsMessage | LiveError;

/** Settings the browser may send; reset_hold clears the max-hold trace once. */
export type LiveSettingsPatch = Partial<LiveSettings> & { reset_hold?: boolean };

/** WebSocket address of the live view for a backend URL such as "/api" or "http://board:8001". */
export function liveSocketUrl(backendUrl: string, location: { protocol: string; host: string }): string {
  const base = backendUrl.replace(/\/+$/, "");
  if (/^https?:\/\//i.test(base)) {
    return `${base.replace(/^http/i, "ws")}/adc/live`;
  }
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${location.host}${base}/adc/live`;
}

export function parseLiveMessage(text: string): LiveMessage | null {
  try {
    const value: unknown = JSON.parse(text);
    if (typeof value !== "object" || value === null) {
      return null;
    }
    const type = (value as { type?: unknown }).type;
    if (type === "info" || type === "frame" || type === "settings" || type === "error") {
      return value as LiveMessage;
    }
    return null;
  } catch {
    return null;
  }
}

export function formatFrequency(hz: number): string {
  if (Math.abs(hz) >= 1e9) {
    return `${(hz / 1e9).toFixed(4)} GHz`;
  }
  return `${(hz / 1e6).toFixed(2)} MHz`;
}
