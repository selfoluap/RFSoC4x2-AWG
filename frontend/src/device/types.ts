// Device state as served by /device/info and /device/status (firmware/device.py).

import type { DacChannel } from "../types";

export type PlaybackMode = "none" | "loop" | "sequence" | "long";
export type OutputState = "idle" | "armed" | "running" | "holding" | "done";

export interface Capabilities {
  loop: boolean;
  sequence: boolean;
  long: boolean;
  external_trigger: boolean;
  adc: boolean;
  analog_trigger: boolean;
}

export interface ChannelInfo {
  channel: DacChannel;
  loop_samples: number;
  sequence_samples?: number;
  long_bytes?: number;
}

export interface DeviceInfo {
  dac_sample_rate_hz: number;
  capabilities: Capabilities;
  channels: ChannelInfo[];
  bitfile: string;
  clock_source: string;
}

export interface OutputStatus {
  channel: DacChannel;
  mode: PlaybackMode;
  state: OutputState;
  block: number;
  loaded_samples: number;
  warnings: string[];
  buffer_low_water: number | null;
}

export interface DeviceStatus {
  outputs: OutputStatus[];
}

export const MODE_LABEL: Record<PlaybackMode, string> = {
  none: "nothing loaded",
  loop: "loop",
  sequence: "sequence",
  long: "long waveform (DDR4)"
};

export const STATE_LABEL: Record<OutputState, string> = {
  idle: "idle",
  armed: "armed, waiting for trigger",
  running: "running",
  holding: "holding last block",
  done: "done"
};

/** Duration of the loaded samples at the full DAC rate, e.g. "13.33 µs". */
export function formatDuration(samples: number, sampleRateHz: number): string {
  const seconds = samples / sampleRateHz;
  if (seconds >= 1e-3) {
    return `${(seconds * 1e3).toFixed(2)} ms`;
  }
  return `${(seconds * 1e6).toFixed(2)} µs`;
}

/** Build features as short labels, for the system card. */
export function capabilityLabels(capabilities: Capabilities): string[] {
  const labels = ["loop playback"];
  if (capabilities.sequence) labels.push("sequence mode");
  if (capabilities.long) labels.push("DDR4 long waveforms");
  if (capabilities.external_trigger) labels.push("external trigger");
  if (capabilities.adc) labels.push("ADC capture");
  if (capabilities.analog_trigger) labels.push("analog trigger");
  return labels;
}
