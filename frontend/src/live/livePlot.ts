// Canvas rendering for the live ADC view. Plotly redraws too slowly for 15 frames per second,
// so the live traces are drawn directly. No React in here: the functions take a canvas and a
// frame and can be exercised on their own.

import type { LiveFrame } from "./protocol";

export interface PlotColors {
  spectrum: string;
  hold: string;
  trace: string;
  grid: string;
  axis: string;
  text: string;
  marker: string;
}

export const DEFAULT_COLORS: PlotColors = {
  spectrum: "#f0c36e",
  hold: "rgba(243, 139, 139, 0.75)",
  trace: "#61d4b3",
  grid: "rgba(237, 242, 247, 0.08)",
  axis: "rgba(237, 242, 247, 0.16)",
  text: "#94a3b8",
  marker: "#edf2f7"
};

const MARGIN = { left: 46, right: 12, top: 12, bottom: 26 };
const FONT = '11px "IBM Plex Mono", monospace';

interface Area {
  ctx: CanvasRenderingContext2D;
  x0: number;
  y0: number;
  width: number;
  height: number;
}

/** Sizes the backing store to the element and device pixel ratio, clears it, returns the plot area. */
function prepare(canvas: HTMLCanvasElement): Area | null {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return null;
  }
  const ratio = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth || 600;
  const cssHeight = canvas.clientHeight || 240;
  const pixelWidth = Math.round(cssWidth * ratio);
  const pixelHeight = Math.round(cssHeight * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  ctx.font = FONT;
  return {
    ctx,
    x0: MARGIN.left,
    y0: MARGIN.top,
    width: Math.max(10, cssWidth - MARGIN.left - MARGIN.right),
    height: Math.max(10, cssHeight - MARGIN.top - MARGIN.bottom)
  };
}

function frameBox(area: Area, colors: PlotColors): void {
  area.ctx.strokeStyle = colors.axis;
  area.ctx.lineWidth = 1;
  area.ctx.strokeRect(area.x0 + 0.5, area.y0 + 0.5, area.width, area.height);
}

function polyline(area: Area, xs: (index: number) => number, ys: number[], toY: (value: number) => number): void {
  const { ctx } = area;
  ctx.beginPath();
  for (let i = 0; i < ys.length; i += 1) {
    const x = xs(i);
    const y = Math.min(area.y0 + area.height, Math.max(area.y0, toY(ys[i])));
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();
}

/** A tick step of 1, 2 or 5 times a power of ten that gives at most `maxTicks` ticks. */
export function niceStep(span: number, maxTicks: number): number {
  const raw = span / Math.max(1, maxTicks);
  const power = Math.pow(10, Math.floor(Math.log10(raw)));
  const scaled = raw / power;
  const factor = scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
  return factor * power;
}

export interface SpectrumOptions {
  nyquistHz: number;
  minDb?: number;
  maxDb?: number;
}

/** Spectrum in dBFS from 0 Hz to Nyquist, with the optional max-hold trace and a peak marker. */
export function drawSpectrum(
  canvas: HTMLCanvasElement,
  frame: LiveFrame | null,
  options: SpectrumOptions,
  colors: PlotColors = DEFAULT_COLORS
): void {
  const area = prepare(canvas);
  if (!area) {
    return;
  }
  const { ctx } = area;
  const minDb = options.minDb ?? -110;
  const maxDb = options.maxDb ?? 0;
  const toX = (hz: number) => area.x0 + (hz / options.nyquistHz) * area.width;
  const toY = (db: number) => area.y0 + ((maxDb - db) / (maxDb - minDb)) * area.height;

  // grid and labels
  ctx.fillStyle = colors.text;
  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  ctx.textBaseline = "middle";
  ctx.textAlign = "right";
  for (let db = maxDb; db >= minDb; db -= 20) {
    const y = Math.round(toY(db)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(area.x0, y);
    ctx.lineTo(area.x0 + area.width, y);
    ctx.stroke();
    ctx.fillText(`${db}`, area.x0 - 6, y);
  }
  ctx.textBaseline = "top";
  ctx.textAlign = "center";
  const stepHz = niceStep(options.nyquistHz, Math.floor(area.width / 70));
  for (let hz = 0; hz <= options.nyquistHz + 1; hz += stepHz) {
    const x = Math.round(toX(hz)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, area.y0);
    ctx.lineTo(x, area.y0 + area.height);
    ctx.stroke();
    ctx.fillText(`${Math.round(hz / 1e6)}`, x, area.y0 + area.height + 6);
  }
  ctx.textAlign = "right";
  ctx.fillText("MHz", area.x0 + area.width - 6, area.y0 + area.height - 16);
  ctx.textAlign = "left";
  ctx.fillText("dBFS", area.x0 + 6, area.y0 + 4);
  frameBox(area, colors);

  if (!frame || frame.spectrum_db.length === 0) {
    return;
  }
  const binX = (index: number) => toX(frame.f_start_hz + index * frame.f_step_hz);

  if (frame.max_hold_db) {
    ctx.strokeStyle = colors.hold;
    ctx.lineWidth = 1;
    polyline(area, binX, frame.max_hold_db, toY);
  }
  ctx.strokeStyle = colors.spectrum;
  ctx.lineWidth = 1.5;
  polyline(area, binX, frame.spectrum_db, toY);

  // peak marker
  const px = toX(frame.peak_hz);
  const py = Math.max(area.y0 + 8, toY(frame.peak_db));
  ctx.fillStyle = colors.marker;
  ctx.beginPath();
  ctx.moveTo(px, py - 3);
  ctx.lineTo(px - 4, py - 10);
  ctx.lineTo(px + 4, py - 10);
  ctx.closePath();
  ctx.fill();
}

/** Start of the captured record against time, auto-scaled to the signal. */
export function drawTrace(
  canvas: HTMLCanvasElement,
  frame: LiveFrame | null,
  colors: PlotColors = DEFAULT_COLORS
): void {
  const area = prepare(canvas);
  if (!area) {
    return;
  }
  const { ctx } = area;
  const samples = frame?.trace ?? [];
  let peak = 64;
  for (const value of samples) {
    peak = Math.max(peak, Math.abs(value));
  }
  const scale = Math.min(32768, peak * 1.4);   // headroom keeps the label clear of the trace
  const count = Math.max(2, samples.length);
  const totalNs = (frame ? frame.trace_dt_s * (count - 1) : 100e-9) * 1e9;
  const toX = (index: number) => area.x0 + (index / (count - 1)) * area.width;
  const toY = (value: number) => area.y0 + (0.5 - value / (2 * scale)) * area.height;

  ctx.fillStyle = colors.text;
  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  ctx.textBaseline = "middle";
  ctx.textAlign = "right";
  for (const fraction of [-1, -0.5, 0, 0.5, 1]) {
    const y = Math.round(toY(fraction * scale)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(area.x0, y);
    ctx.lineTo(area.x0 + area.width, y);
    ctx.stroke();
    if (fraction !== -0.5 && fraction !== 0.5) {
      ctx.fillText(`${Math.round(fraction * scale)}`, area.x0 - 6, y);
    }
  }
  ctx.textBaseline = "top";
  ctx.textAlign = "center";
  const stepNs = niceStep(totalNs, Math.floor(area.width / 80));
  for (let ns = 0; ns <= totalNs + 1e-9; ns += stepNs) {
    const x = Math.round(area.x0 + (ns / totalNs) * area.width) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, area.y0);
    ctx.lineTo(x, area.y0 + area.height);
    ctx.stroke();
    ctx.fillText(`${Math.round(ns * 10) / 10}`, x, area.y0 + area.height + 6);
  }
  ctx.textAlign = "left";
  ctx.fillText("ADC counts, time in ns", area.x0 + 6, area.y0 + 4);
  frameBox(area, colors);

  // trigger thresholds, when the trigger watches the displayed input
  const trigger = frame?.trigger;
  if (frame && trigger && trigger.enabled && trigger.adc === frame.adc) {
    const levels = trigger.mode === "envelope" ? [trigger.threshold, -trigger.threshold] : [trigger.threshold];
    ctx.strokeStyle = colors.marker;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    for (const level of levels) {
      if (Math.abs(level) <= scale) {
        const y = Math.round(toY(level)) + 0.5;
        ctx.beginPath();
        ctx.moveTo(area.x0, y);
        ctx.lineTo(area.x0 + area.width, y);
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);
  }

  if (samples.length > 1) {
    ctx.strokeStyle = colors.trace;
    ctx.lineWidth = 1.5;
    polyline(area, toX, samples, toY);
  }
}
