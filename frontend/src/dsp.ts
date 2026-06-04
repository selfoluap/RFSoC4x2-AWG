import type { SerrodyneForm, SimpleForm, WaveformType } from "./types";

export interface PreviewData {
  x: number[];
  y: number[];
}

export interface SpectrumData {
  frequencies: number[];
  magnitudes: number[];
}

const TWO_PI = 2 * Math.PI;

function sawtoothUnit(phase: number) {
  const cycle = ((phase / TWO_PI) % 1 + 1) % 1;
  return 2 * cycle - 1;
}

function squareUnit(phase: number, duty: number) {
  const cycle = ((phase / TWO_PI) % 1 + 1) % 1;
  return cycle < duty ? 1 : -1;
}

function parseRatios(value: string) {
  const parts = value.includes(":") ? value.split(":") : value.split(",");
  const ratios = parts.map((part) => Number.parseInt(part.trim(), 10)).filter((ratio) => Number.isFinite(ratio));
  if (!ratios.length || ratios.some((ratio) => ratio <= 0)) {
    return [1];
  }
  return ratios;
}

function parseFreqsMhz(value: string) {
  const freqs = value
    .replaceAll("MHz", "")
    .split(",")
    .map((part) => Number.parseFloat(part.trim()))
    .filter((freq) => Number.isFinite(freq));
  return freqs.length ? freqs.map((freq) => freq * 1e6) : [0];
}

export function generateSimplePreview(form: SimpleForm, sampleRate: number, points: number): PreviewData {
  const x: number[] = [];
  const y: number[] = [];
  const freqHz = form.freq_mhz * 1e6;
  const duty = Math.min(Math.max(form.duty_cycle, 0), 1);

  for (let i = 0; i < points; i += 1) {
    const t = i / sampleRate;
    const phase = TWO_PI * freqHz * t;
    let value = 0;

    switch (form.waveform_type as WaveformType) {
      case "sine":
        value = form.amp * Math.sin(phase);
        break;
      case "cos":
        value = form.amp * Math.cos(phase);
        break;
      case "sawtooth":
        value = form.amp * sawtoothUnit(phase);
        break;
      case "square":
        value = form.amp * squareUnit(phase, duty);
        break;
      case "static":
      default:
        value = 0;
        break;
    }

    x.push(t);
    y.push(value);
  }

  return { x, y };
}

export function generateSerrodynePreview(form: SerrodyneForm, sampleRate: number, points: number): PreviewData {
  const x: number[] = [];
  const y = new Array<number>(points).fill(0);
  const ratios = parseRatios(form.ratios_str);
  const freqs = parseFreqsMhz(form.freqs_str);
  const ratioSum = ratios.reduce((sum, ratio) => sum + ratio, 0);
  const segmentLengths = ratios.map((ratio) => Math.round(points * (ratio / ratioSum)));
  let delta = points - segmentLengths.reduce((sum, length) => sum + length, 0);

  for (let index = 0; delta !== 0 && index < segmentLengths.length * 4; index += 1) {
    const segmentIndex = index % segmentLengths.length;
    const nextLength = segmentLengths[segmentIndex] + (delta > 0 ? 1 : -1);
    if (nextLength >= 0) {
      segmentLengths[segmentIndex] = nextLength;
      delta = points - segmentLengths.reduce((sum, length) => sum + length, 0);
    }
  }

  for (let i = 0; i < points; i += 1) {
    x.push(i / sampleRate);
  }

  let start = 0;
  for (let segmentIndex = 0; segmentIndex < segmentLengths.length; segmentIndex += 1) {
    const length = segmentLengths[segmentIndex];
    const freqHz = freqs[segmentIndex] ?? 0;
    for (let local = 0; local < length && start + local < points; local += 1) {
      const t = local / sampleRate;
      y[start + local] = freqHz === 0 ? 0 : form.amp * sawtoothUnit(TWO_PI * freqHz * t);
    }
    start += length;
  }

  return { x, y };
}

function fft(real: number[], imag: number[]) {
  const n = real.length;
  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const angle = -TWO_PI / len;
    const wLenReal = Math.cos(angle);
    const wLenImag = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let wReal = 1;
      let wImag = 0;
      for (let j = 0; j < len / 2; j += 1) {
        const uReal = real[i + j];
        const uImag = imag[i + j];
        const vReal = real[i + j + len / 2] * wReal - imag[i + j + len / 2] * wImag;
        const vImag = real[i + j + len / 2] * wImag + imag[i + j + len / 2] * wReal;
        real[i + j] = uReal + vReal;
        imag[i + j] = uImag + vImag;
        real[i + j + len / 2] = uReal - vReal;
        imag[i + j + len / 2] = uImag - vImag;
        const nextWReal = wReal * wLenReal - wImag * wLenImag;
        wImag = wReal * wLenImag + wImag * wLenReal;
        wReal = nextWReal;
      }
    }
  }
}

export function computeExpectedSpectrum(signal: number[], sampleRate: number): SpectrumData {
  const n = signal.length;
  const min = Math.min(...signal);
  const max = Math.max(...signal);
  const span = max - min;

  if (!n || span === 0) {
    return { frequencies: [], magnitudes: [] };
  }

  const phaseScale = TWO_PI / span;
  const real = signal.map((value) => Math.cos(phaseScale * value));
  const imag = signal.map((value) => Math.sin(phaseScale * value));
  fft(real, imag);

  const frequencies: number[] = [];
  const magnitudes: number[] = [];
  const half = n / 2;

  for (let i = 0; i < n; i += 1) {
    const sourceIndex = (i + half) % n;
    const frequencyHz = (i - half) * (sampleRate / n);
    frequencies.push(frequencyHz / 1e6);
    magnitudes.push(Math.hypot(real[sourceIndex], imag[sourceIndex]));
  }

  return { frequencies, magnitudes };
}
