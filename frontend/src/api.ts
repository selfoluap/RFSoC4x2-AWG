import type {
  ConstantsResponse,
  ErrorMetrics,
  FFTResponse,
  SerrodyneForm,
  SimpleForm,
  StatusResponse,
  WaveformResponse
} from "./types";

const BACKEND_URL = import.meta.env.VITE_RFSOC_BACKEND_URL ?? "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BACKEND_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init
  });

  if (!response.ok) {
    throw new Error(`API Error (${response.status}): ${await response.text()}`);
  }

  return (await response.json()) as T;
}

export const api = {
  backendUrl: BACKEND_URL,
  getStatus: () => request<StatusResponse>("/status"),
  getConstants: () => request<ConstantsResponse>("/constants"),
  captureADC: () => request<WaveformResponse>("/capture"),
  getCaptureFft: () => request<FFTResponse>("/capture/fft"),
  getWaveformFft: () => request<FFTResponse>("/waveform/fft"),
  calculateErrorMetrics: () => request<ErrorMetrics>("/error_metrics", { method: "POST" }),
  generateSerrodyne: (payload: SerrodyneForm) =>
    request<WaveformResponse>(`/waveform/serrodyne?precorrection=${payload.precorrection}`, {
      method: "POST",
      body: JSON.stringify({
        ratios_str: payload.ratios_str,
        freqs_str: payload.freqs_str,
        T_total_us: payload.T_total_us,
        amp: payload.amp
      })
    }),
  generateSimple: (payload: SimpleForm) =>
    request<WaveformResponse>(`/waveform/simple?precorrection=${payload.precorrection}`, {
      method: "POST",
      body: JSON.stringify({
        waveform_type: payload.waveform_type,
        freq_mhz: payload.freq_mhz,
        amp: payload.amp,
        duty_cycle: payload.duty_cycle
      })
    })
};
