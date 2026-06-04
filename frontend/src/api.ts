import type {
  ConstantsResponse,
  DacChannel,
  DacControlResponse,
  FFTResponse,
  SerrodyneForm,
  SimpleForm,
  StatusResponse,
  WaveformLoadResponse
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
  getWaveformFft: () => request<FFTResponse>("/waveform/fft"),
  loadSerrodyne: (payload: SerrodyneForm) =>
    request<WaveformLoadResponse>("/waveform/serrodyne/load", {
      method: "POST",
      body: JSON.stringify({
        channels: payload.channels,
        ratios_str: payload.ratios_str,
        freqs_str: payload.freqs_str,
        T_total_us: payload.T_total_us,
        amp: payload.amp
      })
    }),
  loadSimple: (payload: SimpleForm) =>
    request<WaveformLoadResponse>("/waveform/simple/load", {
      method: "POST",
      body: JSON.stringify({
        channels: payload.channels,
        waveform_type: payload.waveform_type,
        freq_mhz: payload.freq_mhz,
        amp: payload.amp,
        duty_cycle: payload.duty_cycle
      })
    }),
  enableDac: (channel: DacChannel) =>
    request<DacControlResponse>(`/dac/${channel}/enable`, { method: "POST" }),
  disableDac: (channel: DacChannel) =>
    request<DacControlResponse>(`/dac/${channel}/disable`, { method: "POST" }),
  disableAllDacs: () => request<{ success: boolean }>("/dac/all/disable", { method: "POST" })
};
