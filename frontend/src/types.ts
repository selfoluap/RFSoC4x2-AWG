export interface WaveformResponse {
  success: boolean;
  message: string;
  signal?: number[];
  x_axis?: number[];
  num_samples?: number;
}

export interface FFTResponse {
  frequencies: number[];
  magnitudes: number[];
}

export interface StatusResponse {
  hardware_initialized: boolean;
  buf_len: number;
  dac_sr: number;
  dacs: Record<DacChannel, DacStatus>;
}

export interface ConstantsResponse {
  DAC_SR: number;
  DAC_AMP: number;
  BUF_LEN: number;
  overlay_info: Record<string, unknown>;
}

export type WaveformType = "static" | "sine" | "cos" | "sawtooth" | "square";
export type DacChannel = "dac0" | "dac2";

export interface DacStatus {
  enabled: boolean;
  waveform_length: number;
}

export interface WaveformLoadResponse {
  success: boolean;
  message: string;
  channels: DacChannel[];
  num_samples: number;
  dacs: Record<DacChannel, DacStatus>;
}

export interface DacControlResponse {
  success: boolean;
  channel: DacChannel;
  enabled: boolean;
  dacs: Record<DacChannel, DacStatus>;
}

export interface SerrodyneForm {
  channels: DacChannel[];
  ratios_str: string;
  freqs_str: string;
  T_total_us: number;
  amp: number;
}

export interface SimpleForm {
  channels: DacChannel[];
  waveform_type: WaveformType;
  freq_mhz: number;
  amp: number;
  duty_cycle: number;
}
