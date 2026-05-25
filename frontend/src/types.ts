export interface WaveformResponse {
  success: boolean;
  message: string;
  signal?: number[];
  captured?: number[];
  captured_after_precorrection?: number[];
  x_axis?: number[];
  num_samples?: number;
}

export interface FFTResponse {
  frequencies: number[];
  magnitudes: number[];
}

export interface ErrorMetrics {
  E: number;
  E_prime: number;
  E_norm: number;
  E_prime_norm: number;
}

export interface StatusResponse {
  offline_mode: boolean;
  hardware_initialized: boolean;
  buf_len: number;
  dac_sr: number;
  adc_sr: number;
}

export interface ConstantsResponse {
  DAC_SR: number;
  ADC_SR: number;
  DAC_AMP: number;
  BUF_LEN: number;
  OFFLINE_MODE: boolean;
}

export type WaveformType = "static" | "sine" | "cos" | "sawtooth" | "square";

export interface SerrodyneForm {
  ratios_str: string;
  freqs_str: string;
  T_total_us: number;
  amp: number;
  precorrection: boolean;
}

export interface SimpleForm {
  waveform_type: WaveformType;
  freq_mhz: number;
  amp: number;
  duty_cycle: number;
  precorrection: boolean;
}
