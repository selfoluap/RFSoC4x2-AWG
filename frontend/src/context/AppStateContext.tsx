import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import { api } from "../api";
import type { ConstantsResponse, ErrorMetrics, FFTResponse, StatusResponse, WaveformResponse } from "../types";

interface AppState {
  status: StatusResponse | null;
  constants: ConstantsResponse | null;
  waveform: WaveformResponse | null;
  fft: FFTResponse | null;
  metrics: ErrorMetrics | null;
  loading: boolean;
  error: string | null;
  setStatus: (v: StatusResponse | null) => void;
  setConstants: (v: ConstantsResponse | null) => void;
  setWaveform: (v: WaveformResponse | null) => void;
  setFft: (v: FFTResponse | null) => void;
  setMetrics: (v: ErrorMetrics | null) => void;
  loadBackendSummary: () => Promise<void>;
  run: (label: string, fn: () => Promise<void>) => Promise<void>;
  runWaveformGeneration: (label: string, generate: () => Promise<WaveformResponse>) => Promise<void>;
}

const AppStateContext = createContext<AppState | null>(null);

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [constants, setConstants] = useState<ConstantsResponse | null>(null);
  const [waveform, setWaveform] = useState<WaveformResponse | null>(null);
  const [fft, setFft] = useState<FFTResponse | null>(null);
  const [metrics, setMetrics] = useState<ErrorMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async (label: string, fn: () => Promise<void>) => {
    try {
      setLoading(true);
      setError(null);
      await fn();
    } catch (e) {
      setError(`${label} failed: ${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  const runWaveformGeneration = useCallback(async (label: string, generate: () => Promise<WaveformResponse>) => {
    await run(label, async () => {
      setWaveform(await generate());
      setFft(await api.getWaveformFft());
    });
  }, [run]);

  const loadBackendSummary = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const [statusResult, constantsResult] = await Promise.allSettled([api.getStatus(), api.getConstants()]);
      let hasSuccess = false;

      if (statusResult.status === "fulfilled") {
        setStatus(statusResult.value);
        hasSuccess = true;
      } else {
        setStatus(null);
      }

      if (constantsResult.status === "fulfilled") {
        setConstants(constantsResult.value);
        hasSuccess = true;
      } else {
        setConstants(null);
      }

      if (!hasSuccess) {
        const errors = [statusResult, constantsResult]
          .filter((result) => result.status === "rejected")
          .map((result) => (result as PromiseRejectedResult).reason as Error)
          .map((reason) => reason.message)
          .join(" | ");
        setError(`Backend summary failed: ${errors}`);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  const value = useMemo(
    () => ({
      status,
      constants,
      waveform,
      fft,
      metrics,
      loading,
      error,
      setStatus,
      setConstants,
      setWaveform,
      setFft,
      setMetrics,
      loadBackendSummary,
      run,
      runWaveformGeneration
    }),
    [status, constants, waveform, fft, metrics, loading, error, loadBackendSummary, run, runWaveformGeneration]
  );

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState() {
  const ctx = useContext(AppStateContext);
  if (!ctx) {
    throw new Error("useAppState must be used within AppStateProvider");
  }
  return ctx;
}
