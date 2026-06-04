import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import { api } from "../api";
import type { ConstantsResponse, StatusResponse } from "../types";

interface AppState {
  status: StatusResponse | null;
  constants: ConstantsResponse | null;
  loading: boolean;
  error: string | null;
  setStatus: (v: StatusResponse | null) => void;
  setConstants: (v: ConstantsResponse | null) => void;
  loadBackendSummary: () => Promise<void>;
  run: (label: string, fn: () => Promise<void>) => Promise<void>;
}

const AppStateContext = createContext<AppState | null>(null);

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [constants, setConstants] = useState<ConstantsResponse | null>(null);
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
      loading,
      error,
      setStatus,
      setConstants,
      loadBackendSummary,
      run
    }),
    [status, constants, loading, error, loadBackendSummary, run]
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
