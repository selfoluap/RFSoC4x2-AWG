import { useMemo } from "react";
import { useAppState } from "../context/AppStateContext";
import { PlotlyChart } from "./PlotlyChart";

export function WaveformPlots() {
  const { waveform, fft } = useAppState();
  const canPlotWaveform = useMemo(
    () => Boolean(waveform?.x_axis?.length && waveform?.signal?.length),
    [waveform]
  );

  return (
    <>
      {canPlotWaveform && waveform?.x_axis && waveform?.signal && (
        <section className="panel plot-panel">
          <div className="plot-stack">
            <article className="plot-card">
              <PlotlyChart
                title="Generated waveform"
                x={waveform.x_axis}
                y={waveform.signal}
                xLabel="Time (s)"
                yLabel="Amplitude"
                tone="signal"
              />
            </article>
            {waveform.captured && (
              <article className="plot-card">
                <PlotlyChart
                  title="Captured signal"
                  x={waveform.x_axis}
                  y={waveform.captured}
                  xLabel="Time (s)"
                  yLabel="Amplitude"
                  tone="capture"
                />
              </article>
            )}
            {waveform.captured_after_precorrection && (
              <article className="plot-card">
                <PlotlyChart
                  title="Captured after precorrection"
                  x={waveform.x_axis}
                  y={waveform.captured_after_precorrection}
                  xLabel="Time (s)"
                  yLabel="Amplitude"
                  tone="capture"
                />
              </article>
            )}
          </div>
        </section>
      )}

      {fft && (
        <section className="panel plot-panel">
          <div className="plot-stack">
            <article className="plot-card">
              <PlotlyChart
                title="FFT"
                x={fft.frequencies}
                y={fft.magnitudes}
                xLabel="Frequency (MHz)"
                yLabel="Magnitude (dB)"
                tone="spectrum"
              />
            </article>
            <p className="subtitle fft-note">
              This is the expected spectrum after modulation by the EOM. The waveform is mapped to
              a phase-modulated complex exponential, and the FFT is taken from that modulated field
              to show its spectral content.
            </p>
          </div>
        </section>
      )}
    </>
  );
}
