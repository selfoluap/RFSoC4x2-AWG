import { PlotlyChart } from "./PlotlyChart";

interface WaveformPlotsProps {
  xAxis: number[];
  signal: number[];
  frequencies: number[];
  magnitudes: number[];
}

export function WaveformPlots({ xAxis, signal, frequencies, magnitudes }: WaveformPlotsProps) {
  const canPlotWaveform = Boolean(xAxis.length && signal.length);
  const canPlotFft = Boolean(frequencies.length && magnitudes.length);

  return (
    <>
      {canPlotWaveform && (
        <section className="panel plot-panel">
          <div className="plot-stack">
            <article className="plot-card">
              <PlotlyChart
                title="Preview waveform"
                x={xAxis}
                y={signal}
                xLabel="Time (s)"
                yLabel="Amplitude"
                tone="signal"
              />
            </article>
          </div>
        </section>
      )}

      {canPlotFft && (
        <section className="panel plot-panel">
          <div className="plot-stack">
            <article className="plot-card">
              <PlotlyChart
                title="Expected FFT"
                x={frequencies}
                y={magnitudes}
                xLabel="Frequency (MHz)"
                yLabel="Magnitude"
                tone="spectrum"
              />
            </article>
            <p className="subtitle fft-note">
              This is a browser-side expected spectrum preview, not live hardware feedback. The
              waveform is mapped to a phase-modulated complex exponential before the FFT.
            </p>
          </div>
        </section>
      )}
    </>
  );
}
