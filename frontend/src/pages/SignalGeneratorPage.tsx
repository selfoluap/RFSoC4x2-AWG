import { useState } from "react";
import { api } from "../api";
import { WaveformPlots } from "../components/WaveformPlots";
import { useAppState } from "../context/AppStateContext";
import type { SerrodyneForm, SimpleForm, WaveformType } from "../types";

type GeneratorType = WaveformType | "serrodyne";

export function SignalGeneratorPage() {
  const { loading, runWaveformGeneration, constants } = useAppState();
  const [generatorType, setGeneratorType] = useState<GeneratorType>("sine");
  const [simpleForm, setSimpleForm] = useState<SimpleForm>({
    waveform_type: "sine",
    freq_mhz: 250,
    amp: 16383,
    duty_cycle: 0.5,
  });
  const [serrodyneForm, setSerrodyneForm] = useState<SerrodyneForm>({
    ratios_str: "1:5:3",
    freqs_str: "-1330,0,840",
    T_total_us: 1,
    amp: 16383,
  });
  const [simpleScale, setSimpleScale] = useState(1);
  const [serrodyneScale, setSerrodyneScale] = useState(1);

  const isSerrodyne = generatorType === "serrodyne";
  const maxAmplitude = constants?.DAC_AMP ?? 16383;
  const activeScale = isSerrodyne ? serrodyneScale : simpleScale;
  const activeAmplitude = Math.round(maxAmplitude * activeScale);
  const profileLabel = isSerrodyne
    ? `ratios ${serrodyneForm.ratios_str}`
    : `${generatorType} @ ${simpleForm.freq_mhz} MHz`;

  const waveformTabs: Array<{ value: GeneratorType; label: string }> = [
    { value: "static", label: "Static" },
    { value: "sine", label: "Sine" },
    { value: "cos", label: "Cos" },
    { value: "square", label: "Square" },
    { value: "sawtooth", label: "Sawtooth" },
    { value: "serrodyne", label: "Serrodyne" },
  ];

  function handleGeneratorTypeChange(nextType: GeneratorType) {
    setGeneratorType(nextType);

    if (nextType !== "serrodyne") {
      setSimpleForm({ ...simpleForm, waveform_type: nextType });
    }
  }

  function getAmplitudeFromScale(scale: number) {
    return Math.round(maxAmplitude * Math.min(Math.max(scale, 0.01), 1));
  }

  return (
    <>
      <section className="panel section-panel">
        <div className="mode-toggle" role="tablist" aria-label="Waveform type">
          {waveformTabs.map((tab) => (
            <button
              key={tab.value}
              type="button"
              role="tab"
              aria-selected={generatorType === tab.value}
              className={generatorType === tab.value ? "mode-button active" : "mode-button"}
              onClick={() => handleGeneratorTypeChange(tab.value)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="generator-layout">
          <article className="module-card">
            <div className="field-grid">
              {isSerrodyne ? (
                <>
                  <label className="field">
                    <span className="field-label">Ratios</span>
                    <input
                      value={serrodyneForm.ratios_str}
                      onChange={(e) => setSerrodyneForm({ ...serrodyneForm, ratios_str: e.target.value })}
                    />
                  </label>
                  <label className="field">
                    <span className="field-label">Frequencies</span>
                    <input
                      value={serrodyneForm.freqs_str}
                      onChange={(e) => setSerrodyneForm({ ...serrodyneForm, freqs_str: e.target.value })}
                    />
                  </label>
                  <label className="field">
                    <span className="field-label">Total period (µs)</span>
                    <input
                      type="number"
                      value={serrodyneForm.T_total_us}
                      onChange={(e) => setSerrodyneForm({ ...serrodyneForm, T_total_us: Number(e.target.value) })}
                    />
                  </label>
                  <label className="field">
                    <span className="field-label">Scaling factor</span>
                    <input
                      type="number"
                      min="0.01"
                      max="1"
                      step="0.1"
                      value={serrodyneScale}
                      onChange={(e) => {
                        const nextScale = Math.min(Math.max(Number(e.target.value), 0.01), 1);
                        setSerrodyneScale(nextScale);
                        setSerrodyneForm({ ...serrodyneForm, amp: getAmplitudeFromScale(nextScale) });
                      }}
                    />
                  </label>
                </>
              ) : (
                <>
                  <label className="field">
                    <span className="field-label">Frequency (MHz)</span>
                    <input
                      type="number"
                      value={simpleForm.freq_mhz}
                      onChange={(e) => setSimpleForm({ ...simpleForm, freq_mhz: Number(e.target.value) })}
                    />
                  </label>
                  <label className="field">
                    <span className="field-label">Scaling factor</span>
                    <input
                      type="number"
                      min="0.01"
                      max="1"
                      step="0.1"
                      value={simpleScale}
                      onChange={(e) => {
                        const nextScale = Math.min(Math.max(Number(e.target.value), 0.01), 1);
                        setSimpleScale(nextScale);
                        setSimpleForm({ ...simpleForm, amp: getAmplitudeFromScale(nextScale) });
                      }}
                    />
                  </label>
                  <label className="field">
                    <span className="field-label">Duty cycle</span>
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="0.01"
                      value={simpleForm.duty_cycle}
                      onChange={(e) => setSimpleForm({ ...simpleForm, duty_cycle: Number(e.target.value) })}
                    />
                  </label>
                </>
              )}
            </div>
          </article>

          <article className="module-card module-card-emphasis">
            <div className="readout-list">
              <div className="readout-row">
                <span>profile</span>
                <strong>{profileLabel}</strong>
              </div>
              <div className="readout-row">
                <span>amplitude</span>
                <strong>{activeAmplitude}</strong>
              </div>
            </div>
            <div className="button-stack">
              <button
                disabled={loading}
                onClick={() =>
                  isSerrodyne
                    ? runWaveformGeneration("Serrodyne", () => api.generateSerrodyne(serrodyneForm))
                    : runWaveformGeneration("Waveform", () =>
                        api.generateSimple({
                          ...simpleForm,
                          waveform_type: generatorType as WaveformType,
                        })
                      )
                }
              >
                {loading ? "Running..." : isSerrodyne ? "Output serrodyne" : "Output waveform"}
              </button>
            </div>
          </article>
        </div>
      </section>

      <WaveformPlots />
    </>
  );
}
