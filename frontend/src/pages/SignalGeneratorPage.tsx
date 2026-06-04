import { useMemo, useState } from "react";
import { api } from "../api";
import { WaveformPlots } from "../components/WaveformPlots";
import { useAppState } from "../context/AppStateContext";
import { computeExpectedSpectrum, generateSerrodynePreview, generateSimplePreview } from "../dsp";
import type { DacChannel, SerrodyneForm, SimpleForm, WaveformType } from "../types";

type GeneratorType = WaveformType | "serrodyne";

const PREVIEW_POINTS = 4096;
const DAC_CHANNELS: DacChannel[] = ["dac0", "dac2"];

export function SignalGeneratorPage() {
  const { loading, run, constants, status, setStatus } = useAppState();
  const [generatorType, setGeneratorType] = useState<GeneratorType>("sine");
  const [selectedChannels, setSelectedChannels] = useState<DacChannel[]>(["dac0"]);
  const [loadMessage, setLoadMessage] = useState<string | null>(null);
  const [simpleForm, setSimpleForm] = useState<SimpleForm>({
    channels: ["dac0"],
    waveform_type: "sine",
    freq_mhz: 250,
    amp: 16383,
    duty_cycle: 0.5,
  });
  const [serrodyneForm, setSerrodyneForm] = useState<SerrodyneForm>({
    channels: ["dac0"],
    ratios_str: "1:5:3",
    freqs_str: "-1330,0,840",
    T_total_us: 1,
    amp: 16383,
  });
  const [simpleScale, setSimpleScale] = useState(1);
  const [serrodyneScale, setSerrodyneScale] = useState(1);

  const isSerrodyne = generatorType === "serrodyne";
  const maxAmplitude = constants?.DAC_AMP ?? 16383;
  const sampleRate = constants?.DAC_SR ?? status?.dac_sr ?? 9.8304e9;
  const activeScale = isSerrodyne ? serrodyneScale : simpleScale;
  const activeAmplitude = Math.round(maxAmplitude * activeScale);
  const profileLabel = isSerrodyne
    ? `ratios ${serrodyneForm.ratios_str}`
    : `${generatorType} @ ${simpleForm.freq_mhz} MHz`;
  const selectedLabel = selectedChannels.length ? selectedChannels.map((channel) => channel.toUpperCase()).join(" + ") : "none";

  const waveformTabs: Array<{ value: GeneratorType; label: string }> = [
    { value: "static", label: "Static" },
    { value: "sine", label: "Sine" },
    { value: "cos", label: "Cos" },
    { value: "square", label: "Square" },
    { value: "sawtooth", label: "Sawtooth" },
    { value: "serrodyne", label: "Serrodyne" },
  ];

  const preview = useMemo(
    () =>
      isSerrodyne
        ? generateSerrodynePreview({ ...serrodyneForm, amp: activeAmplitude }, sampleRate, PREVIEW_POINTS)
        : generateSimplePreview(
            {
              ...simpleForm,
              waveform_type: generatorType as WaveformType,
              amp: activeAmplitude,
            },
            sampleRate,
            PREVIEW_POINTS
          ),
    [activeAmplitude, generatorType, isSerrodyne, sampleRate, serrodyneForm, simpleForm]
  );
  const spectrum = useMemo(() => computeExpectedSpectrum(preview.y, sampleRate), [preview.y, sampleRate]);

  function handleGeneratorTypeChange(nextType: GeneratorType) {
    setGeneratorType(nextType);

    if (nextType !== "serrodyne") {
      setSimpleForm({ ...simpleForm, waveform_type: nextType });
    }
  }

  function getAmplitudeFromScale(scale: number) {
    return Math.round(maxAmplitude * Math.min(Math.max(scale, 0.01), 1));
  }

  function toggleSelectedChannel(channel: DacChannel) {
    setSelectedChannels((current) =>
      current.includes(channel) ? current.filter((value) => value !== channel) : [...current, channel]
    );
  }

  async function refreshStatus() {
    setStatus(await api.getStatus());
  }

  function loadWaveform() {
    void run("Load waveform", async () => {
      if (!selectedChannels.length) {
        throw new Error("Select at least one DAC channel");
      }

      const response = isSerrodyne
        ? await api.loadSerrodyne({ ...serrodyneForm, channels: selectedChannels, amp: activeAmplitude })
        : await api.loadSimple({
            ...simpleForm,
            channels: selectedChannels,
            waveform_type: generatorType as WaveformType,
            amp: activeAmplitude,
          });

      setLoadMessage(response.message);
      await refreshStatus();
    });
  }

  function setDacOutput(channel: DacChannel, enabled: boolean) {
    void run(`${enabled ? "Enable" : "Disable"} ${channel.toUpperCase()}`, async () => {
      if (enabled) {
        await api.enableDac(channel);
      } else {
        await api.disableDac(channel);
      }
      await refreshStatus();
    });
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
                    <span className="field-label">Total period (us)</span>
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
              <div className="readout-row">
                <span>target DACs</span>
                <strong>{selectedLabel}</strong>
              </div>
            </div>

            <div className="dac-select-grid">
              {DAC_CHANNELS.map((channel) => (
                <button
                  key={channel}
                  type="button"
                  className={selectedChannels.includes(channel) ? "mode-button active" : "mode-button"}
                  onClick={() => toggleSelectedChannel(channel)}
                >
                  Load {channel.toUpperCase()}
                </button>
              ))}
            </div>

            <div className="button-stack">
              <button disabled={loading || !selectedChannels.length} onClick={loadWaveform}>
                {loading ? "Sending..." : "Load to FPGA"}
              </button>
              {loadMessage && <p className="meta-note">{loadMessage}</p>}
            </div>
          </article>
        </div>
      </section>

      <section className="panel section-panel">
        <div className="section-heading">
          <p className="prompt-line">DAC output</p>
          <h2>Output enable</h2>
        </div>
        <div className="dac-output-grid">
          {DAC_CHANNELS.map((channel) => {
            const enabled = Boolean(status?.dacs?.[channel]?.enabled);
            const waveformLength = status?.dacs?.[channel]?.waveform_length ?? 0;
            return (
              <article key={channel} className="module-card">
                <div className="readout-row">
                  <span>{channel.toUpperCase()}</span>
                  <strong>{enabled ? "enabled" : "disabled"}</strong>
                </div>
                <div className="readout-row">
                  <span>loaded samples</span>
                  <strong>{waveformLength.toLocaleString()}</strong>
                </div>
                <button
                  type="button"
                  className={enabled ? "dac-toggle enabled" : "dac-toggle disabled"}
                  disabled={loading}
                  onClick={() => setDacOutput(channel, !enabled)}
                >
                  {enabled ? `Disable ${channel.toUpperCase()}` : `Enable ${channel.toUpperCase()}`}
                </button>
              </article>
            );
          })}
        </div>
      </section>

      <WaveformPlots
        xAxis={preview.x}
        signal={preview.y}
        frequencies={spectrum.frequencies}
        magnitudes={spectrum.magnitudes}
      />
    </>
  );
}
