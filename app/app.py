# Imports and overlay loading
from scipy.signal import sawtooth
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import asyncio
from awg import plot_time_series_interactive, plot_fft, make_piecewise_ramp, calculate_fft, parse_freqs_mhz, parse_ratios

#all of these type ignores are only to keep my IDE happy
import streamlit as st # type: ignore
import plotly.graph_objects as go # type: ignore

# RFSoC overlay and OLED display
from rfsoc4x2 import oled # type: ignore
from rfsoc_mts import mtsOverlay # type: ignore


st.set_page_config(page_title="LaserLab: RFSoC AWG", layout="wide")
st.title("LaserLab: RFSoC AWG")

def _ensure_event_loop_for_streamlit_thread():
    try:
        # Raises in non-main threads if no loop was set
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

if "oled" not in st.session_state:
    st.session_state["oled"] = oled.oled_display()

_ensure_event_loop_for_streamlit_thread()


# Load overlay once per Streamlit session
if "ol" not in st.session_state:
    st.session_state["ol"] = mtsOverlay('mts.bit')

ol = st.session_state["ol"]
oled = st.session_state["oled"]

# Convenience aliases
DAC_SR = 4.0e9  # Hz
ADC_SR = 4.0e9  # Hz
DAC_AMP = 16383.0  # 14-bit +16383/-16384 range
buf_len = ol.dac_player.shape[0]
X_axis = (1.0 / DAC_SR) * np.arange(0, buf_len)  # seconds

waveform_options = ["serrodyne", "sine", "square", "custom"]


with st.sidebar:
    waveform_type = st.selectbox("Select waveform type", waveform_options)


    match waveform_type:
        case "sine" | "square":
            freq = st.number_input("Frequency (MHz)", min_value=1.0, max_value=2000.0, value=250.0, step=1.0) * 1e6
            build = st.button("Build waveform", type="primary")

        case "serrodyne":
            st.header("Waveform Parameters")
            ratios_str = st.text_input("Ratios", value="1:5:3", help="Segment ratios, e.g. 1:5:3")
            freqs_str = st.text_input("Frequencies (MHz)", value="-1330, 0, 840", help="One per ratio; 0 → flat segment")
            T_total_us = st.number_input("Total time (µs)", min_value=0.001, value=1.0, step=0.1)
            amp = st.slider("Amplitude (LSB)", min_value=1000, max_value=16383, value=16383)
            st.info("Sample rate: 4 GS/s")
            build = st.button("Build waveform", type="primary")

        case "custom":
            uploaded_file = st.file_uploader("Upload custom waveform")
            build = st.button("Build waveform", type="primary")

        case _:
            st.error("Something bad happened, so we go with the default.")
            req = st.number_input("Frequency (MHz)", min_value=1.0, max_value=2000.0, value=250.0, step=1.0)
            build = st.button("Build waveform", type="primary")
            


def read_adc():
    #perform loopback and capture signal
    ol.init_tile_sync()
    nonAlignedCaptureSamples = np.zeros((3,len(ol.adc_capture_chA)),dtype=np.int16)
    ol.verify_clock_tree()
    ol.internal_capture(nonAlignedCaptureSamples)
    # nonAlignedCaptureSamples[0][:]=-nonAlignedCaptureSamples[0][:]
    return -nonAlignedCaptureSamples[0][:]




if build:
    match waveform_type:
        case "sine":
            signal = DAC_AMP * np.sin(2 * np.pi * freq * X_axis)
        case "square":
            signal = DAC_AMP * sawtooth(2 * np.pi * freq * X_axis)
        case "serrodyne":
            T_s = T_total_us * 1e-6 # convert us to s
            sr_hz = 4 * 1e9 # convert GS/s to S/s
            ratios = parse_ratios(ratios_str)
            freqs_hz = parse_freqs_mhz(freqs_str)
            max_points = 65536
            x, y, N = make_piecewise_ramp(
                ratios, freqs_hz, T_s, amp=amp, sr_hz=sr_hz,
                max_points=int(max_points),
                continuous_phase=False
            )

            # digital signal has num_points samples, pad to buffer length, repeat
            if len(y) < buf_len:
                reps = int(np.ceil(buf_len / len(y)))
                signal = np.tile(y, reps)[:buf_len]
            else:
                signal = y
        case _:
            signal = DAC_AMP * np.sin(2 * np.pi * freq * X_axis)

    ol.dac_player[:] = np.int16(signal)

    # Plot time series
    fig_time = plot_time_series_interactive(
        X_axis,
        signal[0:4000],
        title="Generated Waveform",
        x_label="Time (s)",
        y_label="Amplitude (LSB)",
        height=450,
    )
    st.plotly_chart(fig_time, use_container_width=True)

    captured = read_adc()

    # Apply np.exp with chosen phase scale
    phase_scale = 2*np.pi / (np.max(captured)-np.min(captured))
    phi = phase_scale * captured
    cos_fz = np.exp(1j * phi)
    fz, mz, _, _ = calculate_fft(cos_fz, ADC_SR)
    fig_fft = plot_fft(fz, mz, title="Spectrum", y_label="Magnitude", name="|FFT{exp}|")
    st.plotly_chart(fig_fft, use_container_width=True)
