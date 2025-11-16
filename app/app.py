# Imports and overlay loading
from scipy.signal import sawtooth, square
import numpy as np
from typing import List, Tuple, Dict
import asyncio
from awg import plot_time_series_interactive, plot_fft, make_piecewise_ramp, calculate_fft, parse_freqs_mhz, parse_ratios

#all of these type ignores are only to keep my IDE happy
import streamlit as st # type: ignore
import plotly.graph_objects as go # type: ignore

# RFSoC overlay and OLED display
from rfsoc4x2 import oled # type: ignore
from rfsoc_mts import mtsOverlay # type: ignore
#from rfsoc import RFSocAWG

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
    st.session_state["ol"] = mtsOverlay('mts_4GS.bit')

ol = st.session_state["ol"]
oled = st.session_state["oled"]

# Convenience aliases
DAC_SR = 4.0e9  # Hz
ADC_SR = 4.0e9  # Hz
DAC_AMP = 16383.0  # 14-bit +16383/-16384 range
buf_len = ol.dac_player.shape[0]
X_axis = (1.0 / DAC_SR) * np.arange(0, buf_len)  # seconds

waveform_options = ["serrodyne", "sine", "sawtooth", "square", "custom"]


with st.sidebar:
    waveform_type = st.selectbox("Select waveform type", waveform_options)

    enable_precorrection = st.checkbox("Enable precorrection", value=False, help="Apply DAC precorrection for serrodyne waveforms")
    match waveform_type:
        case "serrodyne":
            st.header("Waveform Parameters")
            ratios_str = st.text_input("Ratios", value="1:5:3", help="Segment ratios, e.g. 1:5:3")
            freqs_str = st.text_input("Frequencies (MHz)", value="-1330, 0, 840", help="One per ratio; 0 → flat segment")
            T_total_us = st.number_input("Total time (µs)", min_value=0.001, value=1.0, step=0.1)
            amp = st.slider("Amplitude (LSB)", min_value=1000, max_value=16383, value=16383)
            st.info("Sample rate: 4 GS/s")
            build = st.button("Build waveform", type="primary")

        case "square":
            freq = st.number_input("Frequency (MHz)", min_value=1.0, max_value=2000.0, value=250.0, step=1.0) * 1e6
            amp = st.slider("Amplitude (LSB)", min_value=1000, max_value=16383, value=16383)
            duty_cycle = st.slider("Duty Cycle", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                                help="Fraction of period signal is high")
            build = st.button("Build waveform", type="primary")

        case "custom":
            uploaded_file = st.file_uploader("Upload custom waveform")
            build = st.button("Build waveform", type="primary")

        case _: #default
            freq = st.number_input("Frequency (MHz)", min_value=1.0, max_value=2000.0, value=250.0, step=1.0) * 1e6
            amp = st.slider("Amplitude (LSB)", min_value=1000, max_value=16383, value=16383)
            build = st.button("Build waveform", type="primary")
            
    # Standalone ADC capture button (does not modify DAC output)
    capture_only = st.button("Capture ADC samples", help="Capture ADC samples without changing DAC output")


def read_adc():
    #perform loopback and capture signal
    ol.init_tile_sync()
    nonAlignedCaptureSamples = np.zeros((3,len(ol.adc_capture_chA)),dtype=np.int16)
    ol.verify_clock_tree()
    # ol.sync_tiles()
    ol.internal_capture(nonAlignedCaptureSamples)
    # nonAlignedCaptureSamples[0][:]=-nonAlignedCaptureSamples[0][:]
    return -nonAlignedCaptureSamples[0][:]


def calculate_transfer_function(input_signal, output_signal):
    X = np.fft.fft(input_signal)
    Z = np.fft.fft(output_signal)

    # Estimate frequency response: H(f) = Z / X
    # eps = 1e-6
    H_est = Z / (X)

    # Regularized inverse filter
    # H_inv = np.conj(H_est) / (np.abs(H_est)**2)
    H_inv = 1 / (H_est + 1e-8)

    # Apply inverse in frequency domain
    X_corr = X * H_inv

    # Back to time domain (real signal)
    new_signal = np.real(np.fft.ifft(X_corr))

    # Normalize amplitude to original signal range
    new_signal = new_signal / np.max(np.abs(new_signal)) * np.max(np.abs(input_signal))

    return new_signal

def calculate_total_error(original_signal, captured_signal, captured_after_precorrection):
    """
    Calculate total frequency-domain error between ideal (y)
    and measured signals (captured, captured_new).
    Returns both E and E' values.
    """
    # --- Remove DC offsets (important before FFT)
    y_dc = original_signal - np.mean(original_signal)
    captured_dc = captured_signal - np.mean(captured_signal)
    captured_new_dc = captured_after_precorrection - np.mean(captured_after_precorrection)

    # --- Compute FFTs (normalize by N for energy consistency)
    N = len(y_dc)
    X = np.fft.fft(y_dc) / N
    Z = np.fft.fft(captured_dc) / N
    Zp = np.fft.fft(captured_new_dc) / N

    # --- Compute total frequency-domain error
    E = np.sum(np.abs(Z - X)**2)
    E_prime = np.sum(np.abs(Zp - X)**2)

    # --- Optional: normalized errors (dimensionless)
    norm = np.sum(np.abs(X)**2)
    E_norm = E / norm
    E_prime_norm = E_prime / norm

    return E, E_prime, E_norm, E_prime_norm




def output_waveform(signal, precorrection: bool = False):
    """Output a waveform to the DAC and perform optional precorrection.

    Returns a tuple (captured_before, captured_after_or_None).
    - If precorrection is False, returns (captured, None).
    - If precorrection is True, returns (captured_before, captured_after).
    """
    if precorrection:
        # write original signal and capture (pre-precorrection)
        ol.dac_player[:] = np.int16(signal)
        captured_before = read_adc()

        # calculate and write corrected signal
        new_signal = calculate_transfer_function(signal, captured_before)
        ol.dac_player[:] = np.int16(new_signal)

        # capture final (post-precorrection)
        captured_after = read_adc()
        return captured_before, captured_after

    # no precorrection: write and capture once
    ol.dac_player[:] = np.int16(signal)
    captured = read_adc()
    return captured, None

if capture_only:
        # Plot captured time series
    captured = read_adc()
    fig_captured = plot_time_series_interactive(
        X_axis,
        captured[0:4000],
        title="Captured Waveform",
        x_label="Time (s)",
        y_label="Amplitude (LSB)",
        height=450,
    )
    st.plotly_chart(fig_captured, use_container_width=True)

    # Apply np.exp with chosen phase scale
    phase_scale = 2*np.pi / (np.max(captured)-np.min(captured))
    phi = phase_scale * captured
    cos_fz = np.exp(1j * phi)
    fz, mz, _, _ = calculate_fft(cos_fz, ADC_SR)
    fig_fft = plot_fft(fz, mz, title="Spectrum", y_label="Magnitude", name="|FFT{exp}|")
    st.plotly_chart(fig_fft, use_container_width=True)


if build:
    match waveform_type:
        case "sine":
            signal = DAC_AMP * np.sin(2 * np.pi * freq * X_axis)
        case "sawtooth":
            signal = amp * sawtooth(2 * np.pi * freq * X_axis)
        case "square":
            signal = DAC_AMP * np.sign(np.sin(2 * np.pi * freq * X_axis))
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

    # Play waveform and capture ADC samples in the correct order.
    # output_waveform now returns (captured_before, captured_after_or_None).
    captured, captured_after_precorrection = output_waveform(signal, precorrection=enable_precorrection)

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

    #captured = read_adc()

    # Plot captured time series
    fig_captured = plot_time_series_interactive(
        X_axis,
        captured[0:4000],
        title="Captured Waveform",
        x_label="Time (s)",
        y_label="Amplitude (LSB)",
        height=450,
    )
    st.plotly_chart(fig_captured, use_container_width=True)

    # Apply np.exp with chosen phase scale
    phase_scale = 2*np.pi / (np.max(captured)-np.min(captured))
    phi = phase_scale * captured
    cos_fz = np.exp(1j * phi)
    fz, mz, _, _ = calculate_fft(cos_fz, ADC_SR)
    fig_fft = plot_fft(fz, mz, title="Spectrum", y_label="Magnitude", name="|FFT{exp}|")
    st.plotly_chart(fig_fft, use_container_width=True)

    if enable_precorrection:
        # captured_after_precorrection should have been returned by output_waveform,
        # but if not, capture now.
        if captured_after_precorrection is None:
            captured_after_precorrection = read_adc()

        # Calculate and display total error before and after precorrection
        E, E_prime, E_norm, E_prime_norm = calculate_total_error(signal, captured, captured_after_precorrection)
        st.write(f"Total frequency-domain error before precorrection: E = {E:.2f}, normalized E_norm = {E_norm:.6f}")
        st.write(f"Total frequency-domain error after precorrection: E' = {E_prime:.2f},lasormalized E'_norm = {E_prime_norm:.6f}")
