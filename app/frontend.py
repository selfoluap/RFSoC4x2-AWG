"""
Streamlit Frontend for RFSoC AWG Control

This module provides the UI layer that communicates with the FastAPI backend
via REST API calls. No hardware interactions happen here.
"""

import os
from typing import Optional, Dict, Any, List

import requests
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from config import config
from tutorial import show_tutorial
from signal_utils import plot_time_series_interactive, plot_fft


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Backend API URL (can be overridden via environment variable)
from dotenv import load_dotenv
load_dotenv()

_port = os.environ.get("RFSOC_BACKEND_PORT", "8001")
BACKEND_URL = os.environ.get("RFSOC_BACKEND_URL", f"http://localhost:{_port}")

# Request timeout in seconds
REQUEST_TIMEOUT = 30


# ─────────────────────────────────────────────────────────────────────────────
# API Client Functions
# ─────────────────────────────────────────────────────────────────────────────

class APIClient:
    """Client for communicating with the FastAPI backend."""

    def __init__(self, base_url: str = BACKEND_URL):
        self.base_url = base_url.rstrip("/")

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle API response and raise appropriate errors."""
        if response.status_code == 200:
            return response.json()
        else:
            try:
                error_detail = response.json().get("detail", "Unknown error")
            except Exception:
                error_detail = response.text
            raise Exception(f"API Error ({response.status_code}): {error_detail}")

    def get_status(self) -> Dict[str, Any]:
        """Get backend status."""
        response = requests.get(
            f"{self.base_url}/status",
            timeout=REQUEST_TIMEOUT
        )
        return self._handle_response(response)

    def get_constants(self) -> Dict[str, Any]:
        """Get hardware constants from backend."""
        response = requests.get(
            f"{self.base_url}/constants",
            timeout=REQUEST_TIMEOUT
        )
        return self._handle_response(response)

    def capture_adc(self) -> Dict[str, Any]:
        """Capture ADC samples."""
        response = requests.get(
            f"{self.base_url}/capture",
            timeout=REQUEST_TIMEOUT
        )
        return self._handle_response(response)

    def get_capture_fft(self) -> Dict[str, Any]:
        """Get FFT of captured signal."""
        response = requests.get(
            f"{self.base_url}/capture/fft",
            timeout=REQUEST_TIMEOUT
        )
        return self._handle_response(response)

    def generate_serrodyne(
        self,
        ratios_str: str,
        freqs_str: str,
        T_total_us: float,
        amp: int,
        precorrection: bool = False
    ) -> Dict[str, Any]:
        """Generate and output serrodyne waveform."""
        payload = {
            "ratios_str": ratios_str,
            "freqs_str": freqs_str,
            "T_total_us": T_total_us,
            "amp": amp
        }
        response = requests.post(
            f"{self.base_url}/waveform/serrodyne",
            json=payload,
            params={"precorrection": precorrection},
            timeout=REQUEST_TIMEOUT
        )
        return self._handle_response(response)

    def generate_simple_waveform(
        self,
        waveform_type: str,
        freq_mhz: float,
        amp: int,
        duty_cycle: float = 0.5,
        precorrection: bool = False
    ) -> Dict[str, Any]:
        """Generate and output simple waveform."""
        payload = {
            "waveform_type": waveform_type,
            "freq_mhz": freq_mhz,
            "amp": amp,
            "duty_cycle": duty_cycle
        }
        response = requests.post(
            f"{self.base_url}/waveform/simple",
            json=payload,
            params={"precorrection": precorrection},
            timeout=REQUEST_TIMEOUT
        )
        return self._handle_response(response)

    def get_waveform_fft(self) -> Dict[str, Any]:
        """Get FFT of last captured waveform."""
        response = requests.get(
            f"{self.base_url}/waveform/fft",
            timeout=REQUEST_TIMEOUT
        )
        return self._handle_response(response)

    def calculate_error_metrics(self) -> Dict[str, Any]:
        """Calculate error metrics for precorrection."""
        response = requests.post(
            f"{self.base_url}/error_metrics",
            timeout=REQUEST_TIMEOUT
        )
        return self._handle_response(response)


# Initialize API client
api = APIClient()


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def check_backend_connection() -> bool:
    """Check if backend is reachable."""
    try:
        api.get_status()
        return True
    except Exception:
        return False


def display_waveform_results(
    result: Dict[str, Any],
    show_precorrection: bool = False
):
    """Display waveform generation results including plots."""
    x_axis = np.array(result["x_axis"])
    signal = np.array(result["signal"])
    captured = np.array(result["captured"])

    # Plot generated waveform
    fig_time = plot_time_series_interactive(
        x_axis,
        signal,
        title="Generated Waveform",
        x_label="Time (s)",
        y_label="Amplitude (LSB)",
        height=450,
    )
    st.plotly_chart(fig_time, use_container_width=True)

    # Plot captured waveform
    fig_captured = plot_time_series_interactive(
        x_axis,
        captured,
        title="Captured Waveform",
        x_label="Time (s)",
        y_label="Amplitude (LSB)",
        height=450,
    )
    st.plotly_chart(fig_captured, use_container_width=True)

    # Get and plot FFT
    try:
        fft_result = api.get_waveform_fft()
        frequencies = np.array(fft_result["frequencies"])
        magnitudes = np.array(fft_result["magnitudes"])
        fig_fft = plot_fft(
            frequencies,
            magnitudes,
            title="Spectrum",
            y_label="Magnitude",
            name="|FFT{exp}|"
        )
        st.plotly_chart(fig_fft, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute FFT: {e}")

    # Show precorrection results if applicable
    if show_precorrection and result.get("captured_after_precorrection"):
        captured_after = np.array(result["captured_after_precorrection"])

        st.subheader("Precorrection Results")
        fig_corrected = plot_time_series_interactive(
            x_axis,
            captured_after,
            title="Captured After Precorrection",
            x_label="Time (s)",
            y_label="Amplitude (LSB)",
            height=450,
        )
        st.plotly_chart(fig_corrected, use_container_width=True)

        # Calculate and display error metrics
        try:
            metrics = api.calculate_error_metrics()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Error Before (E)", f"{metrics['E']:.2f}")
                st.metric("Normalized E", f"{metrics['E_norm']:.6f}")
            with col2:
                st.metric("Error After (E')", f"{metrics['E_prime']:.2f}")
                st.metric("Normalized E'", f"{metrics['E_prime_norm']:.6f}")
        except Exception as e:
            st.warning(f"Could not calculate error metrics: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="LaserLab: RFSoC AWG", layout="wide")
st.title("LaserLab: RFSoC AWG")

# Show tutorial if not done
if not config.get("tutorial_done", False):
    show_tutorial()

# Check backend connection
backend_connected = check_backend_connection()

if not backend_connected:
    st.error(
        f"⚠️ Cannot connect to backend at {BACKEND_URL}. "
        "Please ensure the backend server is running:\n\n"
        "```bash\n"
        "cd app && uvicorn backend:app --reload --host 0.0.0.0 --port 8000\n"
        "```"
    )
    st.stop()

# Get status from backend
try:
    status = api.get_status()
    if status["offline_mode"]:
        st.warning("Running in OFFLINE mode - hardware interactions are simulated")
except Exception as e:
    st.error(f"Error getting status: {e}")

# Get constants
try:
    constants = api.get_constants()
    ADC_SR = constants["ADC_SR"]
except Exception:
    ADC_SR = 4.0e9  # fallback

# Waveform options
waveform_options = ["serrodyne", "sine", "sawtooth", "square", "custom"]

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    waveform_type = st.selectbox("Select waveform type", waveform_options)

    enable_precorrection = st.checkbox(
        "Enable precorrection",
        value=False,
        help="Apply DAC precorrection for serrodyne waveforms"
    )

    # Initialize variables
    build = False
    ratios_str = ""
    freqs_str = ""
    T_total_us = 1.0
    amp = 16383
    freq = 250.0
    duty_cycle = 0.5

    match waveform_type:
        case "serrodyne":
            st.header("Waveform Parameters")
            ratios_str = st.text_input(
                "Ratios",
                value="1:5:3",
                help="Segment ratios, e.g. 1:5:3"
            )
            freqs_str = st.text_input(
                "Frequencies (MHz)",
                value="-1330, 0, 840",
                help="One per ratio; 0 → flat segment"
            )
            T_total_us = st.number_input(
                "Total time (µs)",
                min_value=0.001,
                value=1.0,
                step=0.1
            )
            amp = st.slider(
                "Amplitude (LSB)",
                min_value=1000,
                max_value=16383,
                value=16383
            )
            st.info("Sample rate: 4 GS/s")
            build = st.button("Build waveform", type="primary")

        case "square":
            freq = st.number_input(
                "Frequency (MHz)",
                min_value=1.0,
                max_value=2000.0,
                value=250.0,
                step=1.0
            )
            amp = st.slider(
                "Amplitude (LSB)",
                min_value=1000,
                max_value=16383,
                value=16383
            )
            duty_cycle = st.slider(
                "Duty Cycle",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Fraction of period signal is high"
            )
            build = st.button("Build waveform", type="primary")

        case "custom":
            uploaded_file = st.file_uploader("Upload custom waveform")
            build = st.button("Build waveform", type="primary")
            if build and uploaded_file:
                st.warning("Custom waveform upload not yet implemented in backend")
                build = False

        case _:  # default (sine, sawtooth)
            freq = st.number_input(
                "Frequency (MHz)",
                min_value=1.0,
                max_value=2000.0,
                value=250.0,
                step=1.0
            )
            amp = st.slider(
                "Amplitude (LSB)",
                min_value=1000,
                max_value=16383,
                value=16383
            )
            build = st.button("Build waveform", type="primary")

    # Standalone ADC capture button
    capture_only = st.button(
        "Capture ADC samples",
        help="Capture ADC samples without changing DAC output"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Content Area
# ─────────────────────────────────────────────────────────────────────────────

# Handle capture-only button
if capture_only:
    with st.spinner("Capturing ADC samples..."):
        try:
            result = api.capture_adc()

            x_axis = np.array(result["x_axis"])
            captured = np.array(result["captured"])

            # Plot captured time series
            fig_captured = plot_time_series_interactive(
                x_axis,
                captured,
                title="Captured Waveform",
                x_label="Time (s)",
                y_label="Amplitude (LSB)",
                height=450,
            )
            st.plotly_chart(fig_captured, use_container_width=True)

            # Get and plot FFT
            fft_result = api.get_capture_fft()
            frequencies = np.array(fft_result["frequencies"])
            magnitudes = np.array(fft_result["magnitudes"])
            fig_fft = plot_fft(
                frequencies,
                magnitudes,
                title="Spectrum",
                y_label="Magnitude",
                name="|FFT{exp}|"
            )
            st.plotly_chart(fig_fft, use_container_width=True)

        except Exception as e:
            st.error(f"Capture failed: {e}")


# Handle build waveform button
if build:
    with st.spinner("Generating and outputting waveform..."):
        try:
            match waveform_type:
                case "serrodyne":
                    result = api.generate_serrodyne(
                        ratios_str=ratios_str,
                        freqs_str=freqs_str,
                        T_total_us=T_total_us,
                        amp=amp,
                        precorrection=enable_precorrection
                    )

                case "sine" | "sawtooth" | "square":
                    result = api.generate_simple_waveform(
                        waveform_type=waveform_type,
                        freq_mhz=freq,
                        amp=amp,
                        duty_cycle=duty_cycle,
                        precorrection=enable_precorrection
                    )

                case _:
                    st.error(f"Unsupported waveform type: {waveform_type}")
                    result = None

            if result and result.get("success"):
                st.success(result["message"])
                display_waveform_results(result, show_precorrection=enable_precorrection)
            elif result:
                st.error(f"Failed: {result.get('message', 'Unknown error')}")

        except Exception as e:
            st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.caption(f"Backend: {BACKEND_URL}")
