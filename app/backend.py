"""
FastAPI Backend for RFSoC AWG Control

This module provides REST API endpoints for hardware control and signal processing.
All functions return literals or dict objects - no Streamlit dependencies.
"""

import os
import asyncio
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any
from contextlib import asynccontextmanager

import numpy as np
from scipy.signal import sawtooth, square
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from signal_utils import calculate_fft, parse_freqs_mhz, parse_ratios
from signal_generator import calculate_serrodyne


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

OFFLINE_MODE = os.environ.get("RFSOC_OFFLINE", "0").lower() in ("1", "true", "yes")

# Hardware constants
DAC_SR = 4.0e9  # Hz
ADC_SR = 4.0e9  # Hz
DAC_AMP = 16383.0  # 14-bit +16383/-16384 range
BUF_LEN = 65536


# ─────────────────────────────────────────────────────────────────────────────
# Mock Classes for Offline Mode
# ─────────────────────────────────────────────────────────────────────────────

class MockOLED:
    """Mock OLED display for offline development."""
    def __init__(self):
        pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class MockOverlay:
    """Mock overlay for offline development."""
    def __init__(self, bitfile: Optional[str] = None):
        self._buf_len = BUF_LEN
        self._dac_player = np.zeros(self._buf_len, dtype=np.int16)
        self._adc_capture = np.zeros(self._buf_len, dtype=np.int16)

    @property
    def dac_player(self):
        return self._dac_player

    @property
    def adc_capture_chA(self):
        return self._adc_capture

    def init_tile_sync(self):
        pass

    def verify_clock_tree(self):
        pass

    def sync_tiles(self):
        pass

    def internal_capture(self, buffer):
        # Simulate captured signal: return the DAC signal with some noise
        noise = np.random.normal(0, 100, len(self._dac_player)).astype(np.int16)
        buffer[0][:] = self._dac_player + noise


# ─────────────────────────────────────────────────────────────────────────────
# Hardware State Manager (replaces st.session_state)
# ─────────────────────────────────────────────────────────────────────────────

class HardwareState:
    """
    Singleton class to manage hardware state.
    Replaces st.session_state for server-side state management.
    """
    _instance: Optional["HardwareState"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._overlay = None
        self._oled = None
        self._last_signal: Optional[np.ndarray] = None
        self._last_captured: Optional[np.ndarray] = None

    def initialize(self):
        """Initialize hardware components."""
        if OFFLINE_MODE:
            self._overlay = MockOverlay()
            self._oled = MockOLED()
        else:
            from rfsoc4x2 import oled
            from rfsoc_mts import mtsOverlay
            self._oled = oled.oled_display()
            self._overlay = mtsOverlay('/home/xilinx/paulo//RFSoC4x2-AWG/overlays/mts_8GS/mts_8GS.bit')

    @property
    def overlay(self):
        if self._overlay is None:
            self.initialize()
        return self._overlay

    @property
    def oled(self):
        if self._oled is None:
            self.initialize()
        return self._oled

    @property
    def buf_len(self) -> int:
        return self.overlay.dac_player.shape[0]

    @property
    def x_axis(self) -> np.ndarray:
        return (1.0 / DAC_SR) * np.arange(0, self.buf_len)

    @property
    def last_signal(self) -> Optional[np.ndarray]:
        return self._last_signal

    @last_signal.setter
    def last_signal(self, value: np.ndarray):
        self._last_signal = value

    @property
    def last_captured(self) -> Optional[np.ndarray]:
        return self._last_captured

    @last_captured.setter
    def last_captured(self, value: np.ndarray):
        self._last_captured = value


# Global hardware state instance
hardware = HardwareState()


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models for API
# ─────────────────────────────────────────────────────────────────────────────

class SerrodyneParams(BaseModel):
    ratios_str: str = "1:5:3"
    freqs_str: str = "-1330, 0, 840"
    T_total_us: float = 1.0
    amp: int = 16383


class SimpleWaveformParams(BaseModel):
    waveform_type: str  # "sine", "sawtooth", "square"
    freq_mhz: float = 250.0
    amp: int = 16383
    duty_cycle: float = 0.5  # Only used for square wave


class WaveformResponse(BaseModel):
    success: bool
    message: str
    signal: Optional[List[float]] = None
    captured: Optional[List[float]] = None
    captured_after_precorrection: Optional[List[float]] = None
    x_axis: Optional[List[float]] = None
    num_samples: Optional[int] = None


class FFTResponse(BaseModel):
    frequencies: List[float]
    magnitudes: List[float]


class ErrorMetrics(BaseModel):
    E: float
    E_prime: float
    E_norm: float
    E_prime_norm: float


class StatusResponse(BaseModel):
    offline_mode: bool
    hardware_initialized: bool
    buf_len: int
    dac_sr: float
    adc_sr: float


# ─────────────────────────────────────────────────────────────────────────────
# Core Signal Processing Functions (no Streamlit dependencies)
# ─────────────────────────────────────────────────────────────────────────────

def read_adc() -> np.ndarray:
    """Perform loopback and capture signal from ADC."""
    ol = hardware.overlay
    ol.init_tile_sync()
    non_aligned_capture_samples = np.zeros((3, len(ol.adc_capture_chA)), dtype=np.int16)
    ol.verify_clock_tree()
    ol.internal_capture(non_aligned_capture_samples)
    return -non_aligned_capture_samples[0][:]


def calculate_transfer_function(input_signal: np.ndarray, output_signal: np.ndarray) -> np.ndarray:
    """Calculate transfer function and return corrected signal."""
    X = np.fft.fft(input_signal)
    Z = np.fft.fft(output_signal)

    # Estimate frequency response: H(f) = Z / X
    H_est = Z / X

    # Regularized inverse filter
    H_inv = 1 / (H_est + 1e-8)

    # Apply inverse in frequency domain
    X_corr = X * H_inv

    # Back to time domain (real signal)
    new_signal = np.real(np.fft.ifft(X_corr))

    # Normalize amplitude to original signal range
    new_signal = new_signal / np.max(np.abs(new_signal)) * np.max(np.abs(input_signal))

    return new_signal


def calculate_total_error(
    original_signal: np.ndarray,
    captured_signal: np.ndarray,
    captured_after_precorrection: np.ndarray
) -> Dict[str, float]:
    """
    Calculate total frequency-domain error between ideal and measured signals.
    Returns dict with E, E_prime, E_norm, E_prime_norm.
    """
    # Remove DC offsets
    y_dc = original_signal - np.mean(original_signal)
    captured_dc = captured_signal - np.mean(captured_signal)
    captured_new_dc = captured_after_precorrection - np.mean(captured_after_precorrection)

    # Compute FFTs (normalize by N for energy consistency)
    N = len(y_dc)
    X = np.fft.fft(y_dc) / N
    Z = np.fft.fft(captured_dc) / N
    Zp = np.fft.fft(captured_new_dc) / N

    # Compute total frequency-domain error
    E = float(np.sum(np.abs(Z - X) ** 2))
    E_prime = float(np.sum(np.abs(Zp - X) ** 2))

    # Normalized errors (dimensionless)
    norm = np.sum(np.abs(X) ** 2)
    E_norm = float(E / norm)
    E_prime_norm = float(E_prime / norm)

    return {
        "E": E,
        "E_prime": E_prime,
        "E_norm": E_norm,
        "E_prime_norm": E_prime_norm
    }


def output_waveform_to_dac(
    signal: np.ndarray,
    precorrection: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Output a waveform to the DAC and perform optional precorrection.

    Returns a tuple (captured_before, captured_after_or_None).
    - If precorrection is False, returns (captured, None).
    - If precorrection is True, returns (captured_before, captured_after).
    """
    ol = hardware.overlay

    if precorrection:
        # Write original signal and capture (pre-precorrection)
        ol.dac_player[:] = np.int16(signal)
        captured_before = read_adc()

        # Calculate and write corrected signal
        new_signal = calculate_transfer_function(signal, captured_before)
        ol.dac_player[:] = np.int16(new_signal)

        # Capture final (post-precorrection)
        captured_after = read_adc()
        return captured_before, captured_after

    # No precorrection: write and capture once
    ol.dac_player[:] = np.int16(signal)
    captured = read_adc()
    return captured, None


@lru_cache(maxsize=32)
def generate_serrodyne_cached(
    ratios_str: str,
    freqs_str: str,
    T_total_us: float,
    amp: int,
    buf_len: int
) -> Tuple[Tuple[float, ...], Tuple[float, ...], int]:
    """
    Generate serrodyne waveform with caching.
    Returns (x, y, N) as tuples for cacheability.
    """
    T_s = T_total_us * 1e-6
    sr_hz = 4e9
    ratios = parse_ratios(ratios_str)
    freqs_hz = parse_freqs_mhz(freqs_str)
    max_points = 65536

    x, y, N = calculate_serrodyne(
        ratios, freqs_hz, T_s, amp=amp, sr_hz=sr_hz,
        max_points=int(max_points),
        continuous_phase=False
    )

    # Pad/tile to buffer length
    if len(y) < buf_len:
        reps = int(np.ceil(buf_len / len(y)))
        y_padded = np.tile(y, reps)[:buf_len]
        x_padded = np.tile(x, reps)[:buf_len]
    else:
        y_padded = y[:buf_len]
        x_padded = x[:buf_len]

    return tuple(x_padded.tolist()), tuple(y_padded.tolist()), N


def generate_simple_waveform(
    waveform_type: str,
    freq_hz: float,
    amp: float,
    duty_cycle: float = 0.5
) -> np.ndarray:
    """Generate simple waveforms (sine, sawtooth, square)."""
    x_axis = hardware.x_axis

    if waveform_type == "sine":
        return amp * np.sin(2 * np.pi * freq_hz * x_axis)
    elif waveform_type == "sawtooth":
        return amp * sawtooth(2 * np.pi * freq_hz * x_axis)
    elif waveform_type == "square":
        return amp * np.sign(np.sin(2 * np.pi * freq_hz * x_axis))
    else:
        # Default to sine
        return amp * np.sin(2 * np.pi * freq_hz * x_axis)


def compute_fft_for_captured(captured: np.ndarray) -> Dict[str, List[float]]:
    """Compute FFT spectrum from captured signal using phase scaling."""
    phase_scale = 2 * np.pi / (np.max(captured) - np.min(captured))
    phi = phase_scale * captured
    cos_fz = np.exp(1j * phi)
    fz, mz, _, _ = calculate_fft(cos_fz, ADC_SR)
    return {
        "frequencies": fz.tolist(),
        "magnitudes": mz.tolist()
    }


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize hardware on startup."""
    # Ensure event loop for async operations
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Initialize hardware
    hardware.initialize()
    print(f"Hardware initialized. Offline mode: {OFFLINE_MODE}")
    yield
    # Cleanup on shutdown (if needed)


app = FastAPI(
    title="RFSoC AWG Backend",
    description="REST API for RFSoC Arbitrary Waveform Generator control",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/status", response_model=StatusResponse)
def get_status():
    """Get current hardware status."""
    return StatusResponse(
        offline_mode=OFFLINE_MODE,
        hardware_initialized=hardware._initialized,
        buf_len=hardware.buf_len,
        dac_sr=DAC_SR,
        adc_sr=ADC_SR
    )


@app.get("/capture", response_model=WaveformResponse)
def capture_adc():
    """Capture ADC samples without changing DAC output."""
    try:
        captured = read_adc()
        hardware.last_captured = captured

        # Return subset for plotting (first 4000 samples)
        return WaveformResponse(
            success=True,
            message="ADC capture successful",
            captured=captured[:4000].tolist(),
            x_axis=hardware.x_axis[:4000].tolist(),
            num_samples=len(captured)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/capture/fft", response_model=FFTResponse)
def get_capture_fft():
    """Get FFT of the last captured signal."""
    if hardware.last_captured is None:
        raise HTTPException(status_code=400, detail="No captured signal available. Run capture first.")

    result = compute_fft_for_captured(hardware.last_captured)
    return FFTResponse(**result)


@app.post("/waveform/serrodyne", response_model=WaveformResponse)
def generate_and_output_serrodyne(
    params: SerrodyneParams,
    precorrection: bool = False
):
    """Generate and output serrodyne waveform."""
    try:
        buf_len = hardware.buf_len
        _, y_tuple, N = generate_serrodyne_cached(
            params.ratios_str,
            params.freqs_str,
            params.T_total_us,
            params.amp,
            buf_len
        )
        signal = np.array(y_tuple)
        hardware.last_signal = signal

        # Output to DAC and capture
        captured, captured_after = output_waveform_to_dac(signal, precorrection)
        hardware.last_captured = captured

        response = WaveformResponse(
            success=True,
            message=f"Serrodyne waveform generated with {N} base samples",
            signal=signal[:4000].tolist(),
            captured=captured[:4000].tolist(),
            x_axis=hardware.x_axis[:4000].tolist(),
            num_samples=len(signal)
        )

        if captured_after is not None:
            response.captured_after_precorrection = captured_after[:4000].tolist()

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/waveform/simple", response_model=WaveformResponse)
def generate_and_output_simple(
    params: SimpleWaveformParams,
    precorrection: bool = False
):
    """Generate and output simple waveform (sine, sawtooth, square)."""
    try:
        freq_hz = params.freq_mhz * 1e6
        signal = generate_simple_waveform(
            params.waveform_type,
            freq_hz,
            params.amp,
            params.duty_cycle
        )
        hardware.last_signal = signal

        # Output to DAC and capture
        captured, captured_after = output_waveform_to_dac(signal, precorrection)
        hardware.last_captured = captured

        response = WaveformResponse(
            success=True,
            message=f"{params.waveform_type} waveform generated at {params.freq_mhz} MHz",
            signal=signal[:4000].tolist(),
            captured=captured[:4000].tolist(),
            x_axis=hardware.x_axis[:4000].tolist(),
            num_samples=len(signal)
        )

        if captured_after is not None:
            response.captured_after_precorrection = captured_after[:4000].tolist()

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/waveform/fft", response_model=FFTResponse)
def get_waveform_fft():
    """Get FFT of the last captured waveform."""
    if hardware.last_captured is None:
        raise HTTPException(status_code=400, detail="No captured signal available")

    result = compute_fft_for_captured(hardware.last_captured)
    return FFTResponse(**result)


@app.post("/error_metrics", response_model=ErrorMetrics)
def calculate_error_metrics():
    """Calculate error metrics for precorrection evaluation."""
    if hardware.last_signal is None:
        raise HTTPException(status_code=400, detail="No signal available")

    # Re-capture for comparison
    captured_before = read_adc()

    # Apply precorrection
    new_signal = calculate_transfer_function(hardware.last_signal, captured_before)
    hardware.overlay.dac_player[:] = np.int16(new_signal)
    captured_after = read_adc()

    metrics = calculate_total_error(
        hardware.last_signal,
        captured_before,
        captured_after
    )

    return ErrorMetrics(**metrics)


@app.get("/constants")
def get_constants():
    """Get hardware constants."""
    return {
        "DAC_SR": DAC_SR,
        "ADC_SR": ADC_SR,
        "DAC_AMP": DAC_AMP,
        "BUF_LEN": hardware.buf_len,
        "OFFLINE_MODE": OFFLINE_MODE
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    host = os.environ.get("RFSOC_BACKEND_HOST", "0.0.0.0")
    port = int(os.environ.get("RFSOC_BACKEND_PORT", "8001"))
    uvicorn.run(app, host=host, port=port)
