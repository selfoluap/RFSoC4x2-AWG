import asyncio
import numpy as np
from typing import Tuple


# Public constants
DAC_SR = 4.0e9  # Hz
ADC_SR = 4.0e9  # Hz
DAC_AMP = 16383.0  # LSB range for 14-bit


def _ensure_event_loop_for_streamlit_thread():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


# Try to import hardware libs; if unavailable, we will provide a software stub
_HAS_HW = True
try:
    from rfsoc4x2 import oled  # type: ignore
    from rfsoc_mts import mtsOverlay  # type: ignore
except Exception:
    _HAS_HW = False
    oled = None  # type: ignore
    mtsOverlay = None  # type: ignore


class OverlayController:
    """Encapsulates RFSoC overlay access with a software-only fallback.

    API methods: read_adc, write_dac, output_waveform, calculate_transfer_function,
    calculate_total_error, and properties buf_len, time_axis.
    """

    def __init__(self, bitfile: str = "mts.bit"):
        _ensure_event_loop_for_streamlit_thread()

        if _HAS_HW:
            # Real hardware-backed implementation
            self.ol = mtsOverlay(bitfile)  # type: ignore
            self.oled = oled.oled_display() if oled else None  # type: ignore
            self.buf_len = self.ol.dac_player.shape[0]
            self.time_axis = (1.0 / DAC_SR) * np.arange(0, self.buf_len)
            self._mode = "hw"
        else:
            # Software stub: no hardware required
            self.ol = None
            self.oled = None
            self.buf_len = 65536
            self.time_axis = (1.0 / DAC_SR) * np.arange(0, self.buf_len)
            self._dac = np.zeros(self.buf_len, dtype=np.int16)
            self._mode = "stub"

    def read_adc(self) -> np.ndarray:
        if self._mode == "hw":
            self.ol.init_tile_sync()
            nonAlignedCaptureSamples = np.zeros((3, len(self.ol.adc_capture_chA)), dtype=np.int16)
            self.ol.verify_clock_tree()
            self.ol.internal_capture(nonAlignedCaptureSamples)
            return -nonAlignedCaptureSamples[0][:]
        # stub: return DAC with small noise as captured signal
        noise = np.random.normal(scale=50.0, size=self._dac.size).astype(np.int16)
        return (self._dac.astype(np.int32) + noise.astype(np.int32)).clip(-32768, 32767).astype(np.int16)

    def write_dac(self, signal: np.ndarray):
        if self._mode == "hw":
            self.ol.dac_player[:] = np.int16(signal)
        else:
            self._dac[:] = np.int16(signal[: self.buf_len])

    @staticmethod
    def calculate_transfer_function(input_signal: np.ndarray, output_signal: np.ndarray) -> np.ndarray:
        X = np.fft.fft(input_signal)
        Z = np.fft.fft(output_signal)
        H_est = Z / (X)
        H_inv = 1 / (H_est + 1e-8)
        X_corr = X * H_inv
        new_signal = np.real(np.fft.ifft(X_corr))
        new_signal = new_signal / np.max(np.abs(new_signal)) * np.max(np.abs(input_signal))
        return new_signal

    @staticmethod
    def calculate_total_error(
        original_signal: np.ndarray,
        captured_signal: np.ndarray,
        captured_after_precorrection: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        y_dc = original_signal - np.mean(original_signal)
        captured_dc = captured_signal - np.mean(captured_signal)
        captured_new_dc = captured_after_precorrection - np.mean(captured_after_precorrection)

        N = len(y_dc)
        X = np.fft.fft(y_dc) / N
        Z = np.fft.fft(captured_dc) / N
        Zp = np.fft.fft(captured_new_dc) / N

        E = np.sum(np.abs(Z - X) ** 2)
        E_prime = np.sum(np.abs(Zp - X) ** 2)

        norm = np.sum(np.abs(X) ** 2)
        E_norm = E / norm
        E_prime_norm = E_prime / norm
        return E, E_prime, E_norm, E_prime_norm

    def output_waveform(self, signal: np.ndarray, precorrection: bool = False):
        if precorrection:
            self.write_dac(signal)
            captured_before = self.read_adc()
            new_signal = self.calculate_transfer_function(signal, captured_before)
            self.write_dac(new_signal)
            captured_after = self.read_adc()
            return captured_before, captured_after

        self.write_dac(signal)
        captured = self.read_adc()
        return captured, None
