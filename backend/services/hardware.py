import threading
from typing import Optional

import numpy as np

from firmware.device import DeviceCore, device_is_loaded, get_device

DAC_AMP = int(np.iinfo(np.int16).max)
DAC_CHANNELS = ("dac0", "dac2")


class HardwareService:
    """What the web backend needs from the board.

    Everything goes through the process-wide device core (firmware/device.py), so the web
    page, the gRPC device server and anything else in this process see one state.
    """

    def __init__(self, device: Optional[DeviceCore] = None):
        self._device = device                 # tests pass one in; on the board it is created lazily
        self.last_signal: Optional[np.ndarray] = None
        # one capture memory, shared by the live view and any other capture
        self._adc_lock = threading.Lock()

    def initialize(self) -> None:
        _ = self.device

    @property
    def initialized(self) -> bool:
        return self._device is not None or device_is_loaded()

    @property
    def device(self) -> DeviceCore:
        if self._device is None:
            self._device = get_device()
        return self._device

    @property
    def controller(self):
        return self.device.controller

    def info(self) -> dict:
        return self.controller.info()

    @property
    def dac_sr(self) -> float:
        value = self.info()["rfdc"].get("dac0_sampling_rate_gsps")
        if value is None:
            raise ValueError("Overlay info does not include DAC0 sampling rate")
        return float(value) * 1.0e9

    @property
    def buf_len(self) -> int:
        return int(self.info()["dac0"]["bram_int16_samples"])

    @property
    def x_axis(self) -> np.ndarray:
        return np.arange(self.buf_len, dtype=float) / self.dac_sr

    # ------------------------------------------------------------------ bench test (loop mode)
    def load_dac(self, signal: np.ndarray, channel: str = "dac0") -> None:
        """Load a looping test waveform; the output stays off until enabled."""
        data = np.round(np.clip(np.asarray(signal, dtype=float), -DAC_AMP, DAC_AMP)).astype(np.int16)
        self.device.load_loop(channel, data)
        self.last_signal = data

    def set_dac_enabled(self, channel: str, enabled: bool) -> None:
        if enabled:
            self.device.arm(channel, trigger="immediate")
        else:
            self.device.stop(channel)

    def dac_status(self, channel: str) -> dict:
        status = next(o for o in self.device.status()["outputs"] if o["channel"] == channel)
        return {
            "enabled": status["state"] in ("running", "holding"),
            "waveform_length": status["loaded_samples"],
        }

    def dacs_status(self) -> dict:
        return {channel: self.dac_status(channel) for channel in DAC_CHANNELS}

    # ------------------------------------------------------------------ ADC
    @property
    def adc(self):
        """ADC capture of the overlay, or None when the bitstream has no ADC path."""
        return getattr(self.controller, "adc", None)

    def capture_adc(self, adc: str, samples: int) -> np.ndarray:
        """Capture now from ADC input "A".."D". Safe to call from several threads."""
        capture = self.adc
        if capture is None:
            raise RuntimeError("this bitstream has no ADC capture (build with RFSOC_AWG_ADC=1)")
        with self._adc_lock:
            return capture.capture(adc, samples)

    @property
    def analog_trigger(self):
        """Analog trigger of the overlay, or None when the bitstream has no ADC path."""
        return getattr(self.controller, "analog_trigger", None)

    def capture_adc_triggered(self, adc: str, samples: int, auto_timeout: float = 0.05):
        """Capture on the analog trigger, or after ``auto_timeout`` without one."""
        capture = self.adc
        if capture is None:
            raise RuntimeError("this bitstream has no ADC capture (build with RFSOC_AWG_ADC=1)")
        with self._adc_lock:
            return capture.capture_triggered(adc, samples, trigger="analog", auto_timeout=auto_timeout)

    def analog_trigger_info(self) -> Optional[dict]:
        trigger = self.analog_trigger
        return None if trigger is None else trigger.info()


hardware_service = HardwareService()


__all__ = [
    "HardwareService",
    "hardware_service",
    "DAC_AMP",
    "DAC_CHANNELS",
]
