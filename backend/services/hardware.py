import os
from typing import Optional

import numpy as np

from firmware import OverlayController

DAC_AMP = int(np.iinfo(np.int16).max)
DEFAULT_BITFILE = os.environ.get("RFSOC_BITFILE")
DAC_CHANNELS = ("dac0", "dac2")


class HardwareService:
    """Shared hardware service wrapper around OverlayController."""

    def __init__(self, bitfile: Optional[str] = DEFAULT_BITFILE):
        self._bitfile = bitfile
        self._controller: Optional[OverlayController] = None
        self.last_signal: Optional[np.ndarray] = None

    def initialize(self) -> None:
        if self._controller is None:
            if self._bitfile:
                self._controller = OverlayController(bitfile=self._bitfile)
            else:
                self._controller = OverlayController()

    @property
    def initialized(self) -> bool:
        return self._controller is not None

    @property
    def controller(self) -> OverlayController:
        self.initialize()
        assert self._controller is not None
        return self._controller

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

    def _dac_player(self, channel: str):
        player = getattr(self.controller, channel, None)
        if channel not in DAC_CHANNELS or player is None:
            raise ValueError(f"Unsupported DAC channel: {channel}")
        return player

    def load_dac(self, signal: np.ndarray, channel: str = "dac0") -> None:
        player = self._dac_player(channel)
        data = np.round(np.clip(np.asarray(signal, dtype=float), -DAC_AMP, DAC_AMP)).astype(np.int16)
        player.load_waveform(data)
        self.last_signal = data

    def set_dac_enabled(self, channel: str, enabled: bool) -> None:
        player = self._dac_player(channel)
        if enabled:
            player.enable()
        else:
            player.disable()

    def dac_status(self, channel: str) -> dict:
        player = self._dac_player(channel)
        return {
            "enabled": player.is_enabled(),
            "waveform_length": player.waveform_length,
        }

    def dacs_status(self) -> dict:
        return {channel: self.dac_status(channel) for channel in DAC_CHANNELS}


hardware_service = HardwareService()


__all__ = [
    "HardwareService",
    "hardware_service",
    "DAC_AMP",
    "DAC_CHANNELS",
]
