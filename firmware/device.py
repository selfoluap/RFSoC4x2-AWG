"""Device core: one object that owns the hardware and gives every output the same operations.

    load_loop / load_sequence / load_long  ->  arm  ->  (trigger)  ->  stop
    status() and info() for anyone who only watches

The three players behind it (on-chip loop, sequence mode, DDR4) differ in what they can play,
not in how they are operated, so callers pick a playback mode and otherwise use one interface.
The gRPC device server, the web backend and notebooks all go through this class; it holds a
lock, so they can share one board.

Nothing here knows about gRPC, FastAPI or PYNQ: the class takes an overlay controller object
(``firmware.OverlayController`` on the board, a stand-in in tests).
"""

from __future__ import annotations

import os
import threading
from typing import Iterable

import numpy as np

DAC_SAMPLE_RATE = 9.8304e9
CHANNELS = ("dac0", "dac2")
MODES = ("loop", "sequence", "long")
TRIGGERS = ("immediate", "software", "external")
END_MODES = ("stop", "hold", "loop")
FULL_RATE_BITS = (14, 12, 8)
INTERPOLATIONS = (2, 4, 8, 16)


class DeviceError(RuntimeError):
    """The request is valid but the device cannot do it in its current state or build."""


class DeviceCore:
    def __init__(self, controller):
        self.controller = controller
        self._lock = threading.RLock()
        self._mode = {channel: None for channel in CHANNELS}        # what is loaded
        self._loaded = {channel: 0 for channel in CHANNELS}         # samples loaded
        self._legacy_running = {channel: False for channel in CHANNELS}

    # ------------------------------------------------------------------ what the build offers
    @property
    def _seq(self) -> dict:
        return getattr(self.controller, "seq", None) or {}

    @property
    def _ddr4(self) -> dict:
        return getattr(self.controller, "ddr4", None) or {}

    def capabilities(self) -> dict:
        return {
            "loop": True,
            "sequence": bool(self._seq),
            "long": bool(self._ddr4),
            "external_trigger": bool(self._seq) or bool(self._ddr4),
            "adc": getattr(self.controller, "adc", None) is not None,
            "analog_trigger": getattr(self.controller, "analog_trigger", None) is not None,
        }

    def info(self) -> dict:
        with self._lock:
            channels = []
            for channel in CHANNELS:
                entry = {"channel": channel, "loop_samples": self._loop_capacity(channel)}
                if channel in self._seq:
                    entry["sequence_samples"] = self._seq[channel].capacity
                if channel in self._ddr4:
                    entry["long_bytes"] = self._ddr4[channel].region_bytes
                channels.append(entry)
            return {
                "dac_sample_rate_hz": DAC_SAMPLE_RATE,
                "capabilities": self.capabilities(),
                "channels": channels,
                "bitfile": str(getattr(self.controller, "bitfile_path", "")),
                "clock_source": str(getattr(self.controller, "rf_clock_source", "")),
            }

    def _loop_capacity(self, channel: str) -> int:
        if channel in self._seq:
            return self._seq[channel].capacity
        return getattr(self.controller, channel).capacity

    # ------------------------------------------------------------------ loading
    def load_loop(self, channel: str, samples) -> dict:
        """One waveform that repeats until stopped (zero-padded to a multiple of 32 samples)."""
        data = self._int16(samples, ndim=1)
        with self._lock:
            self._check_channel(channel)
            self._release(channel)
            if channel in self._seq:
                padded = np.zeros(-(-data.size // 32) * 32, dtype=np.int16)
                padded[: data.size] = data
                self._seq[channel].load_blocks(padded)
            else:
                getattr(self.controller, channel).load_waveform(data)
            return self._loaded_as(channel, "loop", data.size)

    def load_sequence(self, channel: str, blocks) -> dict:
        """Equally long blocks, each played ``repeats`` times in turn (see ``arm``)."""
        data = self._int16(blocks, ndim=2)
        with self._lock:
            self._check_channel(channel)
            if channel not in self._seq:
                raise DeviceError("this bitstream has no sequence mode (build with RFSOC_AWG_SEQ=1)")
            self._release(channel)
            self._seq[channel].load_blocks(data)
            return self._loaded_as(channel, "sequence", data.size)

    def load_long(self, channel: str, chunks, sample_bits: int | None = None,
                  interpolation: int | None = None) -> dict:
        """A long waveform into the PL DDR4.

        ``chunks`` is an int16 array or an iterable of int16 arrays. Give either ``sample_bits``
        (14, 12 or 8: full DAC rate, bit-packed, one channel at a time) or ``interpolation``
        (2, 4, 8 or 16: reduced input rate, longer).
        """
        if (sample_bits is None) == (interpolation is None):
            raise ValueError("give either sample_bits or interpolation")
        if sample_bits is not None and sample_bits not in FULL_RATE_BITS:
            raise ValueError(f"sample_bits must be one of {FULL_RATE_BITS}")
        if interpolation is not None and interpolation not in INTERPOLATIONS:
            raise ValueError(f"interpolation must be one of {INTERPOLATIONS}")
        with self._lock:
            self._check_channel(channel)
            if channel not in self._ddr4:
                raise DeviceError("this bitstream has no DDR4 player (build with RFSOC_AWG_DDR4=1)")
            if sample_bits is not None:
                for other, mode in self._mode.items():
                    if other != channel and mode == "long" and self._ddr4[other].interpolation == 1:
                        raise DeviceError(
                            f"{other} is already in a full-rate DDR4 mode; the DDR4 cannot feed "
                            "two channels at the full rate"
                        )
            self._release(channel)
            player = self._ddr4[channel]
            if sample_bits is not None:
                player.set_full_rate(sample_bits)
            else:
                player.set_interpolation(interpolation)
            segment = player.load(chunks)
            return self._loaded_as(channel, "long", segment.num_samples)

    # ------------------------------------------------------------------ playback
    def arm(self, channel: str, trigger: str = "software", repeats: int = 1, end: str = "stop") -> dict:
        """Prepare playback of what is loaded.

        ``trigger``: "immediate" (start now), "software" (start on ``trigger()``) or "external"
        (start on the trigger pin; ``trigger()`` also works).
        ``repeats``: per block in sequence mode, passes through the waveform in long mode.
        ``end``: "stop", "hold" (keep repeating the last block; sequence mode only) or "loop".
        A loop-mode waveform always repeats until stopped.
        """
        if trigger not in TRIGGERS:
            raise ValueError(f"trigger must be one of {TRIGGERS}")
        if end not in END_MODES:
            raise ValueError(f"end must be one of {END_MODES}")
        if int(repeats) < 1:
            raise ValueError("repeats must be at least 1")
        with self._lock:
            mode = self._require_loaded(channel)
            external = trigger == "external"
            if mode == "loop" and channel not in self._seq:
                if trigger != "immediate":
                    raise DeviceError("this bitstream can only start a loop immediately (no trigger input)")
                getattr(self.controller, channel).enable()
                self._legacy_running[channel] = True
            elif mode in ("loop", "sequence"):
                player = self._seq[channel]
                if mode == "loop":
                    player.arm(repeats=1, end="loop", external_trigger=external)
                else:
                    player.arm(repeats=int(repeats), end=end, external_trigger=external)
                if trigger == "immediate":
                    player.trigger()
            else:
                if end == "hold":
                    raise ValueError('end="hold" is only available in sequence mode')
                player = self._ddr4[channel]
                player.arm(loops=int(repeats), forever=(end == "loop"), external_trigger=external)
                if trigger == "immediate":
                    player.trigger()
            return self._channel_status(channel)

    def trigger(self, channel: str) -> dict:
        """Software trigger for an armed output."""
        with self._lock:
            mode = self._require_loaded(channel)
            if mode == "loop" and channel not in self._seq:
                raise DeviceError("this bitstream has no software trigger for loop mode")
            (self._ddr4 if mode == "long" else self._seq)[channel].trigger()
            return self._channel_status(channel)

    def stop(self, channel: str) -> dict:
        """Stop the output; what is loaded stays loaded and can be armed again."""
        with self._lock:
            self._check_channel(channel)
            self._halt(channel)
            return self._channel_status(channel)

    def stop_all(self) -> dict:
        with self._lock:
            for channel in CHANNELS:
                self._halt(channel)
            return self.status()

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        with self._lock:
            return {"outputs": [self._channel_status(channel) for channel in CHANNELS]}

    def _channel_status(self, channel: str) -> dict:
        mode = self._mode[channel]
        result = {
            "channel": channel,
            "mode": mode or "none",
            "state": "idle",
            "block": 0,
            "loaded_samples": self._loaded[channel],
            "warnings": [],
            "buffer_low_water": None,
        }
        if mode is None:
            return result
        if mode == "loop" and channel not in self._seq:
            result["state"] = "running" if self._legacy_running[channel] else "idle"
        elif mode in ("loop", "sequence"):
            raw = self._seq[channel].status()
            result["block"] = raw["block"]
            if raw["holding"]:
                result["state"] = "holding"
            elif raw["running"]:
                result["state"] = "running"
            elif raw["done"]:
                result["state"] = "done"
            elif raw["armed"]:
                result["state"] = "armed"
            if raw["stall"]:
                result["warnings"].append("output stream stalled during playback")
        else:
            raw = self._ddr4[channel].status()
            if raw["running"]:
                result["state"] = "running"
            elif raw["done"]:
                result["state"] = "done"
            elif raw["ready"]:
                result["state"] = "armed"
            if raw["underflow"]:
                result["warnings"].append("the DDR4 could not keep up: the output had gaps")
            if raw["read_error"]:
                result["warnings"].append("the DDR4 controller reported a read error")
            if raw["trig_early"]:
                result["warnings"].append("a trigger arrived before the output was ready")
            result["buffer_low_water"] = raw.get("buffer_low_water")
        return result

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _int16(samples, ndim: int) -> np.ndarray:
        data = np.asarray(samples)
        if data.dtype != np.int16:
            raise ValueError(f"samples must be int16, got {data.dtype}")
        if data.ndim != ndim or data.size == 0:
            raise ValueError(f"expected a non-empty {ndim}D array of samples")
        return data

    @staticmethod
    def _check_channel(channel: str) -> None:
        if channel not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}")

    def _require_loaded(self, channel: str) -> str:
        self._check_channel(channel)
        mode = self._mode[channel]
        if mode is None:
            raise DeviceError(f"nothing is loaded on {channel}")
        return mode

    def _halt(self, channel: str) -> None:
        """Stop whichever player may be driving the channel."""
        if channel in self._ddr4 and self._mode[channel] == "long":
            self._ddr4[channel].stop()
        if channel in self._seq:
            self._seq[channel].stop()
        else:
            getattr(self.controller, channel).disable()
        self._legacy_running[channel] = False

    def _release(self, channel: str) -> None:
        """Stop the channel and hand the DAC back to the on-chip player before loading."""
        self._halt(channel)
        if channel in self._ddr4 and self._mode[channel] == "long":
            self._ddr4[channel].select_bram_player()
        self._mode[channel] = None
        self._loaded[channel] = 0

    def _loaded_as(self, channel: str, mode: str, samples: int) -> dict:
        self._mode[channel] = mode
        self._loaded[channel] = int(samples)
        return self._channel_status(channel)


_device: DeviceCore | None = None
_device_lock = threading.Lock()


def get_device() -> DeviceCore:
    """The one DeviceCore of this process, created on first use.

    Loading the overlay programs the FPGA, so a process must do it once and share the result:
    the device server and the web backend both call this. The bitstream is the packaged one,
    or the file named by the RFSOC_BITFILE environment variable.
    """
    global _device
    with _device_lock:
        if _device is None:
            from firmware import OverlayController       # needs PYNQ: only on the board

            bitfile = os.environ.get("RFSOC_BITFILE")
            _device = DeviceCore(OverlayController(bitfile=bitfile) if bitfile else OverlayController())
        return _device


def device_is_loaded() -> bool:
    return _device is not None
