"""Sequence mode of the on-chip-memory player (DACSeqStreamer in the FPGA).

The waveform memory holds ``n_blocks`` blocks of equal length. Each block is played ``repeats``
times, then the next one starts on a sample-exact boundary, at the full 9.8304 GS/s and on both
DAC channels at once:

    block 0 x repeats -> block 1 x repeats -> ... -> last block x repeats -> stop / hold / loop

Typical uses: a Doppler staircase (one pattern per frequency step), "waveform A then B", or a
plain looped waveform that starts on a trigger.

Nothing here imports pynq at module level, so it can be unit tested off the board.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

DAC_SAMPLE_RATE = 9.8304e9
SAMPLES_PER_ROW = 32             # one 512-bit memory row
MAX_BLOCKS = (1 << 16) - 1
MAX_REPEATS = (1 << 32) - 1

CTRL_ARM = 1 << 0
CTRL_SW_TRIGGER = 1 << 1
CTRL_EXT_TRIG_EN = 1 << 2
CTRL_END_SHIFT = 3
CTRL_SEQ_MODE = 1 << 5
END_MODES = {"stop": 0, "hold": 1, "loop": 2}

STATUS_BITS = {"armed": 0, "running": 1, "done": 2, "holding": 3, "stall": 4, "seq_mode": 5}
STATUS_BLOCK_SHIFT = 16
STATUS_BLOCK_MASK = 0x1FFF


class SequencerError(RuntimeError):
    pass


class SeqPlayer:
    """One DAC channel's sequence player.

    ``buffer`` is the int16 view of the channel's waveform memory; the GPIO channels are objects
    with ``write(value, mask)`` / ``read()``.
    """

    def __init__(self, name, buffer, ctrl_channel, status_channel, lens_channel, repeats_channel,
                 sleep=time.sleep):
        if buffer.dtype != np.int16 or buffer.ndim != 1 or buffer.size % SAMPLES_PER_ROW:
            raise ValueError("buffer must be a 1D int16 array of whole 32-sample rows")
        self.name = name
        self.buffer = buffer
        self._ctrl_channel = ctrl_channel
        self._status_channel = status_channel
        self._lens_channel = lens_channel
        self._repeats_channel = repeats_channel
        self._sleep = sleep
        self._ctrl = 0
        self.block_samples = 0
        self.n_blocks = 0
        self.repeats = 1
        self.end = "stop"
        self._write_ctrl(0)

    # ------------------------------------------------------------------ sizes
    @property
    def capacity(self) -> int:
        """Samples of waveform memory."""
        return self.buffer.size

    def duration(self) -> float:
        """Seconds for one pass through the armed sequence."""
        return self.n_blocks * self.repeats * self.block_samples / DAC_SAMPLE_RATE

    # ------------------------------------------------------------------ loading
    def load_blocks(self, blocks: Sequence[np.ndarray] | np.ndarray) -> None:
        """Store equally long int16 blocks one after the other.

        The block length must be a multiple of 32 samples: blocks repeat and follow each other
        without a gap, so they cannot be padded.
        """
        if self.status()["running"]:
            raise SequencerError(f"{self.name}: stop() before loading")
        data = np.asarray(blocks)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        if data.ndim != 2 or data.dtype != np.int16:
            raise TypeError("blocks must be int16 arrays of equal length")
        n_blocks, block_samples = data.shape
        if n_blocks == 0 or block_samples == 0:
            raise ValueError("no samples to load")
        if block_samples % SAMPLES_PER_ROW:
            raise ValueError(
                f"block length {block_samples} is not a multiple of {SAMPLES_PER_ROW} samples"
            )
        if n_blocks > MAX_BLOCKS:
            raise ValueError(f"at most {MAX_BLOCKS} blocks")
        if data.size > self.capacity:
            raise ValueError(
                f"{n_blocks} blocks of {block_samples} samples need {data.size} samples, "
                f"{self.name} holds {self.capacity}"
            )
        self.buffer[: data.size] = data.reshape(-1)
        self.block_samples = block_samples
        self.n_blocks = n_blocks

    # ------------------------------------------------------------------ playback
    def arm(self, repeats: int = 1, end: str = "stop", external_trigger: bool = False,
            timeout: float = 1.0) -> None:
        """Latch the sequence and wait for a trigger.

        ``end``: "stop" (output goes to mid-scale), "hold" (keep repeating the last block) or
        "loop" (start again at the first block).
        """
        if self.n_blocks == 0:
            raise SequencerError(f"{self.name}: load_blocks() first")
        if end not in END_MODES:
            raise ValueError(f"end must be one of {sorted(END_MODES)}, got {end!r}")
        if not 1 <= repeats <= MAX_REPEATS:
            raise ValueError(f"repeats must be between 1 and {MAX_REPEATS}")
        self.stop()
        self.repeats, self.end = repeats, end
        self._lens_channel.write(
            (self.n_blocks << 16) | (self.block_samples // SAMPLES_PER_ROW), 0xFFFFFFFF
        )
        self._repeats_channel.write(repeats, 0xFFFFFFFF)
        ctrl = CTRL_SEQ_MODE | (END_MODES[end] << CTRL_END_SHIFT)
        if external_trigger:
            ctrl |= CTRL_EXT_TRIG_EN
        self._write_ctrl(ctrl)                    # configuration settles before arm rises
        self._write_ctrl(ctrl | CTRL_ARM)
        self._wait("armed", True, timeout)

    def trigger(self) -> None:
        """Software trigger; playback starts a fixed few clock cycles later."""
        if not self.status()["armed"]:
            raise SequencerError(f"{self.name}: not armed ({self.status()})")
        self._write_ctrl(self._ctrl | CTRL_SW_TRIGGER)
        self._write_ctrl(self._ctrl & ~CTRL_SW_TRIGGER)

    def play(self, repeats: int = 1, end: str = "stop") -> None:
        self.arm(repeats=repeats, end=end)
        self.trigger()

    def play_loop(self, waveform: np.ndarray) -> None:
        """Loop one waveform forever, starting now (zero-padded to a multiple of 32 samples)."""
        data = np.asarray(waveform)
        if data.ndim != 1 or data.dtype != np.int16:
            raise TypeError("waveform must be a 1D int16 array")
        padded = np.zeros(-(-data.size // SAMPLES_PER_ROW) * SAMPLES_PER_ROW, dtype=np.int16)
        padded[: data.size] = data
        self.stop()
        self.load_blocks(padded)
        self.play(end="loop")

    def stop(self) -> None:
        """Disarm; the output returns to mid-scale within a few clock cycles."""
        self._write_ctrl(self._ctrl & CTRL_SEQ_MODE)
        self._sleep(0.001)

    def wait_done(self, timeout: float | None = None) -> dict:
        """Block until an end="stop" sequence has finished; raises if the stream stalled."""
        if self.end != "stop":
            raise SequencerError(f'{self.name}: end="{self.end}" sequences never finish')
        if timeout is None:
            timeout = 2.0 * self.duration() + 1.0
        self._wait("done", True, timeout)
        status = self.status()
        if status["stall"]:
            raise SequencerError(f"{self.name}: the output stream stalled during playback")
        return status

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        raw = int(self._status_channel.read())
        result = {name: bool(raw >> bit & 1) for name, bit in STATUS_BITS.items()}
        result["block"] = raw >> STATUS_BLOCK_SHIFT & STATUS_BLOCK_MASK
        return result

    def info(self) -> dict:
        return {
            "name": self.name,
            "capacity_samples": self.capacity,
            "block_samples": self.block_samples,
            "n_blocks": self.n_blocks,
            "repeats": self.repeats,
            "end": self.end,
            "duration_s": self.duration(),
            "status": self.status(),
        }

    # ------------------------------------------------------------------ helpers
    def _write_ctrl(self, value: int) -> None:
        self._ctrl = value & 0xFFFFFFFF
        self._ctrl_channel.write(self._ctrl, 0xFFFFFFFF)

    def _wait(self, flag: str, value: bool, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            status = self.status()
            if status[flag] == value:
                return
            if time.monotonic() >= deadline:
                raise SequencerError(f"{self.name}: timed out waiting for {flag}={value} ({status})")
            self._sleep(0.0005)


def attach(overlay) -> dict:
    """Build the sequence players of an overlay built with RFSOC_AWG_SEQ=1.

    Returns ``{}`` for a bitstream without the sequencer, so the existing overlay keeps working.
    """
    if "axi_gpio_seq0_ctl" not in getattr(overlay, "ip_dict", {}):
        return {}
    players = {}
    for name, gpio in (("dac0", "axi_gpio_seq0"), ("dac2", "axi_gpio_seq2")):
        legacy = getattr(overlay, name)           # DacPlayer: already maps this channel's memory
        ctl = getattr(overlay, f"{gpio}_ctl")
        cfg = getattr(overlay, f"{gpio}_cfg")
        players[name] = SeqPlayer(
            name=name,
            buffer=legacy.buffer,
            ctrl_channel=ctl.channel1,
            status_channel=ctl.channel2,
            lens_channel=cfg.channel1,
            repeats_channel=cfg.channel2,
        )
    return players
