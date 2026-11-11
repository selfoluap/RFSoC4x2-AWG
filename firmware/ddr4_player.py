"""Long-waveform playback from the PL DDR4 (DDR4Streamer in the FPGA).

The BRAM player holds 131072 samples (13 us at 9.8304 GS/s). This player keeps the waveform in
the PL DDR4 instead and streams it through a short FIFO, so the length limit becomes the size
of the DDR4. The DDR4 cannot deliver 9.8304 GS/s of 16-bit samples, so the DAC input rate is
reduced and the RF data converter interpolates back up to 9.8304 GS/s:

    interpolation   input rate     usable bandwidth   seconds in 4 GiB (one channel)
         2x         4915.2 MS/s      ~1966 MHz              0.44
         4x         2457.6 MS/s       ~983 MHz              0.87
         8x         1228.8 MS/s       ~491 MHz              1.75
        16x          614.4 MS/s       ~246 MHz              3.50

(usable bandwidth = 80 % of the input Nyquist band, the pass band of the RFDC filters)

For a few hundred milliseconds at the full 9.8304 GS/s there are the full-rate modes
(``set_full_rate``). 16-bit samples at that rate are more than the DDR4 can deliver, so samples
are stored bit-packed and unpacked in the FPGA:

    stored bits   DDR4 read load   seconds in 4 GiB   note
        14          17.2 GB/s          0.250          every bit the 14-bit DAC converts
        12          14.7 GB/s          0.291          quantisation noise 12 dB higher
         8           9.8 GB/s          0.437          for square-ish / coarse waveforms

The DDR4 peaks at 19.2 GB/s, so only one channel can run a full-rate mode at a time, and
whether the 14-bit mode keeps up has to be measured: ``status()["buffer_low_water"]``.

Nothing here imports pynq, so the module can be unit tested off the board.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Union

import numpy as np

DAC_SAMPLE_RATE = 9.8304e9
WORD_BYTES = 64                  # one DDR4 AXI beat / FIFO word
SAMPLES_PER_WORD = WORD_BYTES // 2
PAGE_BYTES = 4096                # waveforms start on a 4 KiB boundary
FULL_RATE_WORDS = 16             # samples per 614.4 MHz RFDC beat at interpolation 1

#: RFDC interpolation factor -> mode[1:0] of DDR4Streamer
RATE_SEL = {2: 0, 4: 1, 8: 2, 16: 3}
#: stored bits per sample in the full-rate modes -> mode[1:0] of DDR4Streamer
FULL_RATE_SEL = {14: 0, 12: 1, 8: 2}
#: stored bits -> samples per group; waveforms are padded to whole groups (7, 3, 1 words)
GROUP_SAMPLES = {14: 256, 12: 128, 8: 64}
FIFO_DEPTH = 4096                # DDR4Streamer FIFO_AW = 12

# control word (AXI GPIO output)
CTRL_ENABLE = 1 << 0
CTRL_SW_TRIGGER = 1 << 1
CTRL_EXT_TRIG_EN = 1 << 2
CTRL_LOOP_FOREVER = 1 << 3
CTRL_RATE_SHIFT = 4
CTRL_RATE_MASK = 0b11 << CTRL_RATE_SHIFT
CTRL_SOURCE_DDR4 = 1 << 6        # DacSourceMux: 0 = BRAM player, 1 = DDR4 player
CTRL_FULL_RATE = 1 << 7          # bit-packed samples, RFDC interpolation 1x
CTRL_MODE_MASK = CTRL_RATE_MASK | CTRL_FULL_RATE
CTRL_LOOPS_SHIFT = 8
MAX_LOOPS = (1 << 24) - 1

# status word (AXI GPIO input)
STATUS_BITS = {
    "busy": 0,
    "ready": 1,
    "running": 2,
    "done": 3,
    "underflow": 4,
    "read_error": 5,
    "trig_early": 6,
    "armed": 7,
}
STATUS_LOW_WATER_SHIFT = 16
STATUS_LOW_WATER_MASK = 0xFFFF

MAX_WORDS = (1 << 28) - 1
ABORT_HOLD_S = 0.002             # DDR4Streamer needs `enable` low for >= 1 ms


class Ddr4PlayerError(RuntimeError):
    pass


@dataclass(frozen=True)
class Segment:
    """A waveform stored in the DDR4 region of one player."""

    base_page: int               # absolute DDR4 address / 4096
    num_words: int               # length in 64-byte words (tail is zero padded)
    num_samples: int             # samples the caller supplied
    bits: int = 16               # stored bits per sample (14, 12, 8: full-rate modes)

    @property
    def padded_samples(self) -> int:
        return self.num_words * WORD_BYTES * 8 // self.bits

    def duration(self, interpolation: int = 1) -> float:
        return self.padded_samples / input_rate(interpolation)


def input_rate(interpolation: int) -> float:
    """DAC input sample rate in samples/s for an RFDC interpolation factor (1 = full rate)."""
    if interpolation != 1 and interpolation not in RATE_SEL:
        raise ValueError(f"interpolation must be 1 or one of {sorted(RATE_SEL)}, got {interpolation}")
    return DAC_SAMPLE_RATE / interpolation


def usable_bandwidth(interpolation: int) -> float:
    """Upper edge in Hz of the usable band: Nyquist at full rate, otherwise the 80 % of the
    input Nyquist band that the RFDC interpolation filters pass cleanly."""
    if interpolation == 1:
        return input_rate(1) / 2
    return 0.8 * input_rate(interpolation) / 2


def pack_samples(samples: np.ndarray, bits: int) -> np.ndarray:
    """Pack int16 samples into a little-endian stream of ``bits``-wide samples (uint8 array).

    14-bit keeps the top 14 bits, which is all the 14-bit DAC converts. 12 and 8 bit round to
    nearest and saturate. The number of samples must be a multiple of 4 (14 bit) or 2 (12 bit).
    """
    if bits not in FULL_RATE_SEL:
        raise ValueError(f"bits must be one of {sorted(FULL_RATE_SEL)}, got {bits}")
    shift = 16 - bits
    wide = samples.astype(np.int32)
    if bits != 14:
        wide = np.minimum(wide + (1 << (shift - 1)), 32767)
    code = ((wide >> shift) & ((1 << bits) - 1)).astype(np.uint64)
    if bits == 8:
        return code.astype(np.uint8)
    per, nbytes, dtype = (4, 7, "<u8") if bits == 14 else (2, 3, "<u4")
    if code.size % per:
        raise ValueError(f"{bits}-bit packing needs a multiple of {per} samples")
    code = code.reshape(-1, per)
    value = np.zeros(code.shape[0], dtype=np.uint64)
    for i in range(per):
        value |= code[:, i] << np.uint64(bits * i)
    return value.astype(dtype).view(np.uint8).reshape(-1, int(dtype[2]))[:, :nbytes].reshape(-1)


Waveform = Union[np.ndarray, Iterable[np.ndarray]]


class Ddr4Player:
    """One DAC channel's DDR4 player.

    Parameters
    ----------
    memory:
        int16 view of the *whole* PL DDR4 (``np.ndarray`` backed by an mmap on the board).
    region_start, region_bytes:
        The part of the DDR4 this channel owns. Must be 4 KiB aligned.
    ctrl_channel, status_channel, base_channel, words_channel:
        AXI GPIO channels (objects with ``write(value, mask)`` / ``read()``).
    rfdc_block:
        The xrfdc DAC block of this channel, used to switch the interpolation factor.
        ``None`` leaves the RF data converter alone.
    """

    def __init__(
        self,
        name: str,
        memory: np.ndarray,
        region_start: int,
        region_bytes: int,
        ctrl_channel,
        status_channel,
        base_channel,
        words_channel,
        rfdc_block=None,
        sleep=time.sleep,
    ):
        if memory.dtype != np.int16 or memory.ndim != 1:
            raise ValueError("memory must be a 1D int16 array")
        if region_start % PAGE_BYTES or region_bytes % PAGE_BYTES or region_bytes <= 0:
            raise ValueError("region_start and region_bytes must be positive multiples of 4096")
        if region_start + region_bytes > memory.size * 2:
            raise ValueError("region does not fit in the DDR4")

        self.name = name
        self.memory = memory
        self._bytes = memory.view(np.uint8)
        self.sample_bits = 16
        self.region_start = region_start
        self.region_bytes = region_bytes
        self._ctrl_channel = ctrl_channel
        self._status_channel = status_channel
        self._base_channel = base_channel
        self._words_channel = words_channel
        self.rfdc_block = rfdc_block
        self._sleep = sleep
        self._ctrl = 0
        self.interpolation = 8
        self.segment: Segment | None = None
        self._write_ctrl(0)

    # ------------------------------------------------------------------ sizes
    @property
    def capacity_samples(self) -> int:
        """Samples that fit in this player's region in the current mode."""
        return self.region_bytes * 8 // self.sample_bits

    def max_duration(self) -> float:
        """Longest waveform in seconds in the current mode."""
        return self.capacity_samples / input_rate(self.interpolation)

    # ------------------------------------------------------------------ loading
    def load(self, waveform: Waveform, offset_bytes: int = 0) -> Segment:
        """Copy a waveform into this player's DDR4 region and return its Segment.

        ``waveform`` is an int16 array, or an iterable of int16 arrays that are written one
        after the other. Use the iterable form for long waveforms: 1.75 s is 2**31 samples
        (4 GiB), more than the processor's free RAM, so it has to be generated in pieces.
        """
        if offset_bytes % PAGE_BYTES or not 0 <= offset_bytes < self.region_bytes:
            raise ValueError("offset_bytes must be a multiple of 4096 inside the region")
        if self.is_busy():
            raise Ddr4PlayerError(f"{self.name}: stop() playback before loading")
        if self.sample_bits != 16:
            return self._load_packed(waveform, offset_bytes, self.sample_bits)

        chunks = [waveform] if isinstance(waveform, np.ndarray) else waveform
        first = (self.region_start + offset_bytes) // 2
        limit = (self.region_start + self.region_bytes) // 2
        position = first
        for chunk in chunks:
            data = np.asarray(chunk)
            if data.dtype != np.int16:
                raise TypeError(f"waveform chunks must be int16, got {data.dtype}")
            if data.ndim != 1:
                raise ValueError("waveform chunks must be 1D")
            if position + data.size > limit:
                raise ValueError(
                    f"waveform exceeds {self.name} DDR4 region of "
                    f"{(limit - first)} samples"
                )
            self.memory[position : position + data.size] = data
            position += data.size

        num_samples = position - first
        if num_samples == 0:
            raise ValueError("waveform is empty")
        pad = -num_samples % SAMPLES_PER_WORD
        if pad:
            self.memory[position : position + pad] = 0   # region is 4 KiB aligned, so this fits
        num_words = (num_samples + pad) // SAMPLES_PER_WORD
        if num_words > MAX_WORDS:
            raise ValueError("waveform is longer than the 28-bit word counter allows")

        self.segment = Segment(
            base_page=(self.region_start + offset_bytes) // PAGE_BYTES,
            num_words=num_words,
            num_samples=num_samples,
        )
        return self.segment

    def _load_packed(self, waveform: Waveform, offset_bytes: int, bits: int) -> Segment:
        """Full-rate modes: quantise to ``bits`` and store as one packed bit stream."""
        chunks = [waveform] if isinstance(waveform, np.ndarray) else waveform
        first = self.region_start + offset_bytes
        limit = self.region_start + self.region_bytes
        position = first
        num_samples = 0
        carry = np.zeros(0, dtype=np.int16)
        align = 4                                   # samples that end on a byte boundary

        def store(samples: np.ndarray) -> int:
            packed = pack_samples(samples, bits)
            if position + packed.size > limit:
                raise ValueError(
                    f"waveform exceeds {self.name} DDR4 region of "
                    f"{(limit - first) * 8 // bits} samples at {bits} bits"
                )
            self._bytes[position : position + packed.size] = packed
            return position + packed.size

        for chunk in chunks:
            data = np.asarray(chunk)
            if data.dtype != np.int16:
                raise TypeError(f"waveform chunks must be int16, got {data.dtype}")
            if data.ndim != 1:
                raise ValueError("waveform chunks must be 1D")
            num_samples += data.size
            data = np.concatenate((carry, data)) if carry.size else data
            whole = data.size - data.size % align
            position = store(data[:whole])
            carry = data[whole:].copy()

        if num_samples == 0:
            raise ValueError("waveform is empty")
        group = GROUP_SAMPLES[bits]
        tail = np.zeros(carry.size + (-num_samples % group), dtype=np.int16)
        tail[: carry.size] = carry
        position = store(tail)

        self.segment = Segment(
            base_page=first // PAGE_BYTES,
            num_words=(position - first) // WORD_BYTES,
            num_samples=num_samples,
            bits=bits,
        )
        return self.segment

    # ------------------------------------------------------------------ rate
    def set_interpolation(self, interpolation: int) -> None:
        """Select the DAC input rate and point the DAC at the DDR4 player.

        Switches the RF data converter block to ``interpolation`` and tells it how many samples
        of each 256-bit stream beat are valid. Re-run DAC multi-tile sync afterwards if the two
        channels have to stay aligned.
        """
        if interpolation not in RATE_SEL:
            raise ValueError(f"interpolation must be one of {sorted(RATE_SEL)}, got {interpolation}")
        if self.is_busy():
            raise Ddr4PlayerError(f"{self.name}: stop() playback before changing the rate")
        if self.rfdc_block is not None:
            self.rfdc_block.InterpolationFactor = interpolation
            self.rfdc_block.FabWrVldWords = FULL_RATE_WORDS // interpolation
        self.interpolation = interpolation
        self.sample_bits = 16
        self._ctrl = (self._ctrl & ~CTRL_MODE_MASK) | (RATE_SEL[interpolation] << CTRL_RATE_SHIFT)
        self._write_ctrl(self._ctrl | CTRL_SOURCE_DDR4)

    def set_full_rate(self, bits: int = 14) -> None:
        """Play at the full 9.8304 GS/s from bit-packed samples (14, 12 or 8 bits each).

        Only one DAC channel can use a full-rate mode at a time: two would ask the DDR4 for
        more than it can deliver. Waveforms must be loaded after this call, because the
        storage format depends on ``bits``.
        """
        if bits not in FULL_RATE_SEL:
            raise ValueError(f"bits must be one of {sorted(FULL_RATE_SEL)}, got {bits}")
        if self.is_busy():
            raise Ddr4PlayerError(f"{self.name}: stop() playback before changing the rate")
        if self.rfdc_block is not None:
            self.rfdc_block.InterpolationFactor = 1
            self.rfdc_block.FabWrVldWords = FULL_RATE_WORDS
        self.interpolation = 1
        self.sample_bits = bits
        self._ctrl = (self._ctrl & ~CTRL_MODE_MASK) | (FULL_RATE_SEL[bits] << CTRL_RATE_SHIFT)
        self._write_ctrl(self._ctrl | CTRL_FULL_RATE | CTRL_SOURCE_DDR4)

    def select_bram_player(self) -> None:
        """Hand the DAC channel back to the full-rate BRAM player."""
        self.stop()
        if self.rfdc_block is not None:
            self.rfdc_block.InterpolationFactor = 1
            self.rfdc_block.FabWrVldWords = FULL_RATE_WORDS
        self.interpolation = 1
        self.sample_bits = 16
        self._write_ctrl(self._ctrl & ~(CTRL_SOURCE_DDR4 | CTRL_MODE_MASK))

    # ------------------------------------------------------------------ playback
    def arm(
        self,
        segment: Segment | None = None,
        loops: int = 1,
        forever: bool = False,
        external_trigger: bool = False,
        timeout: float = 1.0,
    ) -> None:
        """Prefetch the start of the waveform; returns once the player waits for a trigger."""
        segment = segment or self.segment
        if segment is None:
            raise Ddr4PlayerError(f"{self.name}: load() a waveform first")
        if not 1 <= loops <= MAX_LOOPS:
            raise ValueError(f"loops must be between 1 and {MAX_LOOPS}")
        if not self._ctrl & CTRL_SOURCE_DDR4:
            raise Ddr4PlayerError(f"{self.name}: call set_interpolation() or set_full_rate() first")
        if segment.bits != self.sample_bits:
            raise Ddr4PlayerError(
                f"{self.name}: segment was stored with {segment.bits}-bit samples, the current "
                f"mode needs {self.sample_bits}; load() it again"
            )
        start = segment.base_page * PAGE_BYTES
        end = start + segment.num_words * WORD_BYTES
        if start < self.region_start or end > self.region_start + self.region_bytes:
            raise ValueError("segment lies outside this player's DDR4 region")

        self.stop()
        self._base_channel.write(segment.base_page, 0xFFFFFFFF)
        self._words_channel.write(segment.num_words, 0xFFFFFFFF)
        ctrl = self._ctrl & (CTRL_MODE_MASK | CTRL_SOURCE_DDR4)
        ctrl |= loops << CTRL_LOOPS_SHIFT
        if forever:
            ctrl |= CTRL_LOOP_FOREVER
        if external_trigger:
            ctrl |= CTRL_EXT_TRIG_EN
        self._write_ctrl(ctrl)                    # configuration settles before enable rises
        self._write_ctrl(ctrl | CTRL_ENABLE)
        self._wait("ready", True, timeout)

    def trigger(self) -> None:
        """Software trigger. Playback starts a fixed few clock cycles later."""
        status = self.status()
        if not status["ready"]:
            raise Ddr4PlayerError(f"{self.name}: not ready for a trigger ({status})")
        self._write_ctrl(self._ctrl | CTRL_SW_TRIGGER)
        self._write_ctrl(self._ctrl & ~CTRL_SW_TRIGGER)

    def play(self, segment: Segment | None = None, loops: int = 1, forever: bool = False) -> None:
        self.arm(segment, loops=loops, forever=forever)
        self.trigger()

    def stop(self) -> None:
        """Abort playback and wait until the DDR4 reader is idle."""
        self._write_ctrl(self._ctrl & ~(CTRL_ENABLE | CTRL_SW_TRIGGER))
        self._sleep(ABORT_HOLD_S)
        self._wait("busy", False, 1.0)

    def wait_done(self, timeout: float | None = None) -> dict:
        """Block until a one-shot / counted playback finishes; raises if samples were dropped."""
        if timeout is None:
            if self.segment is None:
                raise ValueError("timeout is required when no segment is loaded")
            loops = max(1, self._ctrl >> CTRL_LOOPS_SHIFT)
            timeout = 2.0 * self.segment.duration(self.interpolation) * loops + 1.0
        self._wait("done", True, timeout)
        status = self.status()
        if status["underflow"]:
            raise Ddr4PlayerError(
                f"{self.name}: the DDR4 could not keep up, the output had gaps (underflow)"
            )
        if status["read_error"]:
            raise Ddr4PlayerError(f"{self.name}: the DDR4 controller reported a read error")
        return status

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        raw = int(self._status_channel.read())
        result = {name: bool(raw >> bit & 1) for name, bit in STATUS_BITS.items()}
        # Lowest fill of the short buffer during playback, out of FIFO_DEPTH words. Close to
        # FIFO_DEPTH: the DDR4 keeps up easily. Falling towards 0: it does not. None: not measured.
        low_water = raw >> STATUS_LOW_WATER_SHIFT & STATUS_LOW_WATER_MASK
        result["buffer_low_water"] = low_water if low_water <= FIFO_DEPTH else None
        return result

    def is_busy(self) -> bool:
        return self.status()["busy"]

    def info(self) -> dict:
        return {
            "name": self.name,
            "region_start": self.region_start,
            "region_bytes": self.region_bytes,
            "capacity_samples": self.capacity_samples,
            "interpolation": self.interpolation,
            "sample_bits": self.sample_bits,
            "input_rate_sps": input_rate(self.interpolation),
            "usable_bandwidth_hz": usable_bandwidth(self.interpolation),
            "max_duration_s": self.max_duration(),
            "segment": self.segment,
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
            if status["read_error"]:
                raise Ddr4PlayerError(f"{self.name}: the DDR4 controller reported a read error")
            if time.monotonic() >= deadline:
                raise Ddr4PlayerError(
                    f"{self.name}: timed out waiting for {flag}={value} ({status})"
                )
            self._sleep(0.0005)


def probe_ddr4_bytes(memory: np.ndarray, claimed_bytes: int) -> int:
    """Return how much distinct DDR4 is really behind the address window.

    The RFSoC4x2 is documented with 4 GB of PL DDR4, while the vendor base design maps an
    8 GiB window. If the upper half aliases the lower half, a marker written at the start shows
    up again half way. The two probed locations are restored afterwards.
    """
    half = claimed_bytes // 2 // 2      # index of the half-way point, in int16 samples
    if half == 0 or half >= memory.size:
        return claimed_bytes
    saved = (int(memory[0]), int(memory[half]))
    try:
        memory[half] = 0x1234
        memory[0] = 0x5A5A
        aliased = int(memory[half]) == 0x5A5A
    finally:
        memory[half] = saved[1]
        memory[0] = saved[0]
    return claimed_bytes // 2 if aliased else claimed_bytes


def attach(overlay, ddr4_ip: str = "ddr4_0", split: tuple[float, float] = (0.5, 0.5)) -> dict:
    """Build the two DDR4 players of an overlay that contains the DDR4 design.

    ``split`` is the share of the DDR4 given to DAC0 and DAC2; use ``(1.0, 0.0)`` to give one
    channel everything, which is what the full-rate modes want. Returns ``{}`` when the bitstream has no PL DDR4, so the existing
    BRAM-only overlay keeps working.
    """
    if ddr4_ip not in getattr(overlay, "mem_dict", {}):
        return {}
    from pynq import MMIO   # only available on the board

    entry = overlay.mem_dict[ddr4_ip]
    mmio = MMIO(entry["phys_addr"], entry["addr_range"])
    memory = mmio.array[: mmio.length].view(np.int16)
    total = probe_ddr4_bytes(memory, entry["addr_range"])

    players = {}
    start = 0
    channels = (("dac0", "axi_gpio_ddr0", 0, 0), ("dac2", "axi_gpio_ddr2", 2, 0))
    for (name, gpio, tile, block), share in zip(channels, split):
        size = int(total * share) // PAGE_BYTES * PAGE_BYTES
        if size == 0:
            continue
        ctl = getattr(overlay, f"{gpio}_ctl")
        cfg = getattr(overlay, f"{gpio}_cfg")
        players[name] = Ddr4Player(
            name=name,
            memory=memory,
            region_start=start,
            region_bytes=size,
            ctrl_channel=ctl.channel1,
            status_channel=ctl.channel2,
            base_channel=cfg.channel1,
            words_channel=cfg.channel2,
            rfdc_block=overlay.xrfdc.dac_tiles[tile].blocks[block],
        )
        start += size
    return players
