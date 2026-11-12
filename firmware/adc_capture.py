"""ADC capture and spectrum (ADCCapture in the FPGA).

Records up to 262,144 consecutive samples (53 us) of one ADC input at 4915.2 MS/s into on-chip
memory, started by software, the external trigger pin, or the start of DAC playback. The ADC
rate is exactly half the DAC rate and comes from the same reference, so captures are coherent
with the output.

``spectrum()`` turns a capture into an averaged amplitude spectrum in dBFS.

Nothing here imports pynq at module level, so it can be unit tested off the board.
"""

from __future__ import annotations

import time

import numpy as np

ADC_SAMPLE_RATE = 4.9152e9
SAMPLES_PER_WORD = 16            # one 256-bit capture word
FULL_SCALE = 32768.0             # ADC samples are 14 bit, MSB-aligned in int16

#: Board connector label -> capture input. From the tile/block order used by other RFSoC 4x2
#: firmware (tile 0: D, C; tile 2: B, A). Confirm once with a cable.
ADC_INPUTS = {"D": 0, "C": 1, "B": 2, "A": 3}
TRIGGER_SOURCES = {"software": 0, "external": 1, "dac0": 2, "dac2": 3, "analog": 4}

CTRL_ARM = 1 << 0
CTRL_SW_TRIGGER = 1 << 1
CTRL_SRC_SHIFT = 2
CTRL_ADC_SHIFT = 5

STATUS_ARMED = 1 << 0
STATUS_BUSY = 1 << 1
STATUS_DONE = 1 << 2
STATUS_ROWS_SHIFT = 16
STATUS_ROWS_MASK = 0x7FFF


class CaptureError(RuntimeError):
    pass


class AdcCapture:
    """``buffer`` is the int16 view of the capture memory; channels have write(value, mask)/read()."""

    def __init__(self, buffer, ctrl_channel, status_channel, delay_channel, rows_channel,
                 sleep=time.sleep):
        if buffer.dtype != np.int16 or buffer.ndim != 1 or buffer.size % SAMPLES_PER_WORD:
            raise ValueError("buffer must be a 1D int16 array of whole 16-sample words")
        self.buffer = buffer
        self._ctrl_channel = ctrl_channel
        self._status_channel = status_channel
        self._delay_channel = delay_channel
        self._rows_channel = rows_channel
        self._sleep = sleep
        self._ctrl = 0
        self.samples = 0
        self._write_ctrl(0)

    @property
    def capacity(self) -> int:
        return self.buffer.size

    # ------------------------------------------------------------------ capture
    def arm(self, adc="A", samples=None, trigger="software", delay_s=0.0, timeout=1.0) -> None:
        """Prepare a capture.

        ``adc``: connector label "A".."D" or input index 0..3.
        ``samples``: rounded up to a multiple of 16; default is the whole memory.
        ``trigger``: "software" (call ``trigger()``), "external" (PMOD0 pin 0), "dac0" or "dac2"
        (start of that DAC's sequence playback), or "analog" (the analog trigger, see
        analog_trigger.py). ``trigger()`` works in every mode.
        ``delay_s``: time to skip between the trigger and the first sample, in steps of 3.26 ns.
        """
        index = ADC_INPUTS[adc.upper()] if isinstance(adc, str) else int(adc)
        if not 0 <= index <= 3:
            raise ValueError("adc must be 'A'..'D' or 0..3")
        if trigger not in TRIGGER_SOURCES:
            raise ValueError(f"trigger must be one of {sorted(TRIGGER_SOURCES)}")
        samples = self.capacity if samples is None else int(samples)
        if not 1 <= samples <= self.capacity:
            raise ValueError(f"samples must be between 1 and {self.capacity}")
        if delay_s < 0:
            raise ValueError("delay_s must not be negative")
        rows = -(-samples // SAMPLES_PER_WORD)
        delay_words = int(round(delay_s * ADC_SAMPLE_RATE / SAMPLES_PER_WORD))
        if delay_words > 0xFFFFFFFF:
            raise ValueError("delay_s is too long (maximum about 14 s)")

        self.disarm()
        self._rows_channel.write(rows, 0xFFFFFFFF)
        self._delay_channel.write(delay_words, 0xFFFFFFFF)
        ctrl = (index << CTRL_ADC_SHIFT) | (TRIGGER_SOURCES[trigger] << CTRL_SRC_SHIFT)
        self._write_ctrl(ctrl)                    # configuration settles before arm rises
        self._write_ctrl(ctrl | CTRL_ARM)
        self.samples = rows * SAMPLES_PER_WORD
        self._wait(STATUS_ARMED | STATUS_BUSY | STATUS_DONE, timeout, "to arm")

    def trigger(self) -> None:
        self._write_ctrl(self._ctrl | CTRL_SW_TRIGGER)
        self._write_ctrl(self._ctrl & ~CTRL_SW_TRIGGER)

    def disarm(self) -> None:
        self._write_ctrl(self._ctrl & ~(CTRL_ARM | CTRL_SW_TRIGGER))

    def read(self, timeout=2.0) -> np.ndarray:
        """Wait for the capture to finish and return a copy of the samples (int16)."""
        if self.samples == 0:
            raise CaptureError("arm() a capture first")
        self._wait(STATUS_DONE, timeout, "for the capture to finish (no trigger?)")
        return np.array(self.buffer[: self.samples], dtype=np.int16)

    def capture(self, adc="A", samples=None) -> np.ndarray:
        """Capture now and return the samples."""
        self.arm(adc, samples)
        self.trigger()
        return self.read()

    def capture_triggered(self, adc="A", samples=None, trigger="analog", auto_timeout=0.05):
        """Capture on a trigger, like an oscilloscope in "auto" mode.

        Waits up to ``auto_timeout`` seconds for the trigger; if none comes, captures anyway.
        Returns ``(samples, triggered)``.
        """
        self.arm(adc, samples, trigger=trigger)
        try:
            self._wait(STATUS_DONE, auto_timeout, "for a trigger")
            triggered = True
        except CaptureError:
            self.trigger()
            triggered = False
        return self.read(), triggered

    def status(self) -> dict:
        raw = int(self._status_channel.read())
        return {
            "armed": bool(raw & STATUS_ARMED),
            "busy": bool(raw & STATUS_BUSY),
            "done": bool(raw & STATUS_DONE),
            "samples": (raw >> STATUS_ROWS_SHIFT & STATUS_ROWS_MASK) * SAMPLES_PER_WORD,
        }

    # ------------------------------------------------------------------ helpers
    def _write_ctrl(self, value: int) -> None:
        self._ctrl = value & 0xFFFFFFFF
        self._ctrl_channel.write(self._ctrl, 0xFFFFFFFF)

    def _wait(self, mask: int, timeout: float, what: str) -> None:
        deadline = time.monotonic() + timeout
        while not int(self._status_channel.read()) & mask:
            if time.monotonic() >= deadline:
                raise CaptureError(f"timed out waiting {what} ({self.status()})")
            self._sleep(0.0005)


def spectrum(samples, sample_rate=ADC_SAMPLE_RATE, nfft=None, window="hann"):
    """Averaged amplitude spectrum of a capture.

    Returns ``(freqs_hz, level_dbfs)``. 0 dBFS is a sine that just reaches ADC full scale. The
    capture is cut into segments of ``nfft`` samples (default: one segment of the largest power
    of two that fits, at most 65536) whose power spectra are averaged, which lowers the noise
    on the trace without hiding spurs.

    The ADC sees DC to 2457.6 MHz. Anything above that folds back: a tone at f between 2457.6
    and 4915.2 MHz shows up at 4915.2 MHz - f.
    """
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim != 1 or data.size < 16:
        raise ValueError("samples must be a 1D array of at least 16 samples")
    if nfft is None:
        nfft = min(65536, 1 << (data.size.bit_length() - 1))
    nfft = int(nfft)
    if not 16 <= nfft <= data.size:
        raise ValueError("nfft must be between 16 and the number of samples")
    if window == "hann":
        w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nfft) / nfft)
    elif window in ("rect", None):
        w = np.ones(nfft)
    else:
        raise ValueError("window must be 'hann' or 'rect'")

    segments = data[: data.size // nfft * nfft].reshape(-1, nfft)
    segments = segments - segments.mean(axis=1, keepdims=True)
    amplitude = np.abs(np.fft.rfft(segments * w, axis=1)) * (2.0 / w.sum())
    power = (amplitude ** 2).mean(axis=0)
    level = 10 * np.log10(np.maximum(power, 1e-30) / FULL_SCALE ** 2)
    return np.fft.rfftfreq(nfft, 1.0 / sample_rate), level


def attach(overlay):
    """Return the AdcCapture of an overlay built with RFSOC_AWG_ADC=1, else ``None``."""
    key = "adc_bram_ctrl"
    if key not in getattr(overlay, "mem_dict", {}):
        return None
    from pynq import MMIO   # only available on the board

    entry = overlay.mem_dict[key]
    mmio = MMIO(entry["phys_addr"], entry["addr_range"])
    return AdcCapture(
        buffer=mmio.array[: mmio.length].view(np.int16),
        ctrl_channel=overlay.axi_gpio_adc_ctl.channel1,
        status_channel=overlay.axi_gpio_adc_ctl.channel2,
        delay_channel=overlay.axi_gpio_adc_cfg.channel1,
        rows_channel=overlay.axi_gpio_adc_cfg.channel2,
    )
