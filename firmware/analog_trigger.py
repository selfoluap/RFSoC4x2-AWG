"""Analog trigger on an ADC input (AnalogTrigger in the FPGA).

Watches one ADC input and raises a trigger when a signal appears: an RF burst of any carrier
frequency ("envelope"), or the signal crossing a level ("rising" / "falling"). The trigger

- starts a capture armed with ``trigger="analog"``,
- starts the DAC players armed with ``external_trigger=True`` (it is OR-ed with the PMOD pin),
- holds the live view's time trace still.

The ADC inputs are AC-coupled: a slow level change does not reach the detector, an RF burst or
a fast edge does.

Nothing here imports pynq at module level, so it can be unit tested off the board.
"""

from __future__ import annotations

import math

ADC_SAMPLE_RATE = 4.9152e9
SAMPLES_PER_WORD = 16
WORD_SECONDS = SAMPLES_PER_WORD / ADC_SAMPLE_RATE      # 3.26 ns: time resolution of the detector
ADC_INPUTS = {"D": 0, "C": 1, "B": 2, "A": 3}          # as in adc_capture.py
MODES = {"envelope": 0, "rising": 1, "falling": 2}

CTRL_ENABLE = 1 << 0
CTRL_ADC_SHIFT = 1
CTRL_MODE_SHIFT = 3
CTRL_CLEAR = 1 << 5
MAX_RELEASE_WORDS = 0xFFFF


class AnalogTrigger:
    """GPIO channels are objects with ``write(value, mask)`` / ``read()``."""

    def __init__(self, ctrl_channel, status_channel, thr_channel, release_channel):
        self._ctrl_channel = ctrl_channel
        self._status_channel = status_channel
        self._thr_channel = thr_channel
        self._release_channel = release_channel
        self.adc = "A"
        self.mode = "envelope"
        self.threshold = 2000
        self.off_threshold = 1000
        self.release_s = 8 * WORD_SECONDS
        self.enabled = False
        self._ctrl = 0
        self._write_ctrl(0)

    def configure(self, adc="A", threshold=2000, mode="envelope", off_threshold=None,
                  release_s=None, enable=True) -> None:
        """Set up the detector.

        ``threshold``: ADC counts (full scale is 32767). In "envelope" mode it is the RF
        amplitude that counts as "signal present"; in "rising" / "falling" mode it is a signed
        level.
        ``off_threshold``: level below which the signal counts as gone. Default: half the
        threshold in envelope mode, the threshold itself otherwise.
        ``release_s``: how long the signal must stay gone before the trigger re-arms, in steps
        of 3.26 ns up to 213 us. Use at least one carrier period in envelope mode (default
        26 ns, enough for carriers above about 40 MHz); default 0 in the level modes.
        """
        index = ADC_INPUTS[adc.upper()] if isinstance(adc, str) else int(adc)
        if not 0 <= index <= 3:
            raise ValueError("adc must be 'A'..'D' or 0..3")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}")
        threshold = int(threshold)
        if mode == "envelope":
            if not 1 <= threshold <= 32767:
                raise ValueError("envelope threshold must be between 1 and 32767 counts")
            off = threshold // 2 if off_threshold is None else int(off_threshold)
            if not 0 <= off <= threshold:
                raise ValueError("off_threshold must be between 0 and the threshold")
            default_release = 8 * WORD_SECONDS
        else:
            if not -32768 <= threshold <= 32767:
                raise ValueError("threshold must be between -32768 and 32767 counts")
            off = threshold if off_threshold is None else int(off_threshold)
            if not -32768 <= off <= 32767:
                raise ValueError("off_threshold must be between -32768 and 32767 counts")
            if (mode == "rising" and off > threshold) or (mode == "falling" and off < threshold):
                raise ValueError("off_threshold is on the wrong side of the threshold")
            default_release = 0.0
        release_s = default_release if release_s is None else float(release_s)
        words = int(round(release_s / WORD_SECONDS))
        if not (math.isfinite(release_s) and 0 <= words <= MAX_RELEASE_WORDS):
            raise ValueError("release_s must be between 0 and 213 us")

        # disable while the thresholds change, so a half-written setting cannot fire
        self._write_ctrl(self._ctrl & ~CTRL_ENABLE)
        self._thr_channel.write(((off & 0xFFFF) << 16) | (threshold & 0xFFFF), 0xFFFFFFFF)
        self._release_channel.write(words, 0xFFFFFFFF)
        ctrl = (index << CTRL_ADC_SHIFT) | (MODES[mode] << CTRL_MODE_SHIFT)
        self._write_ctrl(ctrl)
        self.adc = adc.upper() if isinstance(adc, str) else {v: k for k, v in ADC_INPUTS.items()}[index]
        self.mode, self.threshold, self.off_threshold = mode, threshold, off
        self.release_s = words * WORD_SECONDS
        self.clear()
        if enable:
            self.enable()
        else:
            self.enabled = False

    def enable(self) -> None:
        self._write_ctrl(self._ctrl | CTRL_ENABLE)
        self.enabled = True

    def disable(self) -> None:
        self._write_ctrl(self._ctrl & ~CTRL_ENABLE)
        self.enabled = False

    def clear(self) -> None:
        """Reset the peak reading and the event counter."""
        self._write_ctrl(self._ctrl | CTRL_CLEAR)
        self._write_ctrl(self._ctrl & ~CTRL_CLEAR)

    def status(self) -> dict:
        """``peak`` is the largest |sample| on the selected input since ``clear()``: read it
        with the signal applied to choose a threshold (it is measured even while disabled)."""
        raw = int(self._status_channel.read())
        peak = raw >> 1 & 0x7FFF
        return {
            "active": bool(raw & 1),
            "peak": peak,
            "peak_dbfs": round(20 * math.log10(max(peak, 1) / 32768.0), 1),
            "events": raw >> 16 & 0xFFFF,
        }

    def info(self) -> dict:
        return {
            "enabled": self.enabled,
            "adc": self.adc,
            "mode": self.mode,
            "threshold": self.threshold,
            "off_threshold": self.off_threshold,
            "release_s": self.release_s,
            **self.status(),
        }

    def _write_ctrl(self, value: int) -> None:
        self._ctrl = value & 0xFFFFFFFF
        self._ctrl_channel.write(self._ctrl, 0xFFFFFFFF)


def attach(overlay):
    """Return the AnalogTrigger of an overlay built with RFSOC_AWG_ADC=1, else ``None``."""
    if "axi_gpio_adc_trig_ctl" not in getattr(overlay, "ip_dict", {}):
        return None
    return AnalogTrigger(
        ctrl_channel=overlay.axi_gpio_adc_trig_ctl.channel1,
        status_channel=overlay.axi_gpio_adc_trig_ctl.channel2,
        thr_channel=overlay.axi_gpio_adc_trig_cfg.channel1,
        release_channel=overlay.axi_gpio_adc_trig_cfg.channel2,
    )
