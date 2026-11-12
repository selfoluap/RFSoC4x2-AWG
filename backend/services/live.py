"""Live ADC view: free-running captures turned into display frames for the web UI.

Stage 1 of the live view: the processor re-arms the capture block, self-triggers it, reads a
short record and computes the spectrum, a few times per second. Each frame sees at most 53 us of
signal, so this shows continuous or repetitive signals, not rare events.

This module knows nothing about FastAPI or PYNQ: ``LiveView`` takes a capture function and
``run_live_session`` takes send/receive coroutines, so both are unit tested off the board.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Awaitable, Callable

import numpy as np

from firmware.adc_capture import ADC_INPUTS, ADC_SAMPLE_RATE, FULL_SCALE, spectrum

VALID_NFFT = (1024, 2048, 4096, 8192, 16384, 32768, 65536)
MAX_SEGMENTS = 4                 # spectra averaged inside one capture
MAX_AVERAGING = 256              # frames in the running average
MAX_FPS = 30.0
CLIP_LEVEL = 32760               # 14-bit ADC, MSB-aligned: full scale is +/-32764
FLOOR_DB = -160.0


class LiveView:
    """Produces display frames from a capture function ``capture(adc, samples) -> int16 array``."""

    def __init__(self, capture: Callable[[str, int], np.ndarray], capacity: int = 262144,
                 sample_rate: float = ADC_SAMPLE_RATE, points: int = 1024, trace_points: int = 512,
                 capture_triggered=None, trigger_info=None):
        """``capture_triggered(adc, samples) -> (int16 array, triggered)`` enables the "analog"
        trigger mode, which holds the time trace still; ``trigger_info() -> dict`` adds the
        analog trigger's settings and status to every frame."""
        self._capture = capture
        self._capture_triggered = capture_triggered
        self._trigger_info = trigger_info
        self.trigger = "free"
        self.capacity = int(capacity)
        self.sample_rate = float(sample_rate)
        self.points = int(points)
        self.trace_points = int(trace_points)
        self.adc = "A"
        self.nfft = 4096
        self.averaging = 4
        self.max_hold = False
        self.fps = 15.0
        self.running = True
        self.seq = 0
        self._reset()

    # ------------------------------------------------------------------ settings
    def settings(self) -> dict:
        return {
            "adc": self.adc,
            "nfft": self.nfft,
            "averaging": self.averaging,
            "max_hold": self.max_hold,
            "fps": self.fps,
            "running": self.running,
            "trigger": self.trigger,
        }

    def info(self) -> dict:
        return {
            "sample_rate_hz": self.sample_rate,
            "nyquist_hz": self.sample_rate / 2,
            "capacity_samples": self.capacity,
            "inputs": sorted(ADC_INPUTS),
            "nfft_choices": [n for n in VALID_NFFT if n <= self.capacity],
            "max_fps": MAX_FPS,
            "trigger_modes": ["free", "analog"] if self._capture_triggered else ["free"],
            "settings": self.settings(),
        }

    def update(self, adc=None, nfft=None, averaging=None, max_hold=None, fps=None, running=None,
               reset_hold=None, trigger=None, **unknown) -> None:
        """Apply a settings message from the browser. Raises ValueError and changes nothing if
        any value is invalid."""
        if unknown:
            raise ValueError(f"unknown setting: {', '.join(sorted(unknown))}")
        new_adc, new_nfft, new_avg, new_fps = self.adc, self.nfft, self.averaging, self.fps
        if adc is not None:
            new_adc = str(adc).upper()
            if new_adc not in ADC_INPUTS:
                raise ValueError(f"adc must be one of {sorted(ADC_INPUTS)}")
        if nfft is not None:
            new_nfft = int(nfft)
            if new_nfft not in VALID_NFFT or new_nfft > self.capacity:
                raise ValueError(f"nfft must be one of {[n for n in VALID_NFFT if n <= self.capacity]}")
        if averaging is not None:
            new_avg = int(averaging)
            if not 1 <= new_avg <= MAX_AVERAGING:
                raise ValueError(f"averaging must be between 1 and {MAX_AVERAGING}")
        if fps is not None:
            new_fps = float(fps)
            if not (math.isfinite(new_fps) and 0.5 <= new_fps <= MAX_FPS):
                raise ValueError(f"fps must be between 0.5 and {MAX_FPS:g}")

        if trigger is not None:
            if trigger not in ("free", "analog"):
                raise ValueError('trigger must be "free" or "analog"')
            if trigger == "analog" and self._capture_triggered is None:
                raise ValueError("this bitstream has no analog trigger")

        changed_signal = (new_adc, new_nfft) != (self.adc, self.nfft)
        if trigger is not None:
            self.trigger = trigger
        self.adc, self.nfft, self.averaging, self.fps = new_adc, new_nfft, new_avg, new_fps
        if max_hold is not None:
            self.max_hold = bool(max_hold)
            if not self.max_hold:
                self._hold = None
        if running is not None:
            self.running = bool(running)
        if changed_signal:
            self._reset()
        elif reset_hold:
            self._hold = None

    def _reset(self) -> None:
        self._average = None       # running average of the power spectrum (linear)
        self._count = 0
        self._hold = None

    # ------------------------------------------------------------------ frames
    def frame(self) -> dict:
        """Capture once and return one display frame (a JSON-serialisable dict)."""
        started = time.perf_counter()
        segments = max(1, min(MAX_SEGMENTS, self.capacity // self.nfft))
        triggered = None
        if self.trigger == "analog":
            samples, triggered = self._capture_triggered(self.adc, self.nfft * segments)
            samples = np.asarray(samples)
        else:
            samples = np.asarray(self._capture(self.adc, self.nfft * segments))
        if samples.size < self.nfft:
            raise RuntimeError(f"capture returned {samples.size} samples, needed {self.nfft}")

        freqs, level = spectrum(samples, self.sample_rate, nfft=self.nfft)
        power = 10.0 ** (level / 10.0)

        self._count = min(self._count + 1, self.averaging)
        if self._average is None or self._average.shape != power.shape:
            self._average, self._count = power, 1
        else:
            self._average = self._average + (power - self._average) / self._count
        if self.max_hold:
            if self._hold is None or self._hold.shape != power.shape:
                self._hold = power
            else:
                self._hold = np.maximum(self._hold, power)

        peak = int(np.argmax(self._average[1:])) + 1          # skip the DC bin
        display, f_start, f_step = self._reduce(self._average, freqs)
        hold = self._reduce(self._hold, freqs)[0] if self._hold is not None else None

        data = samples.astype(np.float64)
        rms = float(np.sqrt(np.mean((data - data.mean()) ** 2)))
        self.seq += 1
        return {
            "type": "frame",
            "seq": self.seq,
            "adc": self.adc,
            "nfft": self.nfft,
            "segments": segments,
            "averaged": self._count,
            "f_start_hz": f_start,
            "f_step_hz": f_step,
            "spectrum_db": self._to_db_list(display),
            "max_hold_db": self._to_db_list(hold) if hold is not None else None,
            "peak_hz": float(freqs[peak]),
            "peak_db": round(float(10 * np.log10(max(self._average[peak], 1e-30))), 2),
            # 0 dB = the rms of a full-scale sine
            "rms_db": round(20 * math.log10(max(rms * math.sqrt(2) / FULL_SCALE, 1e-9)), 2),
            "clipping": bool(np.abs(samples).max() >= CLIP_LEVEL),
            "trace": samples[: self.trace_points].astype(int).tolist(),
            "trace_dt_s": 1.0 / self.sample_rate,
            # None: free running. True: the record starts at an analog trigger. False: no
            # trigger came in time and the record was taken anyway.
            "triggered": None if triggered is None else bool(triggered),
            "trigger": self._trigger_info() if self._trigger_info else None,
            "capture_ms": round((time.perf_counter() - started) * 1e3, 2),
        }

    def _reduce(self, power, freqs):
        """Reduce a power spectrum to at most ``points`` display points, keeping peaks."""
        bins = power[1:]                                       # the DC bin is not shown
        f_step = float(freqs[1] - freqs[0])
        group = max(1, -(-bins.size // self.points))
        if group > 1:
            usable = bins.size // group * group
            bins = bins[:usable].reshape(-1, group).max(axis=1)
        return bins, f_step * (1 + (group - 1) / 2.0), f_step * group

    @staticmethod
    def _to_db_list(power) -> list:
        level = 10 * np.log10(np.maximum(power, 10 ** (FLOOR_DB / 10)))
        return np.round(level, 1).tolist()


async def run_live_session(
    view: LiveView,
    receive_json: Callable[[], Awaitable[dict]],
    send_json: Callable[[dict], Awaitable[None]],
    to_thread=asyncio.to_thread,
    sleep=asyncio.sleep,
) -> None:
    """Serve one browser connection until it closes.

    Settings messages from the browser are applied as they arrive; frames are captured in a
    worker thread and sent at the requested rate while ``running`` is set. Returns when
    ``receive_json`` or ``send_json`` raises (the socket closed), so the capture loop never
    runs without a viewer.
    """
    closed = asyncio.Event()

    async def reader():
        try:
            while True:
                message = await receive_json()
                try:
                    if not isinstance(message, dict):
                        raise ValueError("settings must be a JSON object")
                    view.update(**message)
                    await send_json({"type": "settings", **view.settings()})
                except (ValueError, TypeError) as error:
                    await send_json({"type": "error", "message": str(error)})
        except Exception:
            pass                                               # disconnect
        finally:
            closed.set()

    reader_task = asyncio.ensure_future(reader())
    try:
        await send_json({"type": "info", **view.info()})
        while not closed.is_set():
            if not view.running:
                await sleep(0.05)
                continue
            started = time.monotonic()
            try:
                frame = await to_thread(view.frame)
            except Exception as error:                         # hardware trouble: report, pause
                view.running = False
                await send_json({"type": "error", "message": f"capture failed: {error}"})
                await send_json({"type": "settings", **view.settings()})
                continue
            await send_json(frame)
            await sleep(max(0.0, 1.0 / view.fps - (time.monotonic() - started)))
    except Exception:
        pass                                                   # send failed: socket closed
    finally:
        reader_task.cancel()
        try:
            await reader_task
        except BaseException:
            pass
