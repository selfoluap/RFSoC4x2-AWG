"""Tests for the live ADC view: frame content and the per-connection session loop."""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from backend.services.live import LiveView, run_live_session  # noqa: E402

FS = 4.9152e9


class FakeAdc:
    """Each input carries a different tone; counts captures."""

    def __init__(self, noise=20.0):
        self.tones = {"A": (400e6, 8000), "B": (900e6, 2000), "C": (1500e6, 30000), "D": (100e6, 500)}
        self.noise = noise
        self.calls = []
        self.rng = np.random.default_rng(0)
        self.burst = None

    def __call__(self, adc, samples):
        self.calls.append((adc, samples))
        freq, amp = self.tones[adc]
        t = (np.arange(samples) + self.rng.integers(0, 10_000)) / FS
        data = amp * np.sin(2 * np.pi * freq * t) + self.rng.normal(0, self.noise, samples)
        if self.burst:
            data += self.burst[1] * np.sin(2 * np.pi * self.burst[0] * t)
            self.burst = None
        return np.clip(np.round(data), -32764, 32764).astype(np.int16)


def frame_freqs(frame):
    return frame["f_start_hz"] + frame["f_step_hz"] * np.arange(len(frame["spectrum_db"]))


def test_frame_finds_the_tone_and_is_json_serialisable():
    adc = FakeAdc()
    view = LiveView(adc)
    frame = view.frame()
    json.dumps(frame)
    assert adc.calls == [("A", 4096 * 4)]
    assert frame["peak_hz"] == pytest.approx(400e6, abs=FS / 4096)
    assert frame["peak_db"] == pytest.approx(20 * np.log10(8000 / 32768), abs=1.5)
    assert frame["rms_db"] == pytest.approx(20 * np.log10(8000 / 32768), abs=0.3)
    assert not frame["clipping"]
    assert len(frame["trace"]) == 512 and frame["trace_dt_s"] == pytest.approx(1 / FS)
    spectrum = np.array(frame["spectrum_db"])
    assert len(spectrum) == 1024
    freqs = frame_freqs(frame)
    assert freqs[int(spectrum.argmax())] == pytest.approx(400e6, abs=2 * frame["f_step_hz"])
    assert freqs[-1] < FS / 2 and freqs[0] > 0


@pytest.mark.parametrize("nfft,points", [(1024, 512), (4096, 1024), (65536, 1024)])
def test_display_reduction_keeps_narrow_peaks(nfft, points):
    view = LiveView(FakeAdc(noise=5.0))
    view.update(nfft=nfft, averaging=1)
    frame = view.frame()
    assert len(frame["spectrum_db"]) == points
    assert max(frame["spectrum_db"]) == pytest.approx(frame["peak_db"], abs=1.6)
    assert frame["segments"] == min(4, 262144 // nfft)


def test_switching_input_restarts_the_average_and_shows_the_other_signal():
    view = LiveView(FakeAdc())
    for _ in range(3):
        view.frame()
    view.update(adc="c")
    frame = view.frame()
    assert frame["adc"] == "C" and frame["averaged"] == 1
    assert frame["peak_hz"] == pytest.approx(1500e6, abs=FS / 4096)
    assert not frame["clipping"]                                  # 30000 is below full scale


def test_clipping_flag_only_at_full_scale():
    adc = FakeAdc(noise=0.0)
    view = LiveView(adc)
    assert not view.frame()["clipping"]
    adc.tones["A"] = (400e6, 40000)                               # overdriven: clipped by the ADC
    assert view.frame()["clipping"]


def test_averaging_smooths_the_noise_floor():
    def floor_spread(averaging):
        view = LiveView(FakeAdc(noise=50.0))
        view.update(averaging=averaging)
        for _ in range(averaging):
            frame = view.frame()
        level = np.array(frame["spectrum_db"])
        away = np.abs(frame_freqs(frame) - 400e6) > 100e6
        return level[away].std(), frame["averaged"]

    single, n1 = floor_spread(1)
    averaged, n16 = floor_spread(16)
    assert (n1, n16) == (1, 16)
    assert averaged < 0.5 * single


def test_max_hold_keeps_a_burst_that_the_average_forgets():
    adc = FakeAdc()
    view = LiveView(adc)
    view.update(max_hold=True, averaging=2)
    view.frame()
    adc.burst = (2000e6, 5000)
    view.frame()
    for _ in range(12):
        frame = view.frame()
    freqs = frame_freqs(frame)
    at_burst = int(np.argmin(np.abs(freqs - 2000e6)))
    window = slice(at_burst - 2, at_burst + 3)
    assert max(frame["max_hold_db"][window]) > -20
    assert max(frame["spectrum_db"][window]) < -50
    view.update(reset_hold=True)
    assert max(view.frame()["max_hold_db"][window]) < -50
    view.update(max_hold=False)
    assert view.frame()["max_hold_db"] is None


def test_invalid_settings_are_rejected_and_change_nothing():
    view = LiveView(FakeAdc())
    before = view.settings()
    for bad in ({"adc": "E"}, {"nfft": 1000}, {"averaging": 0}, {"fps": 100}, {"fps": "nan"},
                {"nfft": 2048, "averaging": 1000}, {"colour": "red"}):
        with pytest.raises(ValueError):
            view.update(**bad)
        assert view.settings() == before
    small = LiveView(FakeAdc(), capacity=8192)
    assert small.info()["nfft_choices"] == [1024, 2048, 4096, 8192]
    with pytest.raises(ValueError):
        small.update(nfft=16384)


# ---------------------------------------------------------------------------- session loop
class FakeSocket:
    def __init__(self, script):
        self.script = list(script)       # (after_n_frames, message) or (after_n_frames, "close")
        self.sent = []
        self.closed = asyncio.Event()

    def frames(self):
        return [m for m in self.sent if m["type"] == "frame"]

    async def receive_json(self):
        while True:
            if self.script and len(self.frames()) >= self.script[0][0]:
                _, message = self.script.pop(0)
                if message == "close":
                    self.closed.set()
                    raise ConnectionError("closed")
                return message
            await asyncio.sleep(0.001)

    async def send_json(self, message):
        if self.closed.is_set():
            raise ConnectionError("closed")
        json.dumps(message)
        self.sent.append(message)


def run_session(view, script, timeout=5.0):
    socket = FakeSocket(script)

    async def main():
        await asyncio.wait_for(
            run_live_session(view, socket.receive_json, socket.send_json,
                             sleep=lambda s: asyncio.sleep(min(s, 0.002))),
            timeout,
        )

    asyncio.run(main())
    return socket


def test_session_sends_info_then_frames_and_stops_capturing_when_the_browser_leaves():
    adc = FakeAdc()
    view = LiveView(adc)
    socket = run_session(view, [(5, "close")])
    assert socket.sent[0]["type"] == "info" and socket.sent[0]["inputs"] == ["A", "B", "C", "D"]
    assert len(socket.frames()) >= 5
    assert [f["seq"] for f in socket.frames()] == list(range(1, len(socket.frames()) + 1))
    captures = len(adc.calls)
    assert captures <= len(socket.frames()) + 1, "no captures without a viewer"


def test_session_applies_settings_and_reports_bad_ones():
    view = LiveView(FakeAdc())
    socket = run_session(view, [
        (2, {"adc": "B", "nfft": 8192}),
        (4, {"nfft": 5}),
        (6, {"running": False}),
        (6, {"running": True, "adc": "D"}),
        (8, "close"),
    ])
    types = [m["type"] for m in socket.sent]
    assert types.count("settings") == 3 and types.count("error") == 1
    assert "nfft must be" in next(m["message"] for m in socket.sent if m["type"] == "error")
    frames = socket.frames()
    assert frames[0]["adc"] == "A" and frames[-1]["adc"] == "D"
    assert any(f["adc"] == "B" and f["nfft"] == 8192 for f in frames)


def test_session_reports_capture_failure_and_pauses():
    def broken(adc, samples):
        raise RuntimeError("timed out waiting for the capture to finish")

    view = LiveView(broken)
    socket = run_session(view, [(0, {"fps": 5}), (0, "close")])
    errors = [m for m in socket.sent if m["type"] == "error"]
    assert errors and "capture failed" in errors[0]["message"]
    assert not view.running


# ---------------------------------------------------------------------------- analog trigger mode
def test_analog_trigger_mode_uses_the_triggered_capture_and_reports_it():
    adc = FakeAdc()
    outcomes = [True, True, False]

    def triggered(name, samples):
        return adc(name, samples), outcomes.pop(0)

    info = {"enabled": True, "adc": "A", "mode": "envelope", "threshold": 4000}
    view = LiveView(adc, capture_triggered=triggered, trigger_info=lambda: info)
    assert view.info()["trigger_modes"] == ["free", "analog"]
    free = view.frame()
    assert free["triggered"] is None and free["trigger"] == info
    view.update(trigger="analog")
    assert view.settings()["trigger"] == "analog"
    assert [view.frame()["triggered"] for _ in range(3)] == [True, True, False]
    json.dumps(view.frame.__self__.settings())
    view.update(trigger="free")
    assert view.frame()["triggered"] is None


def test_analog_trigger_mode_needs_the_hardware():
    view = LiveView(FakeAdc())
    assert view.info()["trigger_modes"] == ["free"]
    with pytest.raises(ValueError, match="no analog trigger"):
        view.update(trigger="analog")
    with pytest.raises(ValueError):
        view.update(trigger="edge")
    assert view.settings()["trigger"] == "free" and view.frame()["trigger"] is None
