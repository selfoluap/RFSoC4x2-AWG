"""Off-board tests: analog trigger driver against the detector model, and triggered capture."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hdl" / "model"))

from adc_capture_model import AdcCaptureModel  # noqa: E402
from analog_trigger_model import AnalogTriggerModel, word_from_samples  # noqa: E402
from firmware import adc_capture as ac  # noqa: E402
from firmware import analog_trigger as at  # noqa: E402

FS = 4.9152e9


class Rig:
    """Capture block and analog trigger models on shared ADC streams, behind GPIO channels.

    Input A (stream 3) carries what `signal(n)` returns for word n; the others are silent.
    """

    def __init__(self, signal):
        self.signal = signal
        self.capture_model = AdcCaptureModel()
        self.trigger_model = AnalogTriggerModel()
        self.buffer = np.zeros(self.capture_model.ROWS * 16, dtype=np.int16)
        self.n = 0
        self.advance(6)

    def advance(self, cycles):
        for _ in range(cycles):
            word = word_from_samples(self.signal(self.n))
            for model in (self.capture_model, self.trigger_model):
                model.inp.tvalid = [1, 1, 1, 1]
                model.inp.tdata = [0, 0, 0, word]
            self.n += 1
            self.trigger_model.tick()
            self.capture_model.inp.analog_trig = self.trigger_model.active
            self.capture_model.tick()
        for row, value in self.capture_model.memory.items():
            self.buffer[row * 16 : row * 16 + 16] = np.frombuffer(value.to_bytes(32, "little"), dtype=np.int16)

    def channel(self, model, register, read_cycles=40):
        rig = self

        class Channel:
            def write(self, value, mask):
                setattr(model.inp, register, value & mask)
                rig.advance(6)

            def read(self):
                rig.advance(read_cycles)
                if model is rig.trigger_model:
                    s = model.status
                    return s["active"] | s["peak"] << 1 | s["events"] << 16
                s = model.status
                return s["armed"] | s["busy"] << 1 | s["done"] << 2 | s["rows"] << 16

        return Channel()

    def drivers(self):
        trigger = at.AnalogTrigger(
            self.channel(self.trigger_model, "ctrl"), self.channel(self.trigger_model, "status"),
            self.channel(self.trigger_model, "thr"), self.channel(self.trigger_model, "release_words"),
        )
        capture = ac.AdcCapture(
            self.buffer,
            self.channel(self.capture_model, "ctrl"), self.channel(self.capture_model, "status"),
            self.channel(self.capture_model, "delay"), self.channel(self.capture_model, "nrows"),
            sleep=lambda seconds: None,
        )
        return trigger, capture


def pulsed_rf(period_words=400, on_words=60, freq=300e6, amplitude=9000):
    """An RF burst of `on_words` words every `period_words` words."""
    def signal(n):
        if n % period_words >= on_words:
            return np.zeros(16, dtype=np.int16)
        t = ((n % period_words) * 16 + np.arange(16)) / FS       # carrier starts with the burst
        return np.round(amplitude * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    return signal


def silence(n):
    return np.zeros(16, dtype=np.int16)


def test_configure_writes_thresholds_before_enabling_and_reports_status():
    rig = Rig(pulsed_rf())
    trigger, _ = rig.drivers()
    trigger.configure("A", threshold=4000)
    model = rig.trigger_model.inp
    assert model.thr == (2000 << 16) | 4000 and model.release_words == 8
    assert model.ctrl == at.CTRL_ENABLE | (3 << at.CTRL_ADC_SHIFT)
    rig.advance(2000)
    status = trigger.status()
    assert status["events"] >= 4 and 8500 < status["peak"] <= 9000
    assert status["peak_dbfs"] == pytest.approx(20 * np.log10(9000 / 32768), abs=0.5)
    info = trigger.info()
    assert info["enabled"] and info["adc"] == "A" and info["release_s"] == pytest.approx(26.04e-9, rel=1e-3)
    trigger.disable()
    trigger.clear()
    rig.advance(1000)
    assert trigger.status()["events"] == 0 and trigger.status()["peak"] > 8500


def test_configure_validates():
    trigger, _ = Rig(silence).drivers()
    for kwargs in ({"threshold": 0}, {"threshold": 40000}, {"mode": "edge"}, {"release_s": 1e-3},
                   {"off_threshold": 5000, "threshold": 4000},
                   {"mode": "rising", "threshold": 100, "off_threshold": 200}):
        with pytest.raises(ValueError):
            trigger.configure("A", **{"threshold": 2000, **kwargs})
    with pytest.raises(KeyError):
        trigger.configure("Z")
    trigger.configure("b", threshold=-500, mode="falling")
    assert trigger.info()["adc"] == "B" and trigger.off_threshold == -500 and trigger.release_s == 0


def test_analog_triggered_capture_always_starts_at_the_same_point_of_the_burst():
    rig = Rig(pulsed_rf())
    trigger, capture = rig.drivers()
    trigger.configure("A", threshold=4000)
    records = []
    for _ in range(4):
        rig.advance(137)                                   # arbitrary phase between captures
        data, triggered = capture.capture_triggered("A", samples=256, auto_timeout=1.0)
        assert triggered
        records.append(data)
    for data in records[1:]:
        assert np.array_equal(data, records[0]), "trace must stand still from capture to capture"
    assert np.abs(records[0][:16]).max() > 4000, "the record starts inside the burst"


def test_without_a_signal_the_capture_falls_back_to_auto():
    rig = Rig(silence)
    trigger, capture = rig.drivers()
    trigger.configure("A", threshold=4000)
    data, triggered = capture.capture_triggered("A", samples=64, auto_timeout=0.005)
    assert not triggered and data.size == 64 and not data.any()


def test_attach_is_a_no_op_without_the_adc_path():
    class Overlay:
        ip_dict = {}

    assert at.attach(Overlay()) is None
