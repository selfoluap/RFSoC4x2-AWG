"""Off-board tests for the capture driver (run against the FPGA model) and the spectrum."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hdl" / "model"))

from adc_capture_model import AdcCaptureModel  # noqa: E402
from firmware import adc_capture as ac  # noqa: E402

FS = ac.ADC_SAMPLE_RATE


class ModelHardware:
    """GPIO channels and capture memory wired to the model of ADCCapture.

    ADC input s carries the sample stream s*1000 + n (int16, wrapping), 16 samples per word.
    """

    def __init__(self):
        self.dut = AdcCaptureModel()
        self.buffer = np.zeros(self.dut.ROWS * 16, dtype=np.int16)
        self.n = 0
        self.advance(6)

    def word(self, s, n):
        values = (s * 1000 + n * 16 + np.arange(16)).astype(np.int16)
        return int.from_bytes(values.tobytes(), "little")

    def advance(self, cycles):
        for _ in range(cycles):
            self.dut.inp.tvalid = [1, 1, 1, 1]
            self.dut.inp.tdata = [self.word(s, self.n) for s in range(4)]
            self.n += 1
            self.dut.tick()
        for row, value in self.dut.memory.items():
            self.buffer[row * 16 : row * 16 + 16] = np.frombuffer(value.to_bytes(32, "little"), dtype=np.int16)

    def channel(self, register):
        hardware = self

        class Channel:
            def write(self, value, mask):
                setattr(hardware.dut.inp, register, value & mask)
                hardware.advance(6)

            def read(self):
                hardware.advance(300)
                s = hardware.dut.status
                return s["armed"] | s["busy"] << 1 | s["done"] << 2 | s["rows"] << 16

        return Channel()


def make_capture():
    hardware = ModelHardware()
    capture = ac.AdcCapture(
        hardware.buffer,
        ctrl_channel=hardware.channel("ctrl"),
        status_channel=hardware.channel("status"),
        delay_channel=hardware.channel("delay"),
        rows_channel=hardware.channel("nrows"),
        sleep=lambda seconds: None,
    )
    return capture, hardware


@pytest.mark.parametrize("adc,index", [("A", 3), ("b", 2), ("C", 1), ("D", 0), (2, 2)])
def test_capture_returns_consecutive_samples_of_the_chosen_input(adc, index):
    capture, _ = make_capture()
    data = capture.capture(adc, samples=1000)
    assert data.dtype == np.int16 and data.size == 1008          # rounded up to whole words
    assert np.array_equal(np.diff(data), np.ones(1007, dtype=np.int16))
    assert (int(data[0]) - index * 1000) % 16 == 0
    assert capture.status() == {"armed": False, "busy": False, "done": True, "samples": 1008}


def test_delay_is_converted_to_whole_words_and_shifts_the_capture():
    capture, hardware = make_capture()
    capture.arm("D", samples=32, delay_s=100 * 16 / FS)
    assert hardware.dut.inp.delay == 100
    capture.trigger()
    delayed = capture.read()
    assert np.array_equal(np.diff(delayed), np.ones(31, dtype=np.int16))


def test_dac_start_trigger_and_timeout_without_trigger():
    capture, hardware = make_capture()
    capture.arm("A", samples=64, trigger="dac0")
    with pytest.raises(ac.CaptureError, match="no trigger"):
        capture.read(timeout=0.01)
    hardware.dut.inp.dac0_status = 2                              # DAC0 sequence starts running
    assert capture.read().size == 64


def test_bad_arguments():
    capture, _ = make_capture()
    with pytest.raises(ac.CaptureError):
        capture.read()
    with pytest.raises(ValueError):
        capture.arm("A", samples=capture.capacity + 1)
    with pytest.raises(ValueError):
        capture.arm("A", trigger="adc")
    with pytest.raises(KeyError):
        capture.arm("E")
    assert capture.capacity == 262144


def test_attach_is_a_no_op_without_the_adc_path():
    class Overlay:
        mem_dict = {}

    assert ac.attach(Overlay()) is None


# ---------------------------------------------------------------------------- spectrum
def tone(freq, amplitude, n=65536):
    return np.round(amplitude * np.sin(2 * np.pi * freq * np.arange(n) / FS)).astype(np.int16)


def test_spectrum_reports_tone_frequency_and_level_in_dbfs():
    freq = 5461 * FS / 65536                                      # ~409.6 MHz, on a bin
    freqs, level = ac.spectrum(tone(freq, 16384))
    peak = int(np.argmax(level))
    assert freqs[peak] == pytest.approx(freq)
    assert level[peak] == pytest.approx(-6.02, abs=0.05)          # half of full scale
    assert freqs[-1] == pytest.approx(2457.6e6)
    assert np.median(level) < -100                                # quantisation floor only


def test_spectrum_level_is_right_between_bins_within_hann_scalloping():
    freq = 5461.5 * FS / 65536
    _, level = ac.spectrum(tone(freq, 32767))
    assert -1.5 < level.max() < 0.01


def test_spectrum_averaging_keeps_levels_and_smooths_noise():
    rng = np.random.default_rng(0)
    data = tone(300e6, 8000, 262144) + rng.normal(0, 200, 262144)
    f1, single = ac.spectrum(data, nfft=4096)
    assert single.max() == pytest.approx(20 * np.log10(8000 / 32768), abs=1.0)
    _, one_segment = ac.spectrum(data[:4096], nfft=4096)
    away = np.abs(f1 - 300e6) > 50e6
    assert single[away].std() < 0.5 * one_segment[away].std()


def test_spectrum_shows_aliasing_of_tones_above_nyquist():
    freqs, level = ac.spectrum(tone(3.0e9, 16384))
    assert freqs[int(np.argmax(level))] == pytest.approx(FS - 3.0e9, abs=FS / 65536)


def test_spectrum_rejects_bad_input():
    with pytest.raises(ValueError):
        ac.spectrum(np.zeros(8))
    with pytest.raises(ValueError):
        ac.spectrum(np.zeros(100), nfft=128)
    with pytest.raises(ValueError):
        ac.spectrum(np.zeros(100), window="kaiser")
