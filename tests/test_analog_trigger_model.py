"""Logic tests for the analog trigger detector, run against the register-transfer model."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hdl" / "model"))

from analog_trigger_model import (  # noqa: E402
    CTRL_ADC_SHIFT,
    CTRL_CLEAR,
    CTRL_ENABLE,
    CTRL_MODE_SHIFT,
    MODE_ENVELOPE,
    MODE_FALLING,
    MODE_RISING,
    AnalogTriggerModel,
    word_from_samples,
)

FS = 4.9152e9


def run(signal, mode=MODE_ENVELOPE, on=2000, off=1000, release=0, adc=0, valid=None, enable=True):
    """Feed `signal` (int16 samples, multiple of 16) on input `adc`; return `active` per word."""
    dut = AnalogTriggerModel()
    dut.inp.ctrl = (CTRL_ENABLE if enable else 0) | (adc << CTRL_ADC_SHIFT) | (mode << CTRL_MODE_SHIFT)
    dut.inp.thr = ((off & 0xFFFF) << 16) | (on & 0xFFFF)
    dut.inp.release_words = release
    for _ in range(4):
        dut.tick()
    words = np.asarray(signal, dtype=np.int16).reshape(-1, 16)
    trace = []
    n = 0
    cycle = 0
    while n < len(words) or cycle < len(words) + 8:
        ok = n < len(words) and (valid is None or valid[cycle % len(valid)])
        dut.inp.tvalid = [0, 0, 0, 0]
        dut.inp.tdata = [0x7FFF7FFF, 0x7FFF7FFF, 0x7FFF7FFF, 0x7FFF7FFF]   # other inputs: loud
        dut.inp.tvalid[adc] = int(ok)
        dut.inp.tdata[adc] = word_from_samples(words[n]) if ok else 0
        n += int(ok)
        dut.tick()
        trace.append(dut.active)
        cycle += 1
        if cycle > 10 * len(words) + 100:
            break
    return trace, dut


def burst(freq, amplitude, quiet_words, burst_words, tail_words=40, noise=30):
    rng = np.random.default_rng(1)
    n = (quiet_words + burst_words + tail_words) * 16
    t = np.arange(n) / FS
    x = rng.normal(0, noise, n)
    sl = slice(quiet_words * 16, (quiet_words + burst_words) * 16)
    x[sl] += amplitude * np.sin(2 * np.pi * freq * t[sl])
    return np.round(x).astype(np.int16)


def edges(trace):
    return [i for i in range(1, len(trace)) if trace[i] and not trace[i - 1]]


@pytest.mark.parametrize("freq", [80e6, 200e6, 1.2e9, 2.3e9])
def test_envelope_mode_fires_once_per_rf_burst_with_fixed_delay(freq):
    signal = burst(freq, 8000, quiet_words=50, burst_words=200)
    trace, dut = run(signal, on=3000, off=1500, release=20)
    assert len(edges(trace)) == 1 and dut.status["events"] == 1
    # the first word that reaches the threshold; `active` is set on the third clock edge after
    # the word is presented, which is trace index + 2
    words = np.abs(signal.reshape(-1, 16)).max(axis=1)
    first = int(np.argmax(words >= 3000))
    assert edges(trace)[0] == first + 2
    assert not trace[-1], "drops after the burst"
    assert dut.status["peak"] == pytest.approx(8000, abs=150)


def test_release_bridges_zero_crossings_of_a_slow_carrier():
    signal = burst(20e6, 8000, quiet_words=20, burst_words=400)      # 15 words per period
    chattering, _ = run(signal, on=6000, off=5000, release=0)
    steady, dut = run(signal, on=6000, off=5000, release=16)
    assert len(edges(chattering)) > 10
    assert len(edges(steady)) == 1 and dut.status["events"] == 1


def test_hysteresis_holds_through_an_amplitude_dip():
    a = burst(500e6, 8000, 20, 100, tail_words=0, noise=0)
    b = burst(500e6, 2500, 0, 100, tail_words=0, noise=0)             # dips below "on", above "off"
    c = burst(500e6, 8000, 0, 100, tail_words=40, noise=0)
    signal = np.concatenate([a, b, c])
    with_hysteresis, _ = run(signal, on=4000, off=1500)
    without, _ = run(signal, on=4000, off=4000)
    assert len(edges(with_hysteresis)) == 1
    assert len(edges(without)) == 2


def test_noise_below_threshold_never_fires_and_disabled_detector_is_silent():
    noise = burst(1e9, 0, 300, 0, tail_words=0, noise=200)
    trace, dut = run(noise, on=2000, off=1000)
    assert not any(trace) and dut.status["events"] == 0
    assert 400 < dut.status["peak"] < 1500
    loud = burst(1e9, 20000, 10, 100)
    trace, dut = run(loud, enable=False)
    assert not any(trace) and dut.status["events"] == 0
    assert dut.status["peak"] > 19000, "peak is still measured for setting a threshold"


def test_signed_level_modes():
    pulse = np.zeros(16 * 60, dtype=np.int16)
    pulse[16 * 20 + 5 : 16 * 30] = 5000
    pulse[16 * 40 : 16 * 45] = -5000
    rising, _ = run(pulse, mode=MODE_RISING, on=3000, off=3000)
    assert edges(rising) == [20 + 2] and not rising[30 + 2 + 1]
    falling, _ = run(pulse, mode=MODE_FALLING, on=-3000, off=-3000)
    assert edges(falling) == [40 + 2]
    envelope, _ = run(pulse, mode=MODE_ENVELOPE, on=3000, off=3000)
    assert edges(envelope) == [22, 42]


def test_input_select_ignores_the_other_inputs():
    quiet = np.zeros(16 * 50, dtype=np.int16)
    for adc in range(4):
        trace, _ = run(quiet, adc=adc)          # the other three inputs are at full scale
        assert not any(trace)


def test_stream_gaps_do_not_change_the_result():
    signal = burst(300e6, 8000, 30, 100)
    trace, dut = run(signal, on=3000, off=1500, release=10, valid=[1, 1, 0, 1, 0])
    assert len(edges(trace)) == 1 and dut.status["events"] == 1


def test_full_scale_negative_sample_and_clear():
    signal = np.zeros(16 * 20, dtype=np.int16)
    signal[100] = -32768
    trace, dut = run(signal, on=32767, off=32767)
    assert len(edges(trace)) == 1 and dut.status["peak"] == 32767
    dut.inp.ctrl |= CTRL_CLEAR
    for _ in range(4):
        dut.tick()
    assert dut.status["peak"] == 0 and dut.status["events"] == 0
