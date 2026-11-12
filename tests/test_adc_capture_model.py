"""Logic tests for the ADC capture block, run against the register-transfer model."""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hdl" / "model"))

from adc_capture_model import CTRL_ADC_SHIFT, CTRL_ARM, CTRL_SRC_SHIFT, CTRL_SW_TRIGGER, AdcCaptureModel  # noqa: E402


class Bench:
    """Feeds four streams; stream s carries the word (s << 40) | n for its n-th valid word."""

    def __init__(self, valid_probability=1.0, seed=1):
        self.dut = AdcCaptureModel()
        self.rng = random.Random(seed)
        self.p = valid_probability
        self.counts = [0, 0, 0, 0]
        self.cycle = 0
        self.step(4, aresetn=0)

    def step(self, n=1, aresetn=1):
        for _ in range(n):
            for s in range(4):
                valid = int(self.rng.random() < self.p)
                self.dut.inp.tvalid[s] = valid
                self.dut.inp.tdata[s] = (s << 40) | self.counts[s] if valid else 0xDEAD
                self.counts[s] += valid
            self.dut.tick(aresetn)
            self.cycle += 1

    def arm(self, adc, nrows, delay=0, source=0):
        i = self.dut.inp
        i.ctrl = (adc << CTRL_ADC_SHIFT) | (source << CTRL_SRC_SHIFT)
        i.nrows, i.delay = nrows, delay
        self.step(3)
        i.ctrl |= CTRL_ARM
        self.run_until(lambda: self.dut.status["armed"])

    def sw_trigger(self):
        self.dut.inp.ctrl |= CTRL_SW_TRIGGER
        self.step(4)
        self.dut.inp.ctrl &= ~CTRL_SW_TRIGGER

    def run_until(self, predicate, limit=200_000):
        for _ in range(limit):
            if predicate():
                return
            self.step()
        raise AssertionError("condition never met")

    def captured(self, nrows):
        self.step(3)
        return [self.dut.memory[r] for r in range(nrows)]


@pytest.mark.parametrize("adc", [0, 1, 2, 3])
@pytest.mark.parametrize("nrows", [1, 2, 100])
def test_records_consecutive_words_of_the_selected_adc(adc, nrows):
    bench = Bench()
    bench.arm(adc, nrows)
    bench.step(20)
    assert bench.dut.memory == {}, "nothing is written before the trigger"
    bench.sw_trigger()
    bench.run_until(lambda: bench.dut.status["done"])
    words = bench.captured(nrows)
    assert all(w >> 40 == adc for w in words)
    first = words[0] & 0xFFFFFFFF
    assert [w & 0xFFFFFFFF for w in words] == list(range(first, first + nrows))
    assert bench.dut.status["rows"] == nrows and len(bench.dut.memory) == nrows


def test_whole_memory_when_nrows_is_zero_or_too_large():
    for nrows in (0, 100_000):
        bench = Bench()
        bench.arm(1, nrows)
        bench.sw_trigger()
        bench.run_until(lambda: bench.dut.status["done"])
        words = bench.captured(16384)
        assert len(bench.dut.memory) == 16384
        assert words[-1] - words[0] == 16383


def test_delay_skips_exactly_that_many_words():
    firsts = {}
    for delay in (0, 1, 2, 500):
        bench = Bench()
        bench.arm(0, 4, delay=delay)
        bench.sw_trigger()
        bench.run_until(lambda: bench.dut.status["done"])
        firsts[delay] = (bench.captured(4)[0] & 0xFFFFFFFF, bench.cycle)
    base = firsts[0][0]
    assert {d: f - base for d, (f, _) in firsts.items()} == {0: 0, 1: 1, 2: 2, 500: 500}


def test_gaps_in_the_stream_lose_no_word():
    bench = Bench(valid_probability=0.6, seed=3)
    bench.arm(2, 300, delay=40)
    bench.sw_trigger()
    bench.run_until(lambda: bench.dut.status["done"])
    words = [w & 0xFFFFFFFF for w in bench.captured(300)]
    assert words == list(range(words[0], words[0] + 300))


@pytest.mark.parametrize("source,field", [(1, "ext_trigger"), (2, "dac0_status"), (3, "dac2_status")])
def test_trigger_sources_are_only_honoured_when_selected(source, field):
    for selected in (0, source):
        bench = Bench()
        bench.arm(3, 8, source=selected)
        setattr(bench.dut.inp, field, 2 if field != "ext_trigger" else 1)
        bench.step(60)
        assert bench.dut.status["done"] == int(selected == source)
        assert bench.dut.status["armed"] == int(selected != source)


def test_capture_starts_a_fixed_number_of_cycles_after_playback_starts():
    offsets = set()
    for wait in (5, 6, 23):
        bench = Bench()
        bench.arm(0, 2, source=2)
        bench.step(wait)
        at = bench.counts[0]
        bench.dut.inp.dac0_status = 2
        bench.run_until(lambda: bench.dut.status["done"])
        offsets.add((bench.captured(2)[0] & 0xFFFFFFFF) - at)
    assert len(offsets) == 1


def test_disarm_and_rearm_overwrites_from_the_start():
    bench = Bench()
    bench.arm(0, 50)
    bench.sw_trigger()
    bench.run_until(lambda: bench.dut.status["done"])
    bench.dut.inp.ctrl &= ~CTRL_ARM
    bench.step(5)
    assert bench.dut.status == {"armed": 0, "busy": 0, "done": 0, "rows": 50}
    bench.arm(1, 10)
    assert bench.dut.status["rows"] == 0
    bench.sw_trigger()
    bench.run_until(lambda: bench.dut.status["done"])
    words = bench.captured(10)
    assert all(w >> 40 == 1 for w in words) and bench.dut.memory[10] >> 40 == 0
