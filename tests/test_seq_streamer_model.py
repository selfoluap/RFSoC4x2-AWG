"""Logic tests for the sequence-mode player, run against the register-transfer model."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hdl" / "model"))

from seq_streamer_model import (  # noqa: E402
    CTRL_ARM,
    CTRL_END_SHIFT,
    CTRL_EXT_TRIG_EN,
    CTRL_SEQ_MODE,
    CTRL_SW_TRIGGER,
    E_HOLD,
    E_LOOP,
    E_STOP,
    SeqStreamerModel,
    expected_rows,
    row_value,
)


class Bench:
    def __init__(self, **kwargs):
        self.dut = SeqStreamerModel(**kwargs)
        self.cycle = 0
        self.out = []            # (cycle, word) for every non-zero output word
        for _ in range(4):
            self.step(aresetn=0)

    def step(self, n=1, aresetn=1):
        for _ in range(n):
            self.dut.tick(aresetn)
            self.cycle += 1
            if self.dut.axis_tdata:
                self.out.append((self.cycle, self.dut.axis_tdata))

    def words(self):
        return [word for _, word in self.out]

    def configure(self, block_rows, n_blocks, repeats, end_mode, ext=False):
        i = self.dut.inp
        i.seq_lens = (n_blocks << 16) | block_rows
        i.seq_repeats = repeats
        i.seq_ctrl = CTRL_SEQ_MODE | (end_mode << CTRL_END_SHIFT) | (CTRL_EXT_TRIG_EN if ext else 0)
        self.step(3)
        i.seq_ctrl |= CTRL_ARM
        self.run_until(lambda: self.dut.status["armed"])

    def trigger(self):
        self.dut.inp.seq_ctrl |= CTRL_SW_TRIGGER
        self.step(4)
        self.dut.inp.seq_ctrl &= ~CTRL_SW_TRIGGER

    def run_until(self, predicate, limit=2_000_000):
        for _ in range(limit):
            if predicate():
                return
            self.step()
        raise AssertionError("condition never met")


def rows_to_words(rows):
    return [row_value(r) for r in rows]


@pytest.mark.parametrize(
    "block_rows,n_blocks,repeats",
    [(1, 1, 1), (1, 5, 3), (61, 3, 4), (7, 9, 1), (2, 2, 100), (4096, 2, 1), (8192, 1, 2)],
)
def test_stop_mode_plays_each_block_the_requested_number_of_times(block_rows, n_blocks, repeats):
    bench = Bench()
    bench.configure(block_rows, n_blocks, repeats, E_STOP)
    bench.step(20)
    assert bench.out == [], "nothing may play before the trigger"
    bench.trigger()
    bench.run_until(lambda: bench.dut.status["done"])
    bench.step(10)
    assert bench.words() == rows_to_words(expected_rows(block_rows, n_blocks, repeats))
    cycles = [c for c, _ in bench.out]
    assert cycles == list(range(cycles[0], cycles[0] + len(cycles))), "output must be gap-free"
    assert not bench.dut.status["stall"]


def test_the_note_design_runs_120_blocks_of_61_rows_168_times_in_4_ms():
    bench = Bench()
    bench.configure(61, 120, 168, E_STOP)
    bench.trigger()
    bench.run_until(lambda: bench.dut.status["done"])
    bench.step(10)
    assert len(bench.out) == 120 * 168 * 61            # 1,229,760 rows = 39,352,320 samples
    assert len(bench.out) * 32 / 9.8304e9 == pytest.approx(4.003e-3, rel=1e-3)
    # spot-check block boundaries without building the whole expected list
    words = bench.words()
    per_block = 168 * 61
    for block in (0, 1, 57, 119):
        chunk = words[block * per_block : (block + 1) * per_block]
        assert chunk == rows_to_words(list(range(block * 61, block * 61 + 61)) * 168)


def test_loop_mode_wraps_seamlessly_to_block_zero():
    bench = Bench()
    bench.configure(5, 3, 2, E_LOOP)
    bench.trigger()
    bench.step(400)
    one_pass = rows_to_words(expected_rows(5, 3, 2))
    words = bench.words()
    assert len(words) > 10 * len(one_pass) and bench.dut.status["running"]
    assert all(word == one_pass[n % len(one_pass)] for n, word in enumerate(words))


def test_hold_mode_keeps_repeating_the_last_block():
    bench = Bench()
    bench.configure(4, 3, 2, E_HOLD)
    bench.trigger()
    bench.step(300)
    words = bench.words()
    head = rows_to_words(expected_rows(4, 3, 2))
    assert words[: len(head)] == head
    tail = words[len(head) :]
    last_block = rows_to_words(range(8, 12))
    assert len(tail) > 100 and all(w == last_block[n % 4] for n, w in enumerate(tail))
    assert bench.dut.status["holding"] and bench.dut.status["running"]
    assert bench.dut.status["block"] == 2


def test_disarm_stops_within_a_few_cycles_and_rearm_starts_clean():
    bench = Bench()
    bench.configure(8, 4, 1000, E_LOOP)
    bench.trigger()
    bench.step(200)
    bench.dut.inp.seq_ctrl &= ~CTRL_ARM
    bench.step(8)
    bench.out.clear()
    bench.step(50)
    assert bench.out == [] and not bench.dut.status["running"]

    bench.configure(3, 2, 2, E_STOP)
    bench.step(30)
    assert bench.out == []
    bench.trigger()
    bench.run_until(lambda: bench.dut.status["done"])
    bench.step(10)
    assert bench.words() == rows_to_words(expected_rows(3, 2, 2))


def test_trigger_latency_is_fixed_and_external_trigger_needs_enabling():
    latencies = set()
    for wait in (0, 3, 17):
        bench = Bench()
        bench.configure(6, 2, 2, E_STOP, ext=True)
        bench.step(wait)
        start = bench.cycle
        bench.dut.inp.ext_trigger = 1
        bench.run_until(lambda: bench.out)
        latencies.add(bench.out[0][0] - start)
    assert len(latencies) == 1 and latencies.pop() <= 9

    bench = Bench()
    bench.configure(6, 2, 2, E_STOP, ext=False)
    bench.dut.inp.ext_trigger = 1
    bench.step(50)
    assert bench.out == [] and bench.dut.status["armed"]


def test_backpressure_while_playing_is_flagged():
    bench = Bench()
    bench.configure(10, 1, 5, E_STOP)
    bench.trigger()
    bench.step(10)
    bench.dut.inp.tready = 0
    bench.step(2)
    bench.dut.inp.tready = 1
    bench.run_until(lambda: bench.dut.status["done"])
    assert bench.dut.status["stall"]
    bench.dut.inp.seq_ctrl &= ~CTRL_ARM
    bench.step(5)
    bench.configure(10, 1, 1, E_STOP)
    assert not bench.dut.status["stall"], "cleared by re-arming"


@pytest.mark.parametrize("num_samples,rows", [(32, 1), (33, 2), (1000, 32), (131072, 4096), (0, 8192)])
def test_legacy_mode_loops_like_dacramstreamer_without_a_trigger(num_samples, rows):
    bench = Bench()
    bench.dut.inp.num_samples = num_samples
    bench.dut.inp.enable = 1
    bench.step(3 * rows + 40)
    words = bench.words()
    assert len(words) >= 2 * rows
    assert all(word == row_value(n % rows) for n, word in enumerate(words))
    assert not bench.dut.status["seq_mode"]
    bench.dut.inp.enable = 0
    bench.step(8)
    bench.out.clear()
    bench.step(20)
    assert bench.out == []
