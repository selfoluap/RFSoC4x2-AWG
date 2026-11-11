"""Logic tests for the DDR4 player, run against the register-transfer model."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hdl" / "model"))

from ddr4_player_model import (  # noqa: E402
    RATE_MODES,
    Bench,
    DDR4StreamerModel,
    MigModel,
    PACKED_MODES,
    MigProfile,
    expected_packed_samples,
    expected_samples,
)


def make_bench(fifo_aw=10, profile=None, seed=1, **kwargs):
    return Bench(
        dut=DDR4StreamerModel(fifo_aw=fifo_aw),
        mig=MigModel(profile or MigProfile(), seed=seed),
        **kwargs,
    )


def arm(bench, **controls):
    for key, value in controls.items():
        setattr(bench.dut.ctl, key, value)
    bench.dut.ctl.enable = 1
    assert bench.run_until(lambda: bench.dut.status["ready"]), "never became ready"


def trigger(bench):
    bench.dut.ctl.sw_trigger = 1
    bench.run_axis_cycles(4)
    bench.dut.ctl.sw_trigger = 0


def play_to_done(bench, **controls):
    arm(bench, **controls)
    trigger(bench)
    assert bench.run_until(lambda: bench.dut.status["done"]), "never finished"
    bench.run_axis_cycles(40)


def assert_clean(bench):
    status = bench.dut.status
    assert not status["underflow"]
    assert not status["read_error"]
    assert not bench.dut.fifo.overflow
    assert not bench.mig.bad_address


@pytest.mark.parametrize("mode", sorted(RATE_MODES))
@pytest.mark.parametrize("num_words", [1, 2, 63, 64, 65, 200, 2500])
def test_one_shot_plays_exactly_the_stored_samples(mode, num_words):
    bench = make_bench()
    play_to_done(bench, mode=mode, num_words=num_words, base_page=5)
    assert bench.samples == expected_samples(5, num_words, 1)
    assert_clean(bench)
    assert bench.dut.status["done"] and not bench.dut.status["running"]


@pytest.mark.parametrize("num_words,loops", [(100, 3), (70, 4), (1, 5), (64, 2), (1500, 2)])
def test_counted_loops_are_seamless(num_words, loops):
    bench = make_bench()
    play_to_done(bench, mode=1, num_words=num_words, base_page=9, loop_count=loops)
    assert bench.samples == expected_samples(9, num_words, loops)
    assert_clean(bench)


def test_loop_forever_then_abort_then_replay_other_waveform():
    bench = make_bench()
    arm(bench, mode=0, num_words=130, base_page=2, loop_forever=1)
    trigger(bench)
    bench.run_axis_cycles(3000)
    assert bench.dut.status["running"]
    one_pass = expected_samples(2, 130, 1)
    assert len(bench.samples) > 5 * len(one_pass)
    for i, value in enumerate(bench.samples):
        assert value == one_pass[i % len(one_pass)]
    assert_clean(bench)

    bench.dut.ctl.enable = 0
    assert bench.run_until(lambda: not bench.dut.status["busy"] and not bench.dut.status["armed"])
    bench.run_axis_cycles(100)
    assert bench.dut.m_axis_tdata == 0

    bench.samples.clear()
    play_to_done(bench, mode=2, num_words=77, base_page=40, loop_forever=0, loop_count=1)
    assert bench.samples == expected_samples(40, 77, 1)
    assert_clean(bench)


def test_downstream_backpressure_does_not_corrupt_the_stream():
    bench = make_bench(tready_gap_probability=0.25)
    play_to_done(bench, mode=0, num_words=900, base_page=1)
    assert bench.samples == expected_samples(1, 900, 1)
    assert_clean(bench)


def test_irregular_ddr4_timing_does_not_corrupt_the_stream():
    profile = MigProfile(latency=40, max_pending=2, gap_probability=0.3, ar_gap_probability=0.5)
    bench = make_bench(profile=profile, seed=11)
    play_to_done(bench, mode=1, num_words=3000, base_page=3)
    assert bench.samples == expected_samples(3, 3000, 1)
    assert_clean(bench)


def test_trigger_before_ready_is_flagged_and_ignored():
    bench = make_bench()
    bench.dut.ctl.num_words = 300
    bench.dut.ctl.enable = 1
    assert bench.run_until(lambda: bench.dut.status["armed"])
    trigger(bench)
    arm(bench)
    bench.run_axis_cycles(500)
    assert bench.dut.status["trig_early"]
    assert bench.dut.status["ready"] and not bench.dut.status["running"]
    assert bench.samples == []
    trigger(bench)
    assert bench.run_until(lambda: bench.dut.status["done"])
    bench.run_axis_cycles(40)
    assert bench.samples == expected_samples(0, 300, 1)


def test_external_trigger_only_when_enabled():
    bench = make_bench()
    arm(bench, num_words=50, ext_trig_en=0)
    bench.dut.ctl.ext_trigger = 1
    bench.run_axis_cycles(50)
    bench.dut.ctl.ext_trigger = 0
    bench.run_axis_cycles(50)
    assert not bench.dut.status["running"] and bench.samples == []

    bench.dut.ctl.ext_trig_en = 1
    bench.run_axis_cycles(10)
    bench.dut.ctl.ext_trigger = 1
    assert bench.run_until(lambda: bench.dut.status["done"])
    bench.run_axis_cycles(40)
    assert bench.samples == expected_samples(0, 50, 1)


def test_trigger_latency_is_fixed():
    latencies = set()
    for seed, num_words, profile in [
        (1, 40, MigProfile()),
        (2, 700, MigProfile(latency=60, gap_probability=0.2)),
        (3, 2000, MigProfile(latency=10, ar_gap_probability=0.4)),
    ]:
        bench = make_bench(profile=profile, seed=seed)
        arm(bench, mode=2, num_words=num_words)
        bench.run_axis_cycles(seed * 13)
        start = bench.axis_cycles
        bench.dut.ctl.sw_trigger = 1
        assert bench.run_until(lambda: bench.first_data_cycle is not None)
        latencies.add(bench.first_data_cycle - start)
    assert len(latencies) == 1, latencies
    assert latencies.pop() <= 6


def test_underflow_is_detected_when_the_ddr4_stalls_too_long():
    profile = MigProfile(long_stall_period=60_000, long_stall=20_000)
    bench = make_bench(profile=profile)
    play_to_done(bench, mode=0, num_words=50_000, base_page=0)
    assert bench.dut.status["underflow"]
    # the stream is late, but nothing is skipped or duplicated
    assert bench.samples == expected_samples(0, 50_000, 1)


def test_short_enable_glitch_recovers():
    bench = make_bench()
    arm(bench, mode=1, num_words=400, base_page=7, loop_forever=1)
    trigger(bench)
    bench.run_axis_cycles(500)
    bench.dut.ctl.enable = 0
    bench.run_axis_cycles(3)
    bench.dut.ctl.enable = 1
    bench.dut.ctl.loop_forever = 0
    bench.run_axis_cycles(30)
    bench.samples.clear()
    assert bench.run_until(lambda: bench.dut.status["ready"])
    assert bench.samples == [], "nothing may be played between abort and the next trigger"
    trigger(bench)
    assert bench.run_until(lambda: bench.dut.status["done"])
    bench.run_axis_cycles(40)
    assert bench.samples == expected_samples(7, 400, 1)
    assert not bench.dut.fifo.overflow


@pytest.mark.parametrize("mode", sorted(RATE_MODES))
def test_default_buffer_rides_through_refresh_stalls(mode):
    bench = make_bench(fifo_aw=11, profile=MigProfile())
    bench.record = False
    arm(bench, mode=mode, num_words=1 << 20, loop_forever=1)
    trigger(bench)
    bench.run_axis_cycles(120_000)
    assert not bench.dut.status["underflow"]
    assert not bench.dut.fifo.overflow
    assert bench.min_fill_running > 3 * (1 << 11) // 4


# ---------------------------------------------------------------------------- full-rate modes
# A DDR4 controller that pauses 0.25 us every 7.8 us (3 % dead time).
LIGHT_REFRESH = MigProfile(refresh_stall=75)


@pytest.mark.parametrize("mode", sorted(PACKED_MODES))
@pytest.mark.parametrize("groups", [1, 2, 9, 10, 37, 150])
def test_full_rate_unpacks_every_stored_sample(mode, groups):
    num_words = groups * PACKED_MODES[mode][2]
    bench = make_bench(fifo_aw=12, profile=LIGHT_REFRESH)
    play_to_done(bench, mode=mode, num_words=num_words, base_page=6)
    assert bench.samples == expected_packed_samples(6, num_words, 1, mode)
    assert len(bench.samples) == groups * PACKED_MODES[mode][1] * 32
    assert_clean(bench)


@pytest.mark.parametrize("mode", sorted(PACKED_MODES))
def test_full_rate_loops_are_seamless(mode):
    num_words = 21
    bench = make_bench(fifo_aw=12, profile=LIGHT_REFRESH)
    play_to_done(bench, mode=mode, num_words=num_words, base_page=3, loop_count=4)
    assert bench.samples == expected_packed_samples(3, num_words, 4, mode)
    assert_clean(bench)


def test_full_rate_survives_backpressure_and_irregular_ddr4():
    profile = MigProfile(refresh_stall=0, latency=40, gap_probability=0.05, ar_gap_probability=0.5)
    bench = make_bench(fifo_aw=12, profile=profile, seed=5, tready_gap_probability=0.2)
    play_to_done(bench, mode=4, num_words=7 * 400, base_page=1)
    assert bench.samples == expected_packed_samples(1, 7 * 400, 1, 4)
    assert_clean(bench)


def test_full_rate_underflow_is_flagged_and_loses_no_sample():
    # 34 % dead time: more than the 14-bit mode can absorb
    bench = make_bench(fifo_aw=10, profile=MigProfile(refresh_stall=800))
    play_to_done(bench, mode=4, num_words=7 * 1200, base_page=0)
    assert bench.dut.status["underflow"]
    assert bench.dut.status["low_water"] < 16
    assert bench.samples == expected_packed_samples(0, 7 * 1200, 1, 4)


def test_low_water_mark_reports_the_margin_and_clears_on_rearm():
    bench = make_bench(fifo_aw=12, profile=LIGHT_REFRESH)
    play_to_done(bench, mode=5, num_words=3 * 2500, base_page=0)
    low = bench.dut.status["low_water"]
    assert 3000 < low <= 4096, low
    bench.dut.ctl.enable = 0
    assert bench.run_until(lambda: not bench.dut.status["busy"])
    assert bench.dut.status["low_water"] == low, "stays readable after stop"
    arm(bench, mode=5, num_words=3)
    assert bench.dut.status["low_water"] == 8191, "cleared by re-arming; too short to measure"


def test_full_rate_trigger_latency_is_fixed():
    latencies = set()
    for seed, groups in [(1, 5), (2, 100), (3, 600)]:
        bench = make_bench(fifo_aw=12, profile=LIGHT_REFRESH, seed=seed)
        arm(bench, mode=4, num_words=7 * groups)
        bench.run_axis_cycles(seed * 17)
        start = bench.axis_cycles
        bench.dut.ctl.sw_trigger = 1
        assert bench.run_until(lambda: bench.first_data_cycle is not None)
        latencies.add(bench.first_data_cycle - start)
    assert len(latencies) == 1 and latencies.pop() <= 7
