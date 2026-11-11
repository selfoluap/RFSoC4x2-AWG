"""Driver -> DDR4 image -> FPGA model -> DAC samples, for the full-rate modes.

Ties the storage format written by firmware/ddr4_player.py to the unpacker in DDR4Streamer.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hdl" / "model"))
sys.path.insert(0, str(ROOT / "tests"))

from ddr4_player_model import Bench, DDR4StreamerModel, MigModel, MigProfile  # noqa: E402
from firmware import ddr4_player as dp  # noqa: E402
from test_ddr4_player import make_player  # noqa: E402


@pytest.mark.parametrize("bits", [14, 12, 8])
def test_waveform_loaded_by_the_driver_comes_out_of_the_dac_port(bits):
    player, hardware, memory, _ = make_player(region_start=8192, region_bytes=1 << 18, memory_bytes=1 << 19)
    player.set_full_rate(bits)
    t = np.arange(20_000)
    wave = (30000 * np.sin(2 * np.pi * 0.0137 * t * (1 + t / 60_000))).astype(np.int16)
    segment = player.load(wave)
    player.arm()

    # the same DDR4 content and the same register values, given to the FPGA model
    bench = Bench(dut=DDR4StreamerModel(fifo_aw=12), mig=MigModel(MigProfile(refresh_stall=75)))
    image = memory.view(np.uint8)
    for index in range(len(image) // 64):
        bench.mig.words[index] = int.from_bytes(image[64 * index : 64 * index + 64].tobytes(), "little")
    ctl = bench.dut.ctl
    ctl.mode = ((hardware.ctrl >> 7 & 1) << 2) | (hardware.ctrl >> 4 & 3)
    ctl.base_page, ctl.num_words = hardware.base, hardware.words
    ctl.loop_count = hardware.ctrl >> dp.CTRL_LOOPS_SHIFT
    ctl.enable = 1
    assert bench.run_until(lambda: bench.dut.status["ready"])
    ctl.sw_trigger = 1
    assert bench.run_until(lambda: bench.dut.status["done"])
    bench.run_axis_cycles(40)

    played = np.array(bench.samples, dtype=np.uint16).view(np.int16)
    assert played.size == segment.padded_samples
    assert not bench.dut.status["underflow"]
    step = 1 << (16 - bits)
    if bits == 14:
        assert np.array_equal(played[: wave.size], wave & ~np.int16(3))
    else:
        assert np.abs(played[: wave.size].astype(np.int32) - wave).max() <= step // 2
    assert np.all(played[wave.size :] == 0)
