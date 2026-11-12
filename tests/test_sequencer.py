"""Off-board tests for the sequence-mode driver and the Doppler staircase helper."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hdl" / "model"))

from firmware import sequencer as sq  # noqa: E402
from firmware import signals  # noqa: E402
from seq_streamer_model import SeqStreamerModel  # noqa: E402


class ModelHardware:
    """GPIO channels wired to the register-transfer model of DACSeqStreamer.

    Every register access advances the model by a few clock cycles, so the driver is tested
    against the same logic the FPGA runs. The model's memory is addressed by row; `played`
    collects the row index of every row that reaches the output.
    """

    def __init__(self):
        self.dut = SeqStreamerModel()
        self.played = []
        self.writes = []
        self.advance(8)

    def advance(self, cycles):
        from seq_streamer_model import row_value

        if not hasattr(self, "_rows"):
            self._rows = {row_value(r): r for r in range(self.dut.ROWS)}
        for _ in range(cycles):
            self.dut.tick()
            if self.dut.axis_tdata:
                self.played.append(self._rows[self.dut.axis_tdata])

    def channel(self, register):
        hardware = self

        class Channel:
            def write(self, value, mask):
                hardware.writes.append((register, value))
                setattr(hardware.dut.inp, register, value & mask)
                hardware.advance(6)

            def read(self):
                hardware.advance(200)
                s = hardware.dut.status
                return (s["armed"] | s["running"] << 1 | s["done"] << 2 | s["holding"] << 3
                        | s["stall"] << 4 | s["seq_mode"] << 5 | s["block"] << 16)

        return Channel()


def make_player(samples=262144):
    hardware = ModelHardware()
    buffer = np.full(samples, 0x5555, dtype=np.int16)
    player = sq.SeqPlayer(
        "dac0", buffer,
        ctrl_channel=hardware.channel("seq_ctrl"),
        status_channel=hardware.channel("status"),
        lens_channel=hardware.channel("seq_lens"),
        repeats_channel=hardware.channel("seq_repeats"),
        sleep=lambda seconds: None,
    )
    return player, hardware, buffer


def test_load_blocks_writes_them_back_to_back():
    player, _, buffer = make_player()
    blocks = np.arange(3 * 64, dtype=np.int16).reshape(3, 64)
    player.load_blocks(blocks)
    assert np.array_equal(buffer[:192], blocks.reshape(-1))
    assert buffer[192] == 0x5555
    assert (player.n_blocks, player.block_samples) == (3, 64)


def test_load_blocks_rejects_bad_input():
    player, _, _ = make_player(samples=1024)
    with pytest.raises(ValueError, match="multiple of 32"):
        player.load_blocks(np.zeros((2, 100), dtype=np.int16))
    with pytest.raises(ValueError, match="holds 1024"):
        player.load_blocks(np.zeros((5, 256), dtype=np.int16))
    with pytest.raises(TypeError):
        player.load_blocks(np.zeros((2, 64), dtype=np.float32))
    with pytest.raises(sq.SequencerError, match="load_blocks"):
        player.arm()


def test_sequence_plays_through_the_fpga_model_in_the_right_order():
    player, hardware, _ = make_player()
    player.load_blocks(np.zeros((4, 96), dtype=np.int16))       # 4 blocks of 3 rows
    player.arm(repeats=5, end="stop")
    assert hardware.played == [] and player.status()["armed"]
    player.trigger()
    status = player.wait_done()
    assert status["done"] and not status["stall"]
    expected = [row for block in range(4) for _ in range(5) for row in range(3 * block, 3 * block + 3)]
    assert hardware.played == expected
    assert player.duration() == pytest.approx(4 * 5 * 96 / 9.8304e9)


def test_configuration_is_written_before_arm_and_trigger_is_a_pulse():
    player, hardware, _ = make_player()
    player.load_blocks(np.zeros((2, 64), dtype=np.int16))
    hardware.writes.clear()
    player.arm(repeats=7, end="hold", external_trigger=True)
    names = [name for name, _ in hardware.writes]
    arm_index = next(i for i, (n, v) in enumerate(hardware.writes) if n == "seq_ctrl" and v & sq.CTRL_ARM)
    assert names.index("seq_lens") < arm_index and names.index("seq_repeats") < arm_index
    assert hardware.dut.inp.seq_lens == (2 << 16) | 2 and hardware.dut.inp.seq_repeats == 7
    ctrl = hardware.dut.inp.seq_ctrl
    assert ctrl & sq.CTRL_EXT_TRIG_EN and ctrl >> sq.CTRL_END_SHIFT & 3 == 1
    player.trigger()
    assert not hardware.dut.inp.seq_ctrl & sq.CTRL_SW_TRIGGER
    assert player.status()["holding"]
    with pytest.raises(sq.SequencerError, match="never finish"):
        player.wait_done()
    player.stop()
    assert not player.status()["running"]


def test_play_loop_pads_and_loops():
    player, hardware, buffer = make_player()
    player.play_loop(np.ones(70, dtype=np.int16))
    assert player.block_samples == 96 and np.all(buffer[70:96] == 0)
    assert player.status()["running"]
    assert hardware.played[:9] == [0, 1, 2] * 3
    with pytest.raises(sq.SequencerError, match="stop"):
        player.load_blocks(np.zeros(64, dtype=np.int16))


def test_attach_is_a_no_op_without_the_sequencer():
    class Overlay:
        ip_dict = {"gpio_control/axi_gpio_dac": {}}

    assert sq.attach(Overlay()) == {}


# ---------------------------------------------------------------------------- Doppler staircase
FS = 9.8304e9


def test_staircase_reproduces_the_design_note_numbers():
    blocks, info = signals.serrodyne_staircase(
        [1, 5, 3], [-1.33e9, 0.0, 0.87e9], 1952, 120, amplitude=30000, sample_rate=FS
    )
    assert blocks.shape == (120, 1952) and blocks.dtype == np.int16
    assert info["grid_hz"] == pytest.approx(5.036e6, rel=1e-3)
    assert info["pattern_period_s"] == pytest.approx(198.57e-9, rel=1e-4)
    assert info["snapped_freqs_hz"] == pytest.approx([-1329.5e6, 0.0, 871.2e6], abs=0.1e6)
    assert info["sweep_hz"] == pytest.approx(599.3e6, abs=0.1e6)
    assert info["final_freqs_hz"] == pytest.approx([-730.2e6, 599.3e6, 1470.5e6], abs=0.1e6)
    assert blocks.size * 2 == 468480                       # 457.5 KiB per channel
    assert 120 * 168 * 1952 / FS == pytest.approx(4.003e-3, rel=1e-3)

    player, _, _ = make_player()
    player.load_blocks(blocks)                             # fits the 8192-row memory
    assert player.n_blocks == 120 and player.block_samples // 32 == 61


def test_staircase_blocks_carry_the_shifted_ramp_frequencies():
    blocks, info = signals.serrodyne_staircase(
        [1, 5, 3], [-1.33e9, 0.0, 0.87e9], 1952, 120, amplitude=30000, sample_rate=FS
    )
    assert np.all(blocks[0][217 : 217 + 1084] == 0), "middle ramp is at exactly 0 Hz in block 0"

    def ramp_frequency(segment):
        """Sawtooth frequency from the mean slope between wraps."""
        steps = np.diff(segment.astype(np.float64))
        slope = np.median(steps)                           # counts per sample; wraps are outliers
        return slope * FS / 60000.0                        # 60000 counts = one full ramp

    for k in (10, 60, 119):
        third = blocks[k][217 + 1084 :]
        assert ramp_frequency(third) == pytest.approx(871.2e6 + k * info["grid_hz"], rel=0.02)
        first = blocks[k][:217]
        assert ramp_frequency(first) == pytest.approx(-1329.5e6 + k * info["grid_hz"], rel=0.02)


def test_staircase_options():
    down, info = signals.serrodyne_staircase(
        [1, 5, 3], [-1.33e9, 0.0, 0.87e9], 1952, 120, amplitude=30000, sample_rate=FS, step_sign=-1
    )
    assert info["final_freqs_hz"] == pytest.approx([-1928.8e6, -599.3e6, 271.9e6], abs=0.1e6)
    halved, _ = signals.serrodyne_staircase(
        [1, 5, 3], [-1.33e9, 0.0, 0.87e9], 1952, 4, amplitude=30000, sample_rate=FS,
        predistort=lambda block: block / 2,
    )
    plain, _ = signals.serrodyne_staircase(
        [1, 5, 3], [-1.33e9, 0.0, 0.87e9], 1952, 4, amplitude=30000, sample_rate=FS
    )
    assert np.abs(halved.astype(np.int32) * 2 - plain).max() <= 2
    with pytest.raises(ValueError, match="multiple of 32"):
        signals.serrodyne_staircase([1], [1e9], 1000, 2, amplitude=1, sample_rate=FS)
