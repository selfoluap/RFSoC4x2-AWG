"""Off-board tests for the DDR4 player driver, using fake GPIO channels."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firmware import ddr4_player as dp  # noqa: E402


class FakeHardware:
    """Just enough of DDR4Streamer's control/status behaviour to drive the driver."""

    def __init__(self):
        self.writes = []            # (register, value) in order
        self.ctrl = 0
        self.base = 0
        self.words = 0
        self.ready = self.running = self.done = False
        self.underflow = False
        self.finish_immediately = True

    def channel(self, register):
        hardware = self

        class Channel:
            def write(self, value, mask):
                hardware.write(register, value & mask)

            def read(self):
                return hardware.status()

        return Channel()

    def write(self, register, value):
        self.writes.append((register, value))
        if register == "base":
            self.base = value
        elif register == "words":
            self.words = value
        elif register == "ctrl":
            rising = value & ~self.ctrl
            self.ctrl = value
            if not value & dp.CTRL_ENABLE:
                self.ready = self.running = self.done = False
            elif rising & dp.CTRL_ENABLE:
                self.ready = True
            if rising & dp.CTRL_SW_TRIGGER and self.ready:
                self.ready = False
                self.running = not self.finish_immediately
                self.done = self.finish_immediately

    def status(self):
        enabled = bool(self.ctrl & dp.CTRL_ENABLE)
        return (
            (self.ready << 1)
            | (self.running << 2)
            | (self.done << 3)
            | (self.underflow << 4)
            | (enabled << 7)
            | (1900 << 16)
        )


class FakeRfdcBlock:
    InterpolationFactor = 1
    FabWrVldWords = 16


def make_player(region_start=0, region_bytes=1 << 20, memory_bytes=1 << 21):
    hardware = FakeHardware()
    memory = np.full(memory_bytes // 2, 0x7777, dtype=np.int16)
    block = FakeRfdcBlock()
    player = dp.Ddr4Player(
        name="dac0",
        memory=memory,
        region_start=region_start,
        region_bytes=region_bytes,
        ctrl_channel=hardware.channel("ctrl"),
        status_channel=hardware.channel("status"),
        base_channel=hardware.channel("base"),
        words_channel=hardware.channel("words"),
        rfdc_block=block,
        sleep=lambda seconds: None,
    )
    return player, hardware, memory, block


def test_rate_table_gives_the_one_to_two_second_target():
    four_gib_samples = (4 << 30) // 2
    assert four_gib_samples / dp.input_rate(8) == pytest.approx(1.7476, abs=1e-3)
    assert four_gib_samples / dp.input_rate(4) == pytest.approx(0.8738, abs=1e-3)
    assert four_gib_samples / dp.input_rate(16) == pytest.approx(3.4953, abs=1e-3)
    assert dp.usable_bandwidth(8) == pytest.approx(491.52e6)
    with pytest.raises(ValueError):
        dp.input_rate(3)


def test_load_pads_the_last_word_and_reports_the_segment():
    player, _, memory, _ = make_player(region_start=8192)
    data = np.arange(1, 71, dtype=np.int16)
    segment = player.load(data)
    assert segment == dp.Segment(base_page=2, num_words=3, num_samples=70)
    first = 8192 // 2
    assert np.array_equal(memory[first : first + 70], data)
    assert np.all(memory[first + 70 : first + 96] == 0)
    assert memory[first + 96] == 0x7777 and memory[first - 1] == 0x7777


def test_load_accepts_chunks_so_long_waveforms_never_sit_in_ram():
    player, _, memory, _ = make_player()
    chunks = (np.full(1000, n, dtype=np.int16) for n in range(5))
    segment = player.load(chunks, offset_bytes=4096)
    assert segment.base_page == 1 and segment.num_samples == 5000
    assert segment.num_words == 157
    assert np.array_equal(memory[2048 : 2048 + 5000], np.repeat(np.arange(5, dtype=np.int16), 1000))


def test_load_rejects_bad_input():
    player, _, _, _ = make_player(region_bytes=8192)
    with pytest.raises(ValueError, match="exceeds"):
        player.load(np.zeros(4097, dtype=np.int16))
    with pytest.raises(TypeError):
        player.load(np.zeros(16, dtype=np.float64))
    with pytest.raises(ValueError):
        player.load(np.zeros(0, dtype=np.int16))
    with pytest.raises(ValueError):
        player.load(np.zeros(16, dtype=np.int16), offset_bytes=100)
    assert player.load(np.zeros(4096, dtype=np.int16)).num_words == 128


def test_set_interpolation_switches_rfdc_and_mux():
    player, hardware, _, block = make_player()
    player.set_interpolation(8)
    assert block.InterpolationFactor == 8 and block.FabWrVldWords == 2
    assert hardware.ctrl & dp.CTRL_SOURCE_DDR4
    assert (hardware.ctrl & dp.CTRL_RATE_MASK) >> dp.CTRL_RATE_SHIFT == 2
    player.set_interpolation(2)
    assert block.FabWrVldWords == 8
    assert (hardware.ctrl & dp.CTRL_RATE_MASK) >> dp.CTRL_RATE_SHIFT == 0
    player.select_bram_player()
    assert block.InterpolationFactor == 1 and block.FabWrVldWords == 16
    assert not hardware.ctrl & dp.CTRL_SOURCE_DDR4


def test_arm_writes_configuration_before_enable():
    player, hardware, _, _ = make_player(region_start=4096)
    player.set_interpolation(4)
    segment = player.load(np.ones(640, dtype=np.int16))
    hardware.writes.clear()
    player.arm(loops=3, external_trigger=True)

    registers = [register for register, _ in hardware.writes]
    enable_index = next(
        i for i, (register, value) in enumerate(hardware.writes)
        if register == "ctrl" and value & dp.CTRL_ENABLE
    )
    assert registers.index("base") < enable_index and registers.index("words") < enable_index
    assert hardware.base == segment.base_page == 1 and hardware.words == 20
    assert hardware.ctrl >> dp.CTRL_LOOPS_SHIFT == 3
    assert hardware.ctrl & dp.CTRL_EXT_TRIG_EN and not hardware.ctrl & dp.CTRL_LOOP_FOREVER
    # the control word written just before enable already carries the final configuration
    assert hardware.writes[enable_index - 1] == ("ctrl", hardware.ctrl & ~dp.CTRL_ENABLE)
    assert player.status()["ready"]


def test_play_and_wait_done():
    player, hardware, _, _ = make_player()
    player.set_interpolation(8)
    player.load(np.ones(64, dtype=np.int16))
    player.play()
    assert not hardware.ctrl & dp.CTRL_SW_TRIGGER, "trigger must be a pulse"
    assert player.wait_done()["done"]


def test_wait_done_reports_underflow():
    player, hardware, _, _ = make_player()
    player.set_interpolation(2)
    player.load(np.ones(64, dtype=np.int16))
    player.play()
    hardware.underflow = True
    with pytest.raises(dp.Ddr4PlayerError, match="underflow"):
        player.wait_done()


def test_misuse_is_rejected():
    player, hardware, _, _ = make_player()
    with pytest.raises(dp.Ddr4PlayerError, match="load"):
        player.arm()
    player.load(np.ones(64, dtype=np.int16))
    with pytest.raises(dp.Ddr4PlayerError, match="set_interpolation"):
        player.arm()
    player.set_interpolation(8)
    with pytest.raises(dp.Ddr4PlayerError, match="not ready"):
        player.trigger()
    with pytest.raises(ValueError):
        player.arm(dp.Segment(base_page=4096, num_words=1, num_samples=32))
    with pytest.raises(ValueError):
        player.arm(loops=0)
    hardware.finish_immediately = False
    player.play(forever=True)
    assert player.status()["running"] and hardware.ctrl & dp.CTRL_LOOP_FOREVER
    player.stop()
    assert not hardware.ctrl & dp.CTRL_ENABLE


def test_probe_detects_aliased_upper_half():
    class Aliased:
        """8 GiB-style window whose upper half mirrors the lower half."""

        def __init__(self, real_samples):
            self.real = np.zeros(real_samples, dtype=np.int16)
            self.size = 2 * real_samples

        def __getitem__(self, index):
            return self.real[index % self.real.size]

        def __setitem__(self, index, value):
            self.real[index % self.real.size] = value

    aliased = Aliased(1 << 12)
    aliased[0] = 111
    assert dp.probe_ddr4_bytes(aliased, 1 << 14) == 1 << 13
    assert aliased[0] == 111

    real = np.zeros(1 << 13, dtype=np.int16)
    real[0], real[1 << 12] = 5, 6
    assert dp.probe_ddr4_bytes(real, 1 << 14) == 1 << 14
    assert (real[0], real[1 << 12]) == (5, 6)


def test_attach_is_a_no_op_without_ddr4():
    class Overlay:
        mem_dict = {"hier_dac_play/axi_bram_ctrl_0": {}}

    assert dp.attach(Overlay()) == {}


# ---------------------------------------------------------------------------- full-rate modes
def unpack(stream: np.ndarray, bits: int, count: int) -> np.ndarray:
    """Reference decoder, written independently of pack_samples: what the FPGA does."""
    value = int.from_bytes(stream.tobytes(), "little")
    out = np.empty(count, dtype=np.int16)
    for i in range(count):
        code = (value >> (bits * i)) & ((1 << bits) - 1)
        out[i] = np.array(code << (16 - bits), dtype=np.uint16).view(np.int16)
    return out


def test_full_rate_capacity_is_a_few_hundred_milliseconds():
    four_gib_bits = (4 << 30) * 8
    assert four_gib_bits // 14 / dp.input_rate(1) == pytest.approx(0.2497, abs=1e-3)
    assert four_gib_bits // 12 / dp.input_rate(1) == pytest.approx(0.2913, abs=1e-3)
    assert four_gib_bits // 8 / dp.input_rate(1) == pytest.approx(0.4369, abs=1e-3)
    assert dp.usable_bandwidth(1) == pytest.approx(4915.2e6)


@pytest.mark.parametrize("bits", [14, 12, 8])
def test_pack_samples_round_trips_through_the_fpga_format(bits):
    rng = np.random.default_rng(bits)
    data = rng.integers(-32768, 32768, size=4096, dtype=np.int16)
    data[:4] = [32767, -32768, 0, -1]
    decoded = unpack(dp.pack_samples(data, bits), bits, data.size)
    step = 1 << (16 - bits)
    if bits == 14:
        assert np.array_equal(decoded, data & ~np.int16(step - 1))   # top 14 bits, exactly
    else:
        error = decoded.astype(np.int32) - data
        assert np.all(np.abs(error) <= step // 2) or np.all(error[data > 32767 - step] <= 0)
        assert np.abs(error[data <= 32767 - step]).max() <= step // 2
        assert decoded.max() <= 32767 - step + 1 and decoded.min() == -32768


@pytest.mark.parametrize("bits,words_per_group", [(14, 7), (12, 3), (8, 1)])
def test_full_rate_load_packs_pads_to_whole_groups_and_handles_odd_chunks(bits, words_per_group):
    player, _, memory, _ = make_player(region_start=4096)
    player.set_full_rate(bits)
    rng = np.random.default_rng(1)
    data = rng.integers(-32768, 32768, size=1001, dtype=np.int16)
    chunks = [data[:3], data[3:10], data[10:513], data[513:]]      # not multiples of 4
    segment = player.load(iter(chunks))

    group = dp.GROUP_SAMPLES[bits]
    padded = -(-1001 // group) * group
    assert segment.bits == bits and segment.num_samples == 1001 and segment.base_page == 1
    assert segment.padded_samples == padded
    assert segment.num_words % words_per_group == 0
    assert segment.num_words == padded * bits // 512

    stored = memory.view(np.uint8)[4096 : 4096 + segment.num_words * 64]
    decoded = unpack(stored, bits, padded)
    reference = unpack(dp.pack_samples(np.resize(data, 1004)[:1004], bits), bits, 1004)[:1001]
    assert np.array_equal(decoded[:1001], reference)
    assert np.all(decoded[1001:] == 0)
    assert memory.view(np.uint8)[4096 + segment.num_words * 64] == 0x77, "wrote past the segment"


def test_set_full_rate_configures_rfdc_and_mode_bits():
    player, hardware, _, block = make_player()
    player.set_interpolation(8)
    player.set_full_rate(12)
    assert block.InterpolationFactor == 1 and block.FabWrVldWords == 16
    assert hardware.ctrl & dp.CTRL_FULL_RATE and hardware.ctrl & dp.CTRL_SOURCE_DDR4
    assert (hardware.ctrl & dp.CTRL_RATE_MASK) >> dp.CTRL_RATE_SHIFT == 1
    assert player.max_duration() == pytest.approx((1 << 20) * 8 // 12 / 9.8304e9)
    player.load(np.ones(128, dtype=np.int16))
    player.play(loops=2)
    assert hardware.ctrl & dp.CTRL_FULL_RATE and hardware.ctrl >> dp.CTRL_LOOPS_SHIFT == 2
    assert player.wait_done()["buffer_low_water"] == 1900
    player.set_interpolation(4)
    assert not hardware.ctrl & dp.CTRL_FULL_RATE
    with pytest.raises(ValueError):
        player.set_full_rate(16)


def test_segment_from_another_mode_is_rejected():
    player, _, _, _ = make_player()
    player.set_interpolation(8)
    segment = player.load(np.ones(64, dtype=np.int16))
    player.set_full_rate(14)
    with pytest.raises(dp.Ddr4PlayerError, match="load"):
        player.arm(segment)


def test_full_rate_load_respects_the_region():
    player, _, _, _ = make_player(region_bytes=4096)
    player.set_full_rate(14)
    assert player.capacity_samples == 4096 * 8 // 14
    with pytest.raises(ValueError, match="exceeds"):
        player.load(np.zeros(2400, dtype=np.int16))
    assert player.load(np.zeros(2304, dtype=np.int16)).num_words == 63
