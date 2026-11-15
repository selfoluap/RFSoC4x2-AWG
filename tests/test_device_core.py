"""Tests for the device core, with the sequence players running on the FPGA model."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from firmware.device import DeviceCore, DeviceError  # noqa: E402
import test_ddr4_player as ddr4_rig  # noqa: E402
import test_sequencer as seq_rig  # noqa: E402


class LegacyPlayer:
    """Stand-in for overlay_controller.DacPlayer."""

    def __init__(self):
        self.capacity = 131072
        self.enabled = False
        self.loaded = None

    def load_waveform(self, data):
        self.loaded = np.array(data)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


class Controller:
    bitfile_path = "/home/xilinx/rfsocawg.bit"
    rf_clock_source = "internal"

    def __init__(self, seq=False, ddr4=False):
        self.dac0, self.dac2 = LegacyPlayer(), LegacyPlayer()
        self.rigs = {}
        if seq:
            self.seq = {}
            for channel in ("dac0", "dac2"):
                player, hardware, buffer = seq_rig.make_player()
                player.name = channel
                self.seq[channel] = player
                self.rigs[channel] = hardware
        if ddr4:
            self.ddr4 = {}
            for channel in ("dac0", "dac2"):
                player, hardware, _, _ = ddr4_rig.make_player()
                player.name = channel
                self.ddr4[channel] = player
                self.rigs["ddr4_" + channel] = hardware


def tone(n=64):
    return (1000 * np.sin(np.arange(n))).astype(np.int16)


def output(core, channel):
    return next(o for o in core.status()["outputs"] if o["channel"] == channel)


# ---------------------------------------------------------------------------- reference bitstream
def test_reference_bitstream_offers_loop_only():
    core = DeviceCore(Controller())
    info = core.info()
    assert info["capabilities"] == {"loop": True, "sequence": False, "long": False,
                                    "external_trigger": False, "adc": False, "analog_trigger": False}
    assert info["channels"][0] == {"channel": "dac0", "loop_samples": 131072}
    assert output(core, "dac0") == {"channel": "dac0", "mode": "none", "state": "idle", "block": 0,
                                    "loaded_samples": 0, "warnings": [], "buffer_low_water": None}

    core.load_loop("dac0", tone())
    assert output(core, "dac0")["mode"] == "loop" and output(core, "dac0")["loaded_samples"] == 64
    with pytest.raises(DeviceError, match="immediately"):
        core.arm("dac0", trigger="external")
    assert core.arm("dac0", trigger="immediate")["state"] == "running"
    assert core.controller.dac0.enabled and not core.controller.dac2.enabled
    assert core.stop("dac0")["state"] == "idle" and not core.controller.dac0.enabled

    with pytest.raises(DeviceError, match="RFSOC_AWG_SEQ"):
        core.load_sequence("dac0", np.zeros((2, 64), dtype=np.int16))
    with pytest.raises(DeviceError, match="RFSOC_AWG_DDR4"):
        core.load_long("dac0", tone(), sample_bits=14)


# ---------------------------------------------------------------------------- sequence bitstream
def test_sequence_plays_in_order_through_the_fpga_model():
    controller = Controller(seq=True)
    core = DeviceCore(controller)
    assert core.info()["capabilities"]["sequence"] and core.info()["capabilities"]["external_trigger"]
    core.load_sequence("dac2", np.zeros((3, 64), dtype=np.int16))            # 3 blocks of 2 rows
    assert core.arm("dac2", trigger="software", repeats=4)["state"] == "armed"
    assert controller.rigs["dac2"].played == []
    core.trigger("dac2")
    status = output(core, "dac2")
    assert status["state"] == "done" and status["mode"] == "sequence" and status["warnings"] == []
    expected = [row for block in range(3) for _ in range(4) for row in (2 * block, 2 * block + 1)]
    assert controller.rigs["dac2"].played == expected
    assert output(core, "dac0")["state"] == "idle"


def test_loop_uses_the_sequence_player_and_gains_a_trigger():
    controller = Controller(seq=True)
    core = DeviceCore(controller)
    core.load_loop("dac0", tone(70))                                         # padded to 96 samples
    assert core.arm("dac0", trigger="external")["state"] == "armed"
    assert controller.rigs["dac0"].played == []
    controller.rigs["dac0"].dut.inp.ext_trigger = 1
    assert output(core, "dac0")["state"] == "running"
    assert controller.rigs["dac0"].played[:6] == [0, 1, 2, 0, 1, 2]
    core.stop_all()
    assert output(core, "dac0")["state"] == "idle"
    assert not controller.dac0.enabled, "the legacy enable is never used on a sequence bitstream"


def test_immediate_trigger_hold_and_rearm():
    core = DeviceCore(Controller(seq=True))
    core.load_sequence("dac0", np.zeros((2, 32), dtype=np.int16))
    assert core.arm("dac0", trigger="immediate", repeats=2, end="hold")["state"] in ("running", "holding")
    held = output(core, "dac0")
    assert held["state"] == "holding" and held["block"] == 1
    core.stop("dac0")
    assert core.arm("dac0", trigger="immediate", repeats=1, end="stop")["state"] in ("running", "done")
    assert output(core, "dac0")["state"] == "done"


# ---------------------------------------------------------------------------- DDR4 bitstream
def test_long_waveform_full_rate_and_back_to_loop():
    controller = Controller(seq=True, ddr4=True)
    core = DeviceCore(controller)
    hardware = controller.rigs["ddr4_dac0"]
    loaded = core.load_long("dac0", (tone(1000) for _ in range(3)), sample_bits=14)
    assert loaded["mode"] == "long" and loaded["loaded_samples"] == 3000
    assert controller.ddr4["dac0"].sample_bits == 14
    assert core.arm("dac0", trigger="software", repeats=2)["state"] == "armed"
    assert core.trigger("dac0")["state"] == "done"
    assert output(core, "dac0")["buffer_low_water"] == 1900
    hardware.underflow = True
    assert "gaps" in output(core, "dac0")["warnings"][0]
    hardware.underflow = False

    with pytest.raises(DeviceError, match="two channels"):
        core.load_long("dac2", tone(), sample_bits=12)
    core.load_long("dac2", tone(), interpolation=8)                          # interpolated is fine
    with pytest.raises(ValueError, match="hold"):
        core.arm("dac2", end="hold")

    core.load_loop("dac0", tone())                                           # DAC back to on-chip
    assert not hardware.ctrl & ddr4_rig.dp.CTRL_SOURCE_DDR4
    assert output(core, "dac0")["mode"] == "loop"
    core.load_long("dac2", tone(), sample_bits=8)                            # now allowed


# ---------------------------------------------------------------------------- validation
def test_bad_requests_are_rejected_before_touching_the_hardware():
    core = DeviceCore(Controller(seq=True, ddr4=True))
    with pytest.raises(ValueError):
        core.load_loop("dac1", tone())
    with pytest.raises(ValueError):
        core.load_loop("dac0", tone().astype(np.float32))
    with pytest.raises(ValueError):
        core.load_sequence("dac0", tone())                                   # not 2D
    with pytest.raises(ValueError):
        core.load_long("dac0", tone())
    with pytest.raises(ValueError):
        core.load_long("dac0", tone(), sample_bits=14, interpolation=2)
    with pytest.raises(ValueError):
        core.load_long("dac0", tone(), sample_bits=16)
    with pytest.raises(DeviceError, match="nothing is loaded"):
        core.arm("dac0")
    core.load_loop("dac0", tone())
    for bad in ({"trigger": "analog"}, {"end": "pause"}, {"repeats": 0}):
        with pytest.raises(ValueError):
            core.arm("dac0", **bad)
    assert output(core, "dac0")["state"] == "idle"


# ---------------------------------------------------------------------------- web backend service
def test_web_backend_bench_test_goes_through_the_device_core():
    from backend.services.hardware import HardwareService

    controller = Controller(seq=True)
    core = DeviceCore(controller)
    service = HardwareService(device=core)
    assert service.initialized and service.adc is None and service.analog_trigger_info() is None

    service.load_dac(np.full(200, 40000.0), "dac0")                # clipped to int16
    assert service.last_signal.max() == 32767
    assert service.dacs_status() == {"dac0": {"enabled": False, "waveform_length": 200},
                                     "dac2": {"enabled": False, "waveform_length": 0}}
    service.set_dac_enabled("dac0", True)
    assert service.dac_status("dac0")["enabled"]
    assert output(core, "dac0")["mode"] == "loop" and output(core, "dac0")["state"] == "running"

    # a script taking over the channel through the same core is visible to the web page
    core.load_sequence("dac0", np.zeros((2, 64), dtype=np.int16))
    assert not service.dac_status("dac0")["enabled"]
    service.set_dac_enabled("dac0", False)
    with pytest.raises(RuntimeError, match="no ADC capture"):
        service.capture_adc("A", 64)
