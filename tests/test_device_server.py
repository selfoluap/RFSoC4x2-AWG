"""Checks that need no gRPC: message translation, and that the .proto, the translation tables,
the servicer template and the device core agree with each other."""

import ast
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from device_server import convert  # noqa: E402
from firmware.device import DeviceCore  # noqa: E402
from test_device_core import Controller  # noqa: E402

PROTO = (ROOT / "device_server/protos/h2pcontrol/rfsocawg/v1/rfsoc_awg.proto").read_text()


def proto_enum(name):
    body = re.search(rf"enum {name} \{{(.*?)\}}", PROTO, re.S).group(1)
    return {key: int(value) for key, value in re.findall(r"(\w+) = (\d+);", body)}


def proto_fields(message):
    body = re.search(rf"message {message} \{{(.*?)\n\}}", PROTO, re.S).group(1)
    return set(re.findall(r"(\w+) = \d+;", body))


def test_enum_numbers_match_the_proto():
    assert {v: f"CHANNEL_{n.upper()}" for v, n in convert.CHANNEL.items()} == {
        v: k for k, v in proto_enum("Channel").items() if v}
    modes = proto_enum("PlaybackMode")
    assert convert.MODE == {"none": modes["PLAYBACK_MODE_UNSPECIFIED"], "loop": modes["PLAYBACK_MODE_LOOP"],
                            "sequence": modes["PLAYBACK_MODE_SEQUENCE"], "long": modes["PLAYBACK_MODE_LONG"]}
    states = proto_enum("OutputState")
    assert convert.STATE == {name: states[f"OUTPUT_STATE_{name.upper()}"] for name in convert.STATE}
    triggers = proto_enum("TriggerSource")
    assert {triggers[f"TRIGGER_SOURCE_{n.upper()}"]: n for n in ("immediate", "software", "external")}.items() \
        <= convert.TRIGGER.items()
    assert convert.TRIGGER[0] == "software"
    ends = proto_enum("EndMode")
    assert {ends[f"END_MODE_{n.upper()}"]: n for n in ("stop", "hold", "loop")}.items() <= convert.END.items()
    assert convert.END[0] == "stop"


def test_translated_dicts_use_exactly_the_message_fields():
    core = DeviceCore(Controller(seq=True, ddr4=True))
    core.load_long("dac0", np.ones(64, dtype=np.int16), sample_bits=14)
    core.arm("dac0", trigger="immediate")
    statuses = core.status()["outputs"]
    assert set(convert.output_status_fields(statuses[0])) == proto_fields("OutputStatus")
    assert set(convert.output_status_fields(statuses[1])) == proto_fields("OutputStatus") - {"buffer_low_water"}
    info = core.info()
    assert set(info["capabilities"]) == proto_fields("Capabilities")
    assert set(convert.channel_info_fields(info["channels"][0])) == proto_fields("ChannelInfo")
    assert set(info) == proto_fields("GetInfoResponse")
    assert convert.output_status_fields(statuses[0])["mode"] == 3


def test_every_rpc_has_a_servicer_method_and_the_template_only_calls_real_core_methods():
    rpcs = re.findall(r"rpc (\w+)\(", PROTO)
    assert len(rpcs) == 9
    tree = ast.parse((ROOT / "device_server/rpcs.py").read_text())
    service = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RfsocAwgRpcs")
    methods = {n.name for n in service.body if isinstance(n, ast.AsyncFunctionDef)}
    assert methods == set(rpcs)
    for message in re.findall(r"returns \((\w+)\)", PROTO) + re.findall(r"rpc \w+\((?:stream )?(\w+)\)", PROTO):
        assert f"message {message} " in PROTO, f"{message} is not defined"
    # calls of the form self._device().<name>(...)
    called = {
        node.func.attr
        for node in ast.walk(service)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Call) and getattr(node.func.value.func, "attr", "") == "_device"
    }
    assert called == {"info", "status", "load_loop", "load_sequence", "load_long", "arm", "trigger",
                      "stop", "stop_all"}
    assert all(callable(getattr(DeviceCore, name)) for name in called)


def test_samples_round_trip_and_bad_payloads_are_rejected():
    data = np.array([0, 1, -1, 32767, -32768], dtype=np.int16)
    assert np.array_equal(convert.samples_from_bytes(data.astype("<i2").tobytes()), data)
    blocks = convert.blocks_from_bytes(np.arange(12, dtype="<i2").tobytes(), 4)
    assert blocks.shape == (3, 4) and blocks[2, 3] == 11
    for call in (lambda: convert.samples_from_bytes(b""), lambda: convert.samples_from_bytes(b"\x00\x01\x02"),
                 lambda: convert.blocks_from_bytes(b"\x00" * 10, 4), lambda: convert.blocks_from_bytes(b"\x00" * 8, 0),
                 lambda: convert.channel_name(0), lambda: convert.trigger_name(9), lambda: convert.end_name(9)):
        with pytest.raises(ValueError):
            call()


def test_request_values_drive_the_device_core_end_to_end():
    """What the servicer does for LoadSequence -> Arm -> Trigger, without the gRPC layer."""
    controller = Controller(seq=True)
    core = DeviceCore(controller)
    payload = np.zeros((3, 64), dtype="<i2").tobytes()
    core.load_sequence(convert.channel_name(2), convert.blocks_from_bytes(payload, 64))
    core.arm(convert.channel_name(2), trigger=convert.trigger_name(0), repeats=0 or 1, end=convert.end_name(0))
    status = core.trigger(convert.channel_name(2))
    fields = convert.output_status_fields(status)
    assert fields["channel"] == 2 and fields["state"] == convert.STATE["done"] and fields["mode"] == 2
    assert controller.rigs["dac2"].played == [0, 1, 2, 3, 4, 5]


def test_waveform_stream_requires_one_header_first():
    stream = convert.WaveformStream()
    with pytest.raises(ValueError, match="header"):
        stream.chunk(b"\x00\x00")
    with pytest.raises(ValueError, match="either"):
        stream.header(1, sample_bits=14, interpolation=2)
    with pytest.raises(ValueError, match="either"):
        stream.header(1)
    stream.header(1, sample_bits=12)
    assert (stream.channel, stream.sample_bits, stream.interpolation) == ("dac0", 12, None)
    assert stream.chunk(b"\x01\x00\x02\x00").tolist() == [1, 2]
    with pytest.raises(ValueError, match="more than one"):
        stream.header(2, interpolation=4)


def test_experiment_template_is_valid_python_and_uses_defined_messages():
    source = (ROOT / "examples/h2pcontrol/doppler_slowing.py").read_text()
    ast.parse(source)
    for name in re.findall(r"awg\.([A-Z]\w+)", source):
        assert re.search(rf"\b{name}\b", PROTO), f"{name} is not in the proto"
    for field in ("channel", "block_samples", "samples"):
        assert field in proto_fields("LoadSequenceRequest")
    assert {"channel", "trigger", "repeats", "end"} == proto_fields("ArmRequest")


def test_server_modules_are_valid_python():
    for name in ("rpcs", "service", "standalone", "stubs", "web", "__main__"):
        ast.parse((ROOT / "device_server" / f"{name}.py").read_text())
