"""Translation between the gRPC messages and the device core.

Kept free of gRPC imports so it is unit tested off the board. The servicer passes request
fields in and gets plain dicts back, which it feeds to the generated message classes. Enum
numbers here must match rfsoc_awg.proto; tests/test_device_server.py checks that they do.
"""

from __future__ import annotations

import numpy as np

CHANNEL = {1: "dac0", 2: "dac2"}
MODE = {"none": 0, "loop": 1, "sequence": 2, "long": 3}
STATE = {"idle": 1, "armed": 2, "running": 3, "holding": 4, "done": 5}
TRIGGER = {0: "software", 1: "immediate", 2: "software", 3: "external"}
END = {0: "stop", 1: "stop", 2: "hold", 3: "loop"}

CHANNEL_NUMBER = {name: number for number, name in CHANNEL.items()}


def channel_name(number: int) -> str:
    if number not in CHANNEL:
        raise ValueError("channel must be CHANNEL_DAC0 or CHANNEL_DAC2")
    return CHANNEL[number]


def trigger_name(number: int) -> str:
    if number not in TRIGGER:
        raise ValueError(f"unknown trigger source {number}")
    return TRIGGER[number]


def end_name(number: int) -> str:
    if number not in END:
        raise ValueError(f"unknown end mode {number}")
    return END[number]


def samples_from_bytes(data: bytes) -> np.ndarray:
    """Little-endian int16 samples from a bytes field."""
    if len(data) == 0:
        raise ValueError("no samples")
    if len(data) % 2:
        raise ValueError("samples must be a whole number of int16 values (even byte count)")
    return np.frombuffer(data, dtype="<i2").astype(np.int16, copy=False)


def blocks_from_bytes(data: bytes, block_samples: int) -> np.ndarray:
    samples = samples_from_bytes(data)
    if block_samples <= 0 or samples.size % block_samples:
        raise ValueError(
            f"{samples.size} samples are not a whole number of blocks of {block_samples} samples"
        )
    return samples.reshape(-1, block_samples)


def output_status_fields(status: dict) -> dict:
    """Keyword arguments for the OutputStatus message from a device-core status dict."""
    fields = {
        "channel": CHANNEL_NUMBER[status["channel"]],
        "mode": MODE[status["mode"]],
        "state": STATE[status["state"]],
        "block": int(status["block"]),
        "loaded_samples": int(status["loaded_samples"]),
        "warnings": list(status["warnings"]),
    }
    if status.get("buffer_low_water") is not None:
        fields["buffer_low_water"] = int(status["buffer_low_water"])
    return fields


def channel_info_fields(entry: dict) -> dict:
    return {
        "channel": CHANNEL_NUMBER[entry["channel"]],
        "loop_samples": int(entry.get("loop_samples", 0)),
        "sequence_samples": int(entry.get("sequence_samples", 0)),
        "long_bytes": int(entry.get("long_bytes", 0)),
    }


class WaveformStream:
    """Collects a LoadWaveform client stream: one header, then sample chunks.

    ``header(...)`` must be called once before the first ``chunk(...)``, which turns one
    message's bytes into int16 samples for ``DeviceCore.load_long``. A chunk with an odd byte
    count is rejected, so every message carries whole samples.
    """

    def __init__(self):
        self.channel = None
        self.sample_bits = None
        self.interpolation = None

    def header(self, channel: int, sample_bits: int = 0, interpolation: int = 0) -> None:
        if self.channel is not None:
            raise ValueError("LoadWaveform: more than one header in the stream")
        if bool(sample_bits) == bool(interpolation):
            raise ValueError("LoadWaveform header: set either sample_bits or interpolation")
        self.channel = channel_name(channel)
        self.sample_bits = sample_bits or None
        self.interpolation = interpolation or None

    def chunk(self, data: bytes) -> np.ndarray:
        if self.channel is None:
            raise ValueError("LoadWaveform: the first message must be the header")
        return samples_from_bytes(data)
