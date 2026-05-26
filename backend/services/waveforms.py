from functools import lru_cache
from typing import Tuple

import numpy as np

from backend.signal_utils import parse_freqs_mhz, parse_ratios
from firmware import signals


@lru_cache(maxsize=32)
def generate_serrodyne_cached(
    ratios_str: str,
    freqs_str: str,
    T_total_us: float,
    amp: int,
    buf_len: int,
    dac_sr: float,
) -> Tuple[Tuple[float, ...], Tuple[float, ...], int]:
    T_s = T_total_us * 1e-6
    ratios = parse_ratios(ratios_str)
    freqs_hz = parse_freqs_mhz(freqs_str)

    x, y, N = signals.serrodyne(
        ratios,
        freqs_hz,
        T_s,
        amplitude=amp,
        sample_rate=dac_sr,
        max_points=int(buf_len),
        continuous_phase=False,
    )

    if len(y) < buf_len:
        reps = int(np.ceil(buf_len / len(y)))
        y_padded = np.tile(y, reps)[:buf_len]
        x_padded = np.tile(x, reps)[:buf_len]
    else:
        y_padded = y[:buf_len]
        x_padded = x[:buf_len]

    return tuple(x_padded.tolist()), tuple(y_padded.tolist()), N


def generate_simple_waveform(
    waveform_type: str,
    freq_hz: float,
    amp: float,
    dac_sr: float,
    buf_len: int,
    duty_cycle: float = 0.5,
) -> np.ndarray:
    match waveform_type:
        case "static":
            return np.zeros(int(buf_len), dtype=float)
        case "sine":
            return signals.sine(freq_hz, dac_sr, buf_len, amplitude=amp)
        case "cos":
            return signals.cosine(freq_hz, dac_sr, buf_len, amplitude=amp)
        case "sawtooth":
            return signals.sawtooth(freq_hz, dac_sr, buf_len, amplitude=amp)
        case "square":
            if not (0.0 <= duty_cycle <= 1.0):
                raise ValueError("duty_cycle must be between 0 and 1")
            return signals.square(freq_hz, dac_sr, buf_len, amplitude=amp, duty=duty_cycle)
        case _:
            raise ValueError(f"Unsupported waveform type: {waveform_type}")
