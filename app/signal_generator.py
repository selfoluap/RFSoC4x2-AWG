from typing import List, Tuple
import numpy as np
from scipy.signal import sawtooth

def calculate_serrodyne(
    ratios: List[int],
    freqs_hz: List[float],
    T_total_s: float,
    *,
    amp: float,
    sr_hz: float,
    max_points: int | None = None,
    width: float = 1.0,
    continuous_phase: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if len(ratios) != len(freqs_hz):
        raise ValueError("ratios and frequencies must have same length")
    if T_total_s <= 0:
        raise ValueError("T_total must be > 0")
    if not (0.0 <= width <= 1.0):
        raise ValueError("width must be in [0, 1]")

    N = max(1, int(round(T_total_s * sr_hz)))
    if max_points is not None:
        N = min(N, int(max_points))
    dt = 1.0 / sr_hz

    rsum = sum(ratios)
    seg_lengths = [int(round(N * (r / rsum))) for r in ratios]
    delta = N - sum(seg_lengths)
    i = 0
    while delta != 0 and i < len(seg_lengths) * 4:
        j = i % len(seg_lengths)
        cand = seg_lengths[j] + (1 if delta > 0 else -1)
        if cand >= 0:
            seg_lengths[j] = cand
            delta = N - sum(seg_lengths)
        i += 1

    x = np.arange(N) * dt
    y = np.zeros(N, dtype=float)
    start = 0
    phi0 = 0.0
    w2pi = 2 * np.pi

    for L, f in zip(seg_lengths, freqs_hz):
        end = start + L
        if L > 0:
            t = np.arange(L) * dt
            if f == 0:
                seg = np.zeros(L)
                if continuous_phase:
                    phi0 = phi0 % (2 * np.pi)
            else:
                phase = (w2pi * f * t) + (phi0 if continuous_phase else 0.0)
                seg = amp * sawtooth(phase, width=width)
                if continuous_phase:
                    phi0 = (w2pi * f * (L * dt) + phi0) % (2 * np.pi)
            y[start:end] = seg
        start = end
    return x, y, N


def sine(freq_hz: float, amp: float, x_axis: np.ndarray) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq_hz * x_axis)


def saw(freq_hz: float, amp: float, x_axis: np.ndarray) -> np.ndarray:
    return amp * sawtooth(2 * np.pi * freq_hz * x_axis)


def square(freq_hz: float, amp: float, x_axis: np.ndarray) -> np.ndarray:
    return amp * np.sign(np.sin(2 * np.pi * freq_hz * x_axis))
