from typing import List, Tuple, Dict, Optional
from scipy.signal import sawtooth
import numpy as np
import plotly.graph_objects as go # type: ignore

def plot_time_series_interactive(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "",
    x_label: str = "Time (s)",
    y_label: str = "Amplitude",
    height: int = 450,
    name: str = "signal",
):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode="x unified",
        height=height,
    )
    return fig


def plot_fft(
    f_hz: np.ndarray,
    magnitude: np.ndarray,
    *,
    title: str = "FFT",
    y_label: str = "Magnitude",
    name: str = "|FFT|",
    height: int = 450,
):
    x_vals = f_hz / 1.0e6  # default to MHz
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=magnitude, mode="lines", name=name))
    fig.update_layout(
        title=title,
        xaxis_title=f"Frequency",
        yaxis_title=y_label,
        height=height,
    )
    return fig


#function to create signal
def make_piecewise_ramp(
    ratios: List[int],
    freqs_hz: List[float],
    T_total_s: float,
    amp: float,
    sr_hz: float,
    max_points: int | None = None,
    *,
    width: float = 1.0,                 # 1.0 rising, 0.0 falling
    continuous_phase: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Dict]:
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
    bounds = []

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
        bounds.append((start, end))
        start = end
    return x, y, N


def parse_ratios(ratio_text: str) -> List[int]:
    if ":" in ratio_text:
        parts = ratio_text.split(":")
    else:
        parts = ratio_text.split(",")
    ratios = [int(p.strip()) for p in parts if p.strip()]
    if not ratios or any(r <= 0 for r in ratios):
        raise ValueError("Ratios must be positive integers, e.g., 1:5:3")
    return ratios


def parse_freqs_mhz(freq_text: str) -> List[float]:
    parts = [p.strip() for p in freq_text.replace("MHz", "").split(",")]
    freqs = []
    for p in parts:
        if p == "":
            continue
        freqs.append(float(p) * 1e6)
    if not freqs:
        raise ValueError("Provide frequencies, e.g., 1330, 0, 840 (MHz)")
    return freqs


def calculate_fft(signal: np.ndarray, sr_hz: float):
    """Return (f_shifted, |FFT|_shifted, peak_freq_MHz, peak_mag)."""
    y = np.asarray(signal)
    if y.size == 0:
        return np.array([]), np.array([]), 0.0, 0.0
    N = len(y)
    T = 1.0 / sr_hz
    Y = np.fft.fft(y)
    f = np.fft.fftfreq(N, T)
    mag = np.abs(Y)
    fsh = np.fft.fftshift(f)
    magsh = np.fft.fftshift(mag)
    idx = int(np.argmax(magsh)) if magsh.size else 0
    peak_f_mhz = (fsh[idx] / 1e6) if fsh.size else 0.0
    peak_mag = float(magsh[idx]) if magsh.size else 0.0
    return fsh, magsh, peak_f_mhz, peak_mag
