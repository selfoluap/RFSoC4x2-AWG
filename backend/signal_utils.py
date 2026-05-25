import numpy as np
from typing import List

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
    """Return (f_shifted_Hz, |FFT|_shifted, peak_freq_MHz, peak_mag)."""
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
