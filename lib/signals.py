import numpy as np
from scipy import signal as scipy_signal


INT16_MAX = np.iinfo(np.int16).max


def time_axis(num_samples, sample_rate):
    """Return a time axis in seconds."""
    return np.arange(int(num_samples), dtype=float) / float(sample_rate)


def sine(freq_hz, sample_rate, num_samples, amplitude=1.0, phase=0.0):
    """Generate a sine wave."""
    t = time_axis(num_samples, sample_rate)
    return float(amplitude) * np.sin(2 * np.pi * float(freq_hz) * t + float(phase))


def sawtooth(freq_hz, sample_rate, num_samples, amplitude=1.0, width=1.0):
    """Generate a sawtooth wave using scipy.signal.sawtooth."""
    t = time_axis(num_samples, sample_rate)
    return float(amplitude) * scipy_signal.sawtooth(
        2 * np.pi * float(freq_hz) * t,
        width=float(width),
    )


def square(freq_hz, sample_rate, num_samples, amplitude=1.0, duty=0.5):
    """Generate a square wave using scipy.signal.square."""
    t = time_axis(num_samples, sample_rate)
    return float(amplitude) * scipy_signal.square(
        2 * np.pi * float(freq_hz) * t,
        duty=float(duty),
    )


def chirp(start_hz, stop_hz, sample_rate, num_samples, amplitude=1.0, method="linear"):
    """Generate a swept-frequency chirp."""
    t = time_axis(num_samples, sample_rate)
    duration = len(t) / float(sample_rate)
    return float(amplitude) * scipy_signal.chirp(
        t,
        f0=float(start_hz),
        f1=float(stop_hz),
        t1=duration,
        method=method,
    )


def normalize_to_int16(signal, peak=INT16_MAX):
    """Scale a real-valued signal to signed int16 samples."""
    data = np.asarray(signal, dtype=float)
    max_abs = np.max(np.abs(data)) if data.size else 0.0
    if max_abs == 0.0:
        return np.zeros(data.shape, dtype=np.int16)

    peak = min(abs(float(peak)), INT16_MAX)
    scaled = data / max_abs * peak
    return np.round(np.clip(scaled, -INT16_MAX, INT16_MAX)).astype(np.int16)
