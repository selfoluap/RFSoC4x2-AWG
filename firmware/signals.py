import numpy as np
from scipy import signal as scipy_signal


def time_axis(num_samples, sample_rate):
    """Return a time axis in seconds."""
    return np.arange(int(num_samples), dtype=float) / float(sample_rate)


def sine(freq_hz, sample_rate, num_samples, amplitude=1.0, phase=0.0):
    """Generate a sine wave."""
    t = time_axis(num_samples, sample_rate)
    return float(amplitude) * np.sin(2 * np.pi * float(freq_hz) * t + float(phase))


def cosine(freq_hz, sample_rate, num_samples, amplitude=1.0, phase=0.0):
    """Generate a cosine wave."""
    t = time_axis(num_samples, sample_rate)
    return float(amplitude) * np.cos(2 * np.pi * float(freq_hz) * t + float(phase))


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


def serrodyne(
    ratios,
    freqs_hz,
    total_seconds,
    *,
    amplitude,
    sample_rate,
    max_points=None,
    width=1.0,
    continuous_phase=False,
):
    """Generate a piecewise serrodyne sawtooth waveform."""
    if len(ratios) != len(freqs_hz):
        raise ValueError("ratios and frequencies must have same length")
    if total_seconds <= 0:
        raise ValueError("total_seconds must be > 0")
    if not (0.0 <= width <= 1.0):
        raise ValueError("width must be in [0, 1]")

    sample_rate = float(sample_rate)
    n_samples = max(1, int(round(float(total_seconds) * sample_rate)))
    if max_points is not None:
        n_samples = min(n_samples, int(max_points))
    dt = 1.0 / sample_rate

    ratio_sum = sum(ratios)
    segment_lengths = [int(round(n_samples * (ratio / ratio_sum))) for ratio in ratios]
    delta = n_samples - sum(segment_lengths)
    index = 0
    while delta != 0 and index < len(segment_lengths) * 4:
        segment_index = index % len(segment_lengths)
        candidate = segment_lengths[segment_index] + (1 if delta > 0 else -1)
        if candidate >= 0:
            segment_lengths[segment_index] = candidate
            delta = n_samples - sum(segment_lengths)
        index += 1

    x = time_axis(n_samples, sample_rate)
    y = np.zeros(n_samples, dtype=float)
    start = 0
    phase_offset = 0.0
    two_pi = 2 * np.pi

    for length, frequency in zip(segment_lengths, freqs_hz):
        end = start + length
        if length > 0:
            t = time_axis(length, sample_rate)
            if frequency == 0:
                segment = np.zeros(length)
                if continuous_phase:
                    phase_offset = phase_offset % two_pi
            else:
                phase = (two_pi * float(frequency) * t) + (phase_offset if continuous_phase else 0.0)
                segment = float(amplitude) * scipy_signal.sawtooth(phase, width=float(width))
                if continuous_phase:
                    phase_offset = (two_pi * float(frequency) * (length * dt) + phase_offset) % two_pi
            y[start:end] = segment
        start = end

    return x, y, n_samples
