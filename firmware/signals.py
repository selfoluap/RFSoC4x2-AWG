import numpy as np
from scipy import signal as scipy_signal


def _to_int16(waveform):
    limits = np.iinfo(np.int16)
    return np.round(np.clip(waveform, limits.min, limits.max)).astype(np.int16)


def time_axis(num_samples, sample_rate):
    """Return a time axis in seconds."""
    return np.arange(int(num_samples), dtype=float) / float(sample_rate)


def sine(freq_hz, sample_rate, num_samples, amplitude=1.0, phase=0.0):
    """Generate an int16 sine wave with amplitude in DAC-code units."""
    t = time_axis(num_samples, sample_rate)
    return _to_int16(float(amplitude) * np.sin(2 * np.pi * float(freq_hz) * t + float(phase)))


def cosine(freq_hz, sample_rate, num_samples, amplitude=1.0, phase=0.0):
    """Generate an int16 cosine wave with amplitude in DAC-code units."""
    t = time_axis(num_samples, sample_rate)
    return _to_int16(float(amplitude) * np.cos(2 * np.pi * float(freq_hz) * t + float(phase)))


def sawtooth(freq_hz, sample_rate, num_samples, amplitude=1.0, width=1.0):
    """Generate an int16 sawtooth wave with amplitude in DAC-code units."""
    t = time_axis(num_samples, sample_rate)
    waveform = float(amplitude) * scipy_signal.sawtooth(
        2 * np.pi * float(freq_hz) * t,
        width=float(width),
    )
    return _to_int16(waveform)


def square(freq_hz, sample_rate, num_samples, amplitude=1.0, duty=0.5):
    """Generate an int16 square wave with amplitude in DAC-code units."""
    t = time_axis(num_samples, sample_rate)
    waveform = float(amplitude) * scipy_signal.square(
        2 * np.pi * float(freq_hz) * t,
        duty=float(duty),
    )
    return _to_int16(waveform)


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
    """Return (time_seconds, int16_waveform, sample_count) for serrodyne.

    Amplitude is expressed in DAC-code units.
    """
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

    return x, _to_int16(y), n_samples
