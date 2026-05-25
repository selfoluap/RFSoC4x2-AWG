import matplotlib.pyplot as plt
import numpy as np

from .fft import magnitude_db, rfft


def plot_time(signal, sample_rate, samples=None, title=None):
    """Plot a signal in the time domain."""
    data = np.asarray(signal)
    if samples is not None:
        data = data[: int(samples)]

    t = np.arange(data.size, dtype=float) / float(sample_rate)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, data)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title(title or "Time Domain")
    ax.grid(True)
    return fig, ax


def plot_fft(signal, sample_rate, db=True, title=None):
    """Plot a one-sided FFT magnitude."""
    frequencies, spectrum = rfft(signal, sample_rate)
    y = magnitude_db(spectrum) if db else np.abs(spectrum)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(frequencies, y)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]" if db else "Magnitude")
    ax.set_title(title or "Spectrum")
    ax.grid(True)
    return fig, ax
