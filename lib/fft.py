import numpy as np


def rfft(signal, sample_rate):
    """Return one-sided FFT frequencies and complex spectrum."""
    data = np.asarray(signal)
    frequencies = np.fft.rfftfreq(data.size, d=1.0 / float(sample_rate))
    spectrum = np.fft.rfft(data)
    return frequencies, spectrum


def magnitude_db(spectrum, floor_db=-300.0):
    """Convert a spectrum to dB magnitude with a finite floor."""
    magnitude = np.abs(spectrum)
    floor = 10 ** (float(floor_db) / 20.0)
    return 20 * np.log10(np.maximum(magnitude, floor))


def dominant_frequency(signal, sample_rate):
    """Return the strongest non-DC frequency component in Hz."""
    frequencies, spectrum = rfft(signal, sample_rate)
    if frequencies.size <= 1:
        return 0.0

    index = np.argmax(np.abs(spectrum[1:])) + 1
    return float(frequencies[index])
