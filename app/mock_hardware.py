"""Mock hardware classes for offline development without RFSoC hardware."""

import numpy as np


class MockOLED:
    """Mock OLED display for offline development."""
    def __init__(self):
        pass
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class MockOverlay:
    """Mock overlay for offline development."""
    def __init__(self, bitfile=None):
        self._buf_len = 65536
        self._dac_player = np.zeros(self._buf_len, dtype=np.int16)
        self._adc_capture = np.zeros(self._buf_len, dtype=np.int16)
    
    @property
    def dac_player(self):
        return self._dac_player
    
    @property
    def adc_capture_chA(self):
        return self._adc_capture
    
    def init_tile_sync(self):
        pass
    
    def verify_clock_tree(self):
        pass
    
    def sync_tiles(self):
        pass
    
    def internal_capture(self, buffer):
        """Simulate captured signal: return the DAC signal with some noise."""
        noise = np.random.normal(0, 100, len(self._dac_player)).astype(np.int16)
        buffer[0][:] = self._dac_player + noise
