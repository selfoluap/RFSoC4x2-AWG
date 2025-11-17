import numpy as np
# from rfsoc4x2 import oled # type: ignore
# no oled display usable with xilinx user (needs higher privileges as it seems)
from rfsoc_mts import mtsOverlay # type: ignore

class RFSocAWG:
    def __init__(self):
        # self.oled = oled.oled_display()
        self.ol = mtsOverlay("mts.bit")

    def play_waveform(self, waveform: np.ndarray):
        self.ol.dac_player[:] = waveform.astype(np.int16)

    def capture_waveform(self, num_samples: int) -> np.ndarray:
        self.ol.trigger_capture()
        rx = self.ol.dac_capture.astype(np.float64)  # adjust scaling if needed
        return rx[:num_samples]
