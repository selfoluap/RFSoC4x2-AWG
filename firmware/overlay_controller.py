from pathlib import Path
import time

import numpy as np
from pynq import MMIO, Overlay
import xrfclk
import xrfdc


LMK_FREQ_MHZ = 245.76
LMX_FREQ_MHZ = 491.52
RF_CLOCK_SOURCE_INTERNAL = "internal"
RF_CLOCK_SOURCE_EXTERNAL = "external"
DEFAULT_BITFILE = Path(__file__).resolve().parents[1] / "overlays" / "rfsocawg.bit"
ACTIVE_DAC_TILES = 0b0101
DAC_REF_TILE = 2
MTS_TARGET_LATENCY_AUTO = -1


class DacPlayer:
    """Controller for one DAC BRAM player."""

    def __init__(self, overlay, name, bram_ip, enable_gpio, length_gpio, length_width=18):
        self.overlay = overlay
        self.name = name
        self.enable_gpio = enable_gpio
        self.length_gpio = length_gpio
        self.length_mask = (1 << length_width) - 1
        self.waveform_length = 0

        memory = overlay.mem_dict[bram_ip]
        self.bram_ip = bram_ip
        self.bram_base_addr = memory["phys_addr"]
        self.bram_bytes = memory["addr_range"]
        self.mmio = MMIO(self.bram_base_addr, self.bram_bytes)
        self.buffer = self.mmio.array[: self.mmio.length].view(np.int16)

    @property
    def capacity(self):
        """Maximum number of int16 samples that fit in the player BRAM."""
        return self.buffer.size

    def enable(self):
        self.enable_gpio.on()

    def disable(self):
        self.enable_gpio.off()

    def is_enabled(self):
        return bool(self.enable_gpio.read())

    def set_waveform_length(self, length):
        length = int(length)
        if length < 0:
            raise ValueError("waveform length must be non-negative")
        if length > self.capacity:
            raise ValueError(
                f"waveform length {length} exceeds {self.name} capacity "
                f"of {self.capacity} int16 samples"
            )
        if length > self.length_mask:
            raise ValueError(
                f"waveform length {length} exceeds 18-bit hardware limit "
                f"of {self.length_mask} samples"
            )

        self.length_gpio.write(length, self.length_mask)
        self.waveform_length = length

    def load_waveform(self, waveform):
        data = np.asarray(waveform, dtype=np.int16)
        if data.ndim != 1:
            raise ValueError("waveform must be a 1D array")
        if data.size > self.capacity:
            raise ValueError(
                f"waveform has {data.size} samples, but {self.name} capacity "
                f"is {self.capacity} int16 samples"
            )

        self.buffer[: data.size] = np.ascontiguousarray(data)
        self.set_waveform_length(data.size)

    def info(self):
        return {
            "name": self.name,
            "bram_ip": self.bram_ip,
            "bram_base_addr": self.bram_base_addr,
            "bram_bytes": self.bram_bytes,
            "bram_int16_samples": self.capacity,
            "waveform_length": self.waveform_length,
            "enabled": self.is_enabled(),
        }


class OverlayController(Overlay):
    """PYNQ overlay controller for the RFSoC4x2 AWG bitstream."""

    def __init__(self, bitfile=DEFAULT_BITFILE, download=True, **kwargs):
        xrfclk.set_ref_clks(lmk_freq=LMK_FREQ_MHZ, lmx_freq=LMX_FREQ_MHZ)
        self.set_internal_rf_clks()
        time.sleep(0.1)

        self.bitfile_path = str(Path(bitfile).expanduser())
        super().__init__(self.bitfile_path, download=download, **kwargs)
        self.xrfdc = self.usp_rf_data_converter_1       

        self.dac0 = DacPlayer(
            overlay=self,
            name="dac0",
            bram_ip="hier_dac_play/axi_bram_ctrl_0",
            enable_gpio=self.gpio_control.axi_gpio_dac.channel1[0],
            length_gpio=self.gpio_control.axi_gpio_dac.channel2,
            length_width=18,
        )
        self.dac2 = DacPlayer(
            overlay=self,
            name="dac2",
            bram_ip="hier_dac2_play/axi_bram_ctrl_0",
            enable_gpio=self.gpio_control.axi_gpio_dac.channel1[1],
            length_gpio=self.gpio_control.axi_gpio_dac2.channel1,
            length_width=18,
        )

        self.dac0.disable()
        self.dac2.disable()

    def info(self):
        return {
            "bitfile": self.bitfile_path,
            "clocks": {
                "lmk_freq_mhz": LMK_FREQ_MHZ,
                "lmx_freq_mhz": LMX_FREQ_MHZ,
                "rf_clock_source": self.rf_clock_source,
            },
            "rfdc": {
                "dac0_sampling_rate_gsps": self._rfdc_parameter("C_DAC0_Sampling_Rate"),
                "dac0_fabric_freq_mhz": self._rfdc_parameter("C_DAC0_Fabric_Freq"),
                "dac2_sampling_rate_gsps": self._rfdc_parameter("C_DAC2_Sampling_Rate"),
                "dac2_fabric_freq_mhz": self._rfdc_parameter("C_DAC2_Fabric_Freq"),
            },
            "dac0": self.dac0.info(),
            "dac2": self.dac2.info(),
        }

    def _rfdc_parameter(self, name):
        parameters = self.ip_dict["usp_rf_data_converter_1"].get("parameters", {})
        value = parameters.get(name)
        try:
            return float(value)
        except (TypeError, ValueError):
            return value

    def sync_dac_tiles(self, target_latency=MTS_TARGET_LATENCY_AUTO):
        """Configure RFDC multi-tile synchronization for the active DAC tiles."""
        self.xrfdc.mts_dac_config.Tiles = ACTIVE_DAC_TILES
        self.xrfdc.mts_dac_config.RefTile = DAC_REF_TILE
        self.xrfdc.mts_dac_config.SysRef_Enable = 1
        self.xrfdc.mts_dac_config.Target_Latency = target_latency
        return self.xrfdc.mts_dac()

    def dac_mts_info(self):
        """Return the current DAC MTS configuration fields."""
        config = self.xrfdc.mts_dac_config
        return {
            "tiles": config.Tiles,
            "ref_tile": config.RefTile,
            "sysref_enable": config.SysRef_Enable,
            "target_latency": config.Target_Latency,
        }

    def set_external_rf_clks(self):
        """Use the 10 MHz reference connected to CLK_IN for the RF clocks."""
        for lmk in xrfclk.lmk_devices:
            with open(lmk["spi_device"], "rb+", buffering=0) as f:
                data = b"\x01\x47\x0A"
                f.write(data)

        self.rf_clock_source = RF_CLOCK_SOURCE_EXTERNAL

    def set_internal_rf_clks(self):
        """Use the on-board oscillator as the RF clock reference."""
        for lmk in xrfclk.lmk_devices:
            with open(lmk["spi_device"], "rb+", buffering=0) as f:
                data = b"\x01\x47\x1A"
                f.write(data)

        self.rf_clock_source = RF_CLOCK_SOURCE_INTERNAL
