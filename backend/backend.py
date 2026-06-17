"""FastAPI backend for RFSoC AWG control."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, List, Literal

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.signal_utils import calculate_fft
from backend.services.hardware import DAC_AMP, DAC_CHANNELS, HardwareService, hardware_service
from backend.services.waveforms import generate_serrodyne_cached, generate_simple_waveform


logger = logging.getLogger(__name__)


class SerrodyneParams(BaseModel):
    channels: List[Literal["dac0", "dac2"]] = Field(default_factory=lambda: ["dac0"])
    ratios_str: str = "1:5:3"
    freqs_str: str = "-1330, 0, 840"
    T_total_us: float = 1.0
    amp: int = 16383


class SimpleWaveformParams(BaseModel):
    channels: List[Literal["dac0", "dac2"]] = Field(default_factory=lambda: ["dac0"])
    waveform_type: str
    freq_mhz: float = 250.0
    amp: int = 16383
    duty_cycle: float = Field(default=0.5, ge=0.0, le=1.0)


class FFTResponse(BaseModel):
    frequencies: List[float]
    magnitudes: List[float]


class StatusResponse(BaseModel):
    hardware_initialized: bool
    buf_len: int
    dac_sr: float
    dacs: dict[str, dict[str, Any]]


class DacControlResponse(BaseModel):
    success: bool
    channel: str
    enabled: bool
    dacs: dict[str, dict[str, Any]]


class WaveformLoadResponse(BaseModel):
    success: bool
    message: str
    channels: List[str]
    num_samples: int
    dacs: dict[str, dict[str, Any]]


class ConstantsResponse(BaseModel):
    DAC_SR: float
    DAC_AMP: int
    BUF_LEN: int
    overlay_info: dict[str, Any]


def get_hardware_service() -> HardwareService:
    return hardware_service


def normalize_channels(channels: List[str]) -> List[str]:
    unique_channels = list(dict.fromkeys(channels))
    if not unique_channels:
        raise ValueError("Select at least one DAC channel")
    unsupported = [channel for channel in unique_channels if channel not in DAC_CHANNELS]
    if unsupported:
        raise ValueError(f"Unsupported DAC channel: {', '.join(unsupported)}")
    return unique_channels


def compute_fft_for_waveform(signal: np.ndarray, dac_sr: float) -> dict[str, List[float]]:
    signal_span = float(np.max(signal) - np.min(signal))
    if signal_span == 0.0:
        raise ValueError("Cannot compute FFT for constant signal (zero dynamic range)")

    phase_scale = 2 * np.pi / signal_span
    phase = phase_scale * signal
    modulated_field = np.exp(1j * phase)
    frequencies_hz, magnitudes, _, _ = calculate_fft(modulated_field, dac_sr)
    return {
        "frequencies": (frequencies_hz / 1.0e6).tolist(),
        "magnitudes": magnitudes.tolist(),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    hardware_service.initialize()
    print("Hardware initialized")
    yield


app = FastAPI(
    title="RFSoC AWG Backend",
    description="REST API for RFSoC Arbitrary Waveform Generator control",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/status", response_model=StatusResponse)
def get_status(hardware: HardwareService = Depends(get_hardware_service)):
    hardware.initialize()
    return StatusResponse(
        hardware_initialized=hardware.initialized,
        buf_len=hardware.buf_len,
        dac_sr=hardware.dac_sr,
        dacs=hardware.dacs_status(),
    )


@app.post("/waveform/serrodyne/load", response_model=WaveformLoadResponse)
def load_serrodyne_waveform(
    params: SerrodyneParams,
    hardware: HardwareService = Depends(get_hardware_service),
):
    try:
        channels = normalize_channels(params.channels)
        _, y_tuple, n_samples = generate_serrodyne_cached(
            params.ratios_str,
            params.freqs_str,
            params.T_total_us,
            params.amp,
            hardware.buf_len,
            hardware.dac_sr,
        )
        signal = np.array(y_tuple)
        for channel in channels:
            hardware.load_dac(signal, channel)
            hardware.set_dac_enabled(channel, False)
        return WaveformLoadResponse(
            success=True,
            message=f"Serrodyne waveform loaded with {n_samples} base samples; output disabled",
            channels=channels,
            num_samples=len(signal),
            dacs=hardware.dacs_status(),
        )
    except ValueError as e:
        logger.warning("Validation error generating serrodyne waveform: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error generating serrodyne waveform")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/waveform/simple/load", response_model=WaveformLoadResponse)
def load_simple_waveform(
    params: SimpleWaveformParams,
    hardware: HardwareService = Depends(get_hardware_service),
):
    try:
        channels = normalize_channels(params.channels)
        signal = generate_simple_waveform(
            params.waveform_type,
            params.freq_mhz * 1e6,
            params.amp,
            hardware.dac_sr,
            hardware.buf_len,
            params.duty_cycle,
        )
        for channel in channels:
            hardware.load_dac(signal, channel)
            hardware.set_dac_enabled(channel, False)
        return WaveformLoadResponse(
            success=True,
            message=f"{params.waveform_type} waveform loaded at {params.freq_mhz} MHz; output disabled",
            channels=channels,
            num_samples=len(signal),
            dacs=hardware.dacs_status(),
        )
    except ValueError as e:
        logger.warning("Validation error generating simple waveform: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error generating simple waveform")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dac/all/disable", response_model=dict[str, Any])
def disable_all_dacs(hardware: HardwareService = Depends(get_hardware_service)):
    for channel in DAC_CHANNELS:
        hardware.set_dac_enabled(channel, False)
    return {
        "success": True,
        "dacs": hardware.dacs_status(),
    }


@app.post("/dac/{channel}/enable", response_model=DacControlResponse)
def enable_dac(channel: Literal["dac0", "dac2"], hardware: HardwareService = Depends(get_hardware_service)):
    hardware.set_dac_enabled(channel, True)
    return DacControlResponse(
        success=True,
        channel=channel,
        enabled=True,
        dacs=hardware.dacs_status(),
    )


@app.post("/dac/{channel}/disable", response_model=DacControlResponse)
def disable_dac(channel: Literal["dac0", "dac2"], hardware: HardwareService = Depends(get_hardware_service)):
    hardware.set_dac_enabled(channel, False)
    return DacControlResponse(
        success=True,
        channel=channel,
        enabled=False,
        dacs=hardware.dacs_status(),
    )


@app.get("/waveform/fft", response_model=FFTResponse)
def get_waveform_fft(hardware: HardwareService = Depends(get_hardware_service)):
    if hardware.last_signal is None:
        raise HTTPException(status_code=400, detail="No signal available")
    try:
        result = compute_fft_for_waveform(hardware.last_signal, hardware.dac_sr)
        return FFTResponse(**result)
    except ValueError as e:
        logger.warning("Invalid waveform data for FFT: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during waveform FFT")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/constants", response_model=ConstantsResponse)
def get_constants(hardware: HardwareService = Depends(get_hardware_service)):
    overlay_info = hardware.info()
    return {
        "DAC_SR": hardware.dac_sr,
        "DAC_AMP": DAC_AMP,
        "BUF_LEN": hardware.buf_len,
        "overlay_info": overlay_info,
    }


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("RFSOC_BACKEND_HOST", "0.0.0.0")
    port = int(os.environ.get("RFSOC_BACKEND_PORT", "8001"))
    uvicorn.run(app, host=host, port=port)
