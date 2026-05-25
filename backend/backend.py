"""FastAPI backend for RFSoC AWG control."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, List, Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.signal_utils import calculate_fft
from backend.services.hardware import DAC_AMP, HardwareService, hardware_service
from backend.services.waveforms import generate_serrodyne_cached, generate_simple_waveform


logger = logging.getLogger(__name__)


class SerrodyneParams(BaseModel):
    ratios_str: str = "1:5:3"
    freqs_str: str = "-1330, 0, 840"
    T_total_us: float = 1.0
    amp: int = 16383


class SimpleWaveformParams(BaseModel):
    waveform_type: str
    freq_mhz: float = 250.0
    amp: int = 16383
    duty_cycle: float = Field(default=0.5, ge=0.0, le=1.0)


class WaveformResponse(BaseModel):
    success: bool
    message: str
    signal: Optional[List[float]] = None
    x_axis: Optional[List[float]] = None
    num_samples: Optional[int] = None


class FFTResponse(BaseModel):
    frequencies: List[float]
    magnitudes: List[float]


class StatusResponse(BaseModel):
    hardware_initialized: bool
    buf_len: int
    dac_sr: float


class ConstantsResponse(BaseModel):
    DAC_SR: float
    DAC_AMP: int
    BUF_LEN: int
    overlay_info: dict[str, Any]


def get_hardware_service() -> HardwareService:
    return hardware_service


def build_waveform_response(
    message: str,
    hardware: HardwareService,
    signal: np.ndarray,
    preview_samples: int = 4000,
) -> WaveformResponse:
    preview_count = max(1, min(preview_samples, len(signal), len(hardware.x_axis)))
    return WaveformResponse(
        success=True,
        message=message,
        signal=signal[:preview_count].tolist(),
        x_axis=hardware.x_axis[:preview_count].tolist(),
        num_samples=len(signal),
    )


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
    )


@app.post("/waveform/serrodyne", response_model=WaveformResponse)
def generate_and_output_serrodyne(
    params: SerrodyneParams,
    hardware: HardwareService = Depends(get_hardware_service),
):
    try:
        _, y_tuple, n_samples = generate_serrodyne_cached(
            params.ratios_str,
            params.freqs_str,
            params.T_total_us,
            params.amp,
            hardware.buf_len,
            hardware.dac_sr,
        )
        signal = np.array(y_tuple)
        hardware.write_dac(signal)
        return build_waveform_response(
            message=f"Serrodyne waveform generated with {n_samples} base samples",
            hardware=hardware,
            signal=signal,
            preview_samples=max(4000, n_samples),
        )
    except ValueError as e:
        logger.warning("Validation error generating serrodyne waveform: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error generating serrodyne waveform")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/waveform/simple", response_model=WaveformResponse)
def generate_and_output_simple(
    params: SimpleWaveformParams,
    hardware: HardwareService = Depends(get_hardware_service),
):
    try:
        signal = generate_simple_waveform(
            params.waveform_type,
            params.freq_mhz * 1e6,
            params.amp,
            hardware.dac_sr,
            hardware.buf_len,
            params.duty_cycle,
        )
        hardware.write_dac(signal)
        return build_waveform_response(
            message=f"{params.waveform_type} waveform generated at {params.freq_mhz} MHz",
            hardware=hardware,
            signal=signal,
        )
    except ValueError as e:
        logger.warning("Validation error generating simple waveform: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error generating simple waveform")
        raise HTTPException(status_code=500, detail=str(e))


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


hardware = hardware_service


if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv()
    host = os.environ.get("RFSOC_BACKEND_HOST", "0.0.0.0")
    port = int(os.environ.get("RFSOC_BACKEND_PORT", "8001"))
    uvicorn.run(app, host=host, port=port)
