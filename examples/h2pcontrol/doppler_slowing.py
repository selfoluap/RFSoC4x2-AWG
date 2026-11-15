"""Doppler-compensated serrodyne slowing: h2pcontrol experiment template.

TEMPLATE - NOT YET RUN. The structure follows the Picoscope example experiment (devices,
parameters, results, one `shot`). The h2pcontrol names (`Experiment`, `service_stub`, `param`,
`result`, `Results`), the stub and message import paths, and every PulseBlaster call are
placeholders taken from that one example: adjust them to the real SDK and services.

What one shot does:
  1. build the Doppler staircase for the current parameters (120 blocks by default),
  2. load it into the RFSoC and arm it on the external trigger,
  3. let the PulseBlaster run its sequence; its TTL into PMOD0 pin 0 starts the RF,
  4. read the RFSoC status back as the shot's record.

Scanning `sweep_mhz`, `repeats` or the ramp frequencies needs nothing more: they are params.
"""

import numpy as np

# --- adjust to the real packages ------------------------------------------------------------
from h2pcontrol import Experiment, Results, param, result, service_stub
from h2pcontrol.rfsocawg.v1 import rfsoc_awg_pb2 as awg
from h2pcontrol.rfsocawg.v1.rfsoc_awg_pb2_grpc import RfsocAwgServiceStub
# ---------------------------------------------------------------------------------------------

from firmware import signals            # staircase generator; pure numpy, runs on the lab PC

DAC_RATE = 9.8304e9
PATTERN_SAMPLES = 1952                  # 198.57 ns pattern, line grid 5.036 MHz


class DopplerSlowingExperiment(Experiment):
    # Devices
    rfsoc = service_stub("rfsoc-awg-service", RfsocAwgServiceStub)
    pulseblaster = service_stub("pulseblaster-service", ...)

    # Parameters
    sweep_mhz = param(600.0, min=0.0, max=1000.0, unit="MHz")        # Doppler compensation range
    sweep_ms = param(4.0, min=0.1, max=20.0, unit="ms")              # duration of the sweep
    freq_a_mhz = param(-1330.0, min=-2500.0, max=2500.0, unit="MHz")
    freq_b_mhz = param(0.0, min=-2500.0, max=2500.0, unit="MHz")
    freq_c_mhz = param(870.0, min=-2500.0, max=2500.0, unit="MHz")
    amplitude = param(30000, min=0, max=32767)

    # Results
    class Record(Results):
        n_blocks: int = result(description="Doppler steps played")
        repeats: int = result(description="pattern repeats per step")
        step_mhz: float = result(unit="MHz", description="Doppler step size")
        duration_ms: float = result(unit="ms", description="actual sweep duration")
        finished: bool = result(description="the RFSoC reported the sequence done")
        warnings: str = result(description="RFSoC warnings, empty when all is well")

    async def shot(self, ctx) -> list[Record]:
        grid_hz = DAC_RATE / PATTERN_SAMPLES
        n_blocks = max(1, int(round(self.sweep_mhz * 1e6 / grid_hz)) + 1)
        pattern_s = PATTERN_SAMPLES / DAC_RATE
        repeats = max(1, int(round(self.sweep_ms * 1e-3 / (n_blocks * pattern_s))))

        blocks, info = signals.serrodyne_staircase(
            ratios=[1, 5, 3],
            freqs_hz=[self.freq_a_mhz * 1e6, self.freq_b_mhz * 1e6, self.freq_c_mhz * 1e6],
            pattern_samples=PATTERN_SAMPLES,
            n_steps=n_blocks,
            amplitude=self.amplitude,
            sample_rate=DAC_RATE,
            predistort=None,             # the per-block predistortion function goes here
        )

        await self.rfsoc.LoadSequence(awg.LoadSequenceRequest(
            channel=awg.CHANNEL_DAC0,
            block_samples=PATTERN_SAMPLES,
            samples=np.ascontiguousarray(blocks, dtype="<i2").tobytes(),
        ))
        await self.rfsoc.Arm(awg.ArmRequest(
            channel=awg.CHANNEL_DAC0,
            trigger=awg.TRIGGER_SOURCE_EXTERNAL,
            repeats=repeats,
            end=awg.END_MODE_STOP,
        ))

        # The PulseBlaster is the timing master: one of its outputs goes to the RFSoC trigger.
        await self.pulseblaster.Run(...)                 # placeholder: start and wait for the shot

        status = (await self.rfsoc.GetStatus(awg.GetStatusRequest())).outputs[0]
        return [self.Record(
            n_blocks=n_blocks,
            repeats=repeats,
            step_mhz=info["grid_hz"] / 1e6,
            duration_ms=n_blocks * repeats * pattern_s * 1e3,
            finished=status.state == awg.OUTPUT_STATE_DONE,
            warnings="; ".join(status.warnings),
        )]
