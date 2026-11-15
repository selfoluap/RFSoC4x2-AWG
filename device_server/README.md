# Device server

Makes the board a network device: a control system (h2pcontrol, or anything that speaks gRPC)
loads waveforms, arms the outputs and reads their state; the timing master's TTL starts them.

**Status: the logic underneath is tested; the gRPC layer and the h2pcontrol integration have
never been run.** gRPC was not available where this was written, and the h2pcontrol names come
from two slides of its presentation, not from its code.

## Roles

```
h2pcontrol experiment (lab PC)  --gRPC-->  device server (board)  -->  FPGA players
PulseBlaster (timing master)    --TTL--->  PMOD0 pin 0            -->  starts armed outputs
web page (any browser)          --HTTP-->  same process           -->  shows the same state
```

One process on the board owns the hardware. The gRPC service, the web backend and the status
page all use the same device core (`firmware/device.py`), so the page shows what a script did.
Notebooks on the board can still use `OverlayController` directly, but not while this server
runs: two processes cannot share the FPGA.

## Service

`h2pcontrol.rfsocawg.v1.RfsocAwgService`, defined in
`protos/h2pcontrol/rfsocawg/v1/rfsoc_awg.proto`. Every output is operated the same way:

| Call | Does |
| --- | --- |
| `GetInfo` | what the bitstream offers (sequence mode, DDR4, trigger input, ADC) and capacities |
| `GetStatus` | per output: playback mode, state, current block, warnings |
| `LoadLoop` | one waveform that repeats until stopped |
| `LoadSequence` | equal blocks, each played `repeats` times in turn (the Doppler staircase) |
| `LoadWaveform` | a long waveform into the DDR4, sent as a stream of chunks |
| `Arm` | prepare playback: trigger (immediate, software, external), repeats, end behaviour |
| `Trigger` | software trigger |
| `Stop`, `StopAll` | stop; what is loaded stays loaded |

Samples are little-endian int16. A bad request returns `INVALID_ARGUMENT`; a request the
device cannot serve in its state or with its bitstream returns `FAILED_PRECONDITION`.

ADC capture and the analog trigger are not in the service yet.

## Files

| File | Purpose | State |
| --- | --- | --- |
| `../firmware/device.py` | device core: one interface over the three players, with a lock | tested against the FPGA models |
| `convert.py` | request and response translation, no gRPC imports | tested |
| `protos/.../rfsoc_awg.proto` | service definition | checked for consistency with the two above, not compiled |
| `rpcs.py` | the RPC methods, a few lines each | not run |
| `service.py` | h2pcontrol server (template) | not run; SDK import is a guess |
| `standalone.py`, `__main__.py` | plain gRPC server, no h2pcontrol needed | not run |
| `web.py` | runs the web backend in the same process | not run |
| `../examples/h2pcontrol/doppler_slowing.py` | experiment template for the Doppler staircase | not run; h2pcontrol names are guesses |

## First run

1. On the board: `pip install grpcio grpcio-tools`, then `scripts/gen_protos.sh`.
2. Start the plain server: `python -m device_server`. It listens for gRPC on port 50051 and
   serves the web backend on 127.0.0.1:8001. Without gRPC installed it starts the web backend
   only, so the web page keeps working either way.
3. Try it with grpcui or a few lines of Python:
   ```python
   import grpc, numpy as np
   from device_server.stubs import pb, pb_grpc

   awg = pb_grpc.RfsocAwgServiceStub(grpc.insecure_channel("<board-ip>:50051"))
   print(awg.GetInfo(pb.GetInfoRequest()))
   tone = (20000 * np.sin(2 * np.pi * 0.01 * np.arange(3200))).astype("<i2")
   awg.LoadLoop(pb.LoadLoopRequest(channel=pb.CHANNEL_DAC0, samples=tone.tobytes()))
   awg.Arm(pb.ArmRequest(channel=pb.CHANNEL_DAC0, trigger=pb.TRIGGER_SOURCE_IMMEDIATE))
   print(awg.GetStatus(pb.GetStatusRequest()))
   awg.StopAll(pb.StopAllRequest())
   ```
4. The systemd unit (`deploy/systemd/rfsoc-backend.service`) now starts this combined server.

## Connecting it to h2pcontrol

1. Add `rfsoc_awg.proto` to h2pcontrol-protos so the stubs are published with the others.
2. In `service.py`, correct the SDK import (`Server`, `ServerConfig`) and compare the class
   with an existing device server, for example the PulseBlaster one: constructor arguments,
   health check, how `start()` is used.
3. Check whether `Server.start()` allows the web backend to run as a second task in the same
   event loop (`main()` in `service.py` assumes it does). If not, run the web backend as a
   gRPC client of this service.
4. In `examples/h2pcontrol/doppler_slowing.py`, correct the h2pcontrol imports and replace the
   PulseBlaster placeholder with its real calls.

## Design notes

- The core is independent of h2pcontrol, so labs on other control systems can use the plain
  server.
- Large waveforms are streamed: a 4 ms DDR4 waveform is 69 MB, above gRPC's default message
  limit of 4 MB. Sequences (at most 512 KiB) go in one message.
- Loading is per shot only if the waveform changes; otherwise load once and `Arm` each shot.
