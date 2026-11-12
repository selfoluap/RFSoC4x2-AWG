# ADC capture, spectrum, live view and analog trigger

Records one of the four ADC inputs into on-chip memory on a trigger, turns a capture into a
spectrum in Python, shows both live in the web app, and can trigger on the ADC signal itself.

**Status: logic verified on a register-transfer model, not yet simulated in an HDL simulator,
not yet built in Vivado, not yet run on a board.**

## What you get

| Property | Value |
| --- | --- |
| Inputs | ADC A to D, one recorded at a time (selected per capture) |
| Sample rate | 4915.2 MS/s, 14-bit: exactly half the DAC rate, same reference |
| Frequency range | DC to 2457.6 MHz; higher frequencies fold back (a tone at f shows at 4915.2 MHz - f) |
| Record length | up to 262,144 samples = 53.3 us (16 UltraRAM) |
| Trigger | software, PMOD0 pin 0, the start of DAC0 / DAC2 sequence playback, or the analog trigger |
| Trigger delay | 0 to about 14 s in steps of 3.26 ns, to place the window anywhere in a long sequence |
| Spectrum | computed in Python from a capture: Hann window, averaged, in dBFS; 75 kHz bins at 65,536 points |

The ADC inputs of the RFSoC 4x2 are coupled through on-board baluns, so they show RF, not DC or
slow levels. Check the low-frequency cutoff in the board manual.

With a cable from a DAC output to an ADC input this gives a self-test of the players without a
scope: trigger the capture on playback start, set the delay, and look at any 53 us window of
the output up to 2.46 GHz.

## Use

```python
from firmware import OverlayController
from firmware.adc_capture import spectrum

ol = OverlayController("rfsocawg_adc.bit")
adc = ol.adc                                  # None when the bitstream has no ADC path

samples = adc.capture("A")                    # 262,144 samples now, as int16
freqs, level_dbfs = spectrum(samples)         # averaged spectrum, 0 dBFS = full-scale sine

# 53 us window, 2 ms into a Doppler staircase on DAC0, started by the same trigger
adc.arm("A", trigger="dac0", delay_s=2e-3)
ol.seq["dac0"].arm(repeats=168, external_trigger=True)
window = adc.read(timeout=10)
```

`spectrum(samples, nfft=4096)` averages more segments for a smoother trace with coarser bins.

## Analog trigger

A detector on one ADC input that raises a trigger when a signal appears, as an alternative to
the digital PMOD pin.

| Type | Fires when | Use |
| --- | --- | --- |
| `envelope` | any \|sample\| reaches the threshold | an RF burst of any carrier frequency |
| `rising` | a sample rises to the threshold (signed) | a positive edge or pulse |
| `falling` | a sample falls to the threshold (signed) | a negative edge or pulse |

| Property | Value |
| --- | --- |
| Time resolution | one 16-sample word, 3.26 ns |
| Detection delay | fixed: 3 clock cycles (9.8 ns) inside the FPGA, plus the ADC's own delay (to be measured) |
| Hysteresis | separate "off" threshold; default half the threshold in envelope mode |
| Release time | 0 to 213 us; bridges the zero crossings of a slow carrier (default 26 ns) |
| Readings | active, event count, and the input's peak level for choosing a threshold |

What it triggers:

- **Capture:** `ol.adc.arm("A", trigger="analog")`.
- **DAC players:** the players' trigger input becomes "PMOD pin OR analog trigger", so a player
  armed with `external_trigger=True` starts on either. The analog trigger is silent unless it
  is enabled.
- **Live view:** "On signal" in the web panel starts every record at the trigger, so the time
  trace stands still (to within one word, 3.26 ns); without a signal it falls back to
  untriggered after 50 ms, like an oscilloscope's auto mode.

```python
trig = ol.analog_trigger                      # None when the bitstream has no ADC path
print(trig.status()["peak"])                  # with the signal applied: pick about half of this
trig.configure("B", threshold=4000)           # RF burst on ADC B
ol.seq["dac0"].arm(repeats=168, external_trigger=True)   # the burst on ADC B starts playback
```

The ADC inputs are AC-coupled, so a slow level change (a photodiode's DC level) does not reach
the detector; an RF burst or a fast edge does.

State: 8 logic tests on the detector model, 5 driver tests running the detector and capture
models together, 2 live-view tests. The Verilog (`src/rtl/AnalogTrigger.v`, testbench
`../hdl/tb_AnalogTrigger.v`) is not compiled or simulated; expected last line of the testbench:
`PASS: analog trigger tests, no errors`. Compile it together with the capture block:
`iverilog -g2012 -o tb_adc hdl/tb_ADCCapture.v fpga/src/rtl/ADCCapture.v fpga/src/rtl/AnalogTrigger.v`.

## Live view in the web app

The web app has a "Live view" panel: a spectrum and a time trace of one ADC input that update
continuously, with no manual triggering.

| Control | Options |
| --- | --- |
| Input | ADC A to D |
| FFT size | 1,024 to 65,536 points (4.8 MHz down to 75 kHz per bin) |
| Averaging | off, or a running average over 2 to 64 frames |
| Max hold | keeps the highest level seen per frequency; "Clear" restarts it |
| Frames per second | 5 to 30 requested; the readout shows the rate achieved |

Readouts: peak frequency and level, total level with a clipping warning, resolution, and the
achieved frame rate.

How it works (stage 1, software only, no FPGA change):

- The browser opens a WebSocket to `/api/adc/live`. For each frame the backend re-arms the
  capture, triggers it itself, reads a short record, computes the spectrum and sends about
  10 kB of JSON. The browser draws it on a canvas.
- The board only captures while a browser has the panel open and running.
- The live view and other captures share the one capture memory through a lock in the backend.
- Each frame is a snapshot of at most 53 us, so the ADC is watched for well under 1 % of the
  time: good for continuous or repeating signals, rare events are usually missed. Averaging and
  max hold in the FPGA (stage 2) would raise that.
- In "Free run" the time trace does not stand still. "On signal" uses the analog trigger
  for that and draws the threshold on the trace.

Deploying it: rebuild the frontend (`npm run build`), re-run `scripts/deploy_frontend.sh`, and
re-run `scripts/install_frontend_nginx.sh` once, because the nginx site now forwards WebSocket
upgrades. Restart the backend service.

State: the frame builder and the connection loop pass 12 tests off the board, and the plotting
code was rendered in a headless browser with a frame from the backend. **Not yet run:** the
FastAPI endpoint itself, the React panel in a real build, and the whole chain on the board. The
frame rate on the board's ARM cores is not measured; computing a frame took 0.5 to 5 ms on a
desktop CPU, depending on FFT size.

Files: `backend/services/live.py`, `/adc/info` and `/adc/live` in `backend/backend.py`,
`frontend/src/components/LiveAdcPanel.tsx`, `frontend/src/live/`, `deploy/nginx/rfsoc-awg.conf`.

## How it works

```
RFDC ADC stream (8 samples, 614.4 MHz) -> 16 samples per word -> 307.2 MHz --+
                                                    (one chain per ADC)      +-> input select
                                                                             -> ADCCapture -> memory -> PS
```

After the trigger, `ADCCapture` skips `delay` words and writes the next `nrows` words to the
memory. The processor reads them through the memory's other port. The ADC streams are never
stalled, and gaps in the stream's valid signal lose no samples.

## Files

| File | Purpose |
| --- | --- |
| `src/rtl/ADCCapture.v` | input select, trigger, delay, recorder |
| `src/bd/add_adc_capture.tcl` | enables the ADCs, adds stream chains, memory and GPIOs |
| `src/constraints/adc_capture.xdc` | CDC constraints |
| `../hdl/tb_ADCCapture.v` | self-checking Verilog testbench |
| `../hdl/model/adc_capture_model.py` | register-transfer model |
| `../firmware/adc_capture.py` | driver (`ol.adc`) and `spectrum()` |
| `../tests/test_adc_capture*.py` | tests |

## What is verified

| Item | State |
| --- | --- |
| Capture logic: input select, record length, delay, all trigger sources, stream gaps, re-arm | 20 tests pass on the Python model |
| Driver, run against that model; spectrum frequency, level in dBFS, averaging, aliasing | 14 tests pass |
| Verilog source, block design script, constraints | **not compiled, simulated or built** |
| Hardware | **not tested** |

## Bring-up

1. Simulate:
   ```bash
   iverilog -g2012 -o tb_adc hdl/tb_ADCCapture.v fpga/src/rtl/ADCCapture.v \
       fpga/src/rtl/AnalogTrigger.v && vvp tb_adc
   iverilog -g2012 -o tb_atrig hdl/tb_AnalogTrigger.v fpga/src/rtl/AnalogTrigger.v && vvp tb_atrig
   ```
   Expected last lines: `PASS: capture tests, no errors` and
   `PASS: analog trigger tests, no errors`.
2. Build: `RFSOC_AWG_SEQ=1 RFSOC_AWG_ADC=1 ./build_all.sh` (the ADC path needs the sequencer).

Points to check on the first run:

- **ADC settings in the converter IP.** Tiles and clocking follow the board's base overlay. The
  data-path values (real, no decimation, mixer bypassed, 8 samples per word) are my reading of
  the IP parameters: confirm them in the IP dialog, and that the stream is 128 bit at 614.4 MHz.
- **ADC reference clocks.** The script adds `adc0_clk` and `adc2_clk` inputs at 491.52 MHz, as
  the base overlay has them; the existing clock setup in `OverlayController` already programs
  that frequency.
- **Input labels.** `ADC_INPUTS` maps connector labels A to D onto the four streams from other
  RFSoC 4x2 firmware. Confirm with a cable and swap entries if needed.
- **Sample order.** A captured sine should be smooth; a sawtooth pattern every 8 or 16 samples
  means the word packing is reversed somewhere.
- **Timing at 614.4 MHz** for the four width converters.
- **Alignment to playback.** The ADC tiles are not in the multi-tile sync, so the delay between
  DAC output and capture is fixed within a power cycle but can change by a few samples between
  power cycles.

## Limits of this first version

- One input per capture; no simultaneous multi-channel recording.
- 53 us at full rate. Longer windows need decimation or recording into the DDR4.
- The spectrum is computed on the processor, a few updates per second at best; no live FFT in
  the FPGA, and not yet in the web app.
- No pre-trigger samples.
