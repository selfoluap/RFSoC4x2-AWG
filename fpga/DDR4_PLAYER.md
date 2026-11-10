# PL DDR4 long-waveform player

Plays waveforms of seconds instead of microseconds by keeping them in the PL DDR4 and streaming
them to the DAC through a short FIFO.

**Status: logic verified on a register-transfer model, not yet simulated in an HDL simulator,
not yet built in Vivado, not yet run on a board.** See [What is verified](#what-is-verified).

## What you get

The DAC keeps sampling at 9.8304 GS/s. The DDR4 cannot supply that many 16-bit samples
(19.7 GB/s needed, 19.2 GB/s is the theoretical peak of DDR4-2400 x 64 bit), so the player feeds
the RF data converter at a lower rate and lets its interpolation filters do the rest.

| RFDC interpolation | DAC input rate | Usable bandwidth | DDR4 read load | One channel, 4 GiB | Two channels, 2 GiB each |
| --- | --- | --- | --- | --- | --- |
| 2x | 4915.2 MS/s | DC to ~1966 MHz | 9.8 GB/s | 0.44 s | 0.22 s |
| 4x | 2457.6 MS/s | DC to ~983 MHz | 4.9 GB/s | 0.87 s | 0.44 s |
| 8x | 1228.8 MS/s | DC to ~491 MHz | 2.5 GB/s | **1.75 s** | 0.87 s |
| 16x | 614.4 MS/s | DC to ~246 MHz | 1.2 GB/s | **3.50 s** | **1.75 s** |

Usable bandwidth is 80 % of the input Nyquist band (the pass band of the RFDC interpolation
filters). The full-rate BRAM player is untouched and stays available for short waveforms; a mux
per DAC selects the source.

Full-rate (9.8304 GS/s) playback of 1 s is not possible from raw samples on this board: it would
need 19.7 GB of storage.

## How it works

```
PS (loads waveform) --+
                      +--> SmartConnect --> DDR4 controller (MIG) <--> PL DDR4
DDR4Player (reads) ---+        300 MHz ui_clk, 512-bit AXI

DDR4Player:  AXI read master --> short FIFO (2048 x 64 bytes) --> rate gearbox --> mux --> existing
             (300 MHz domain)        gray-pointer CDC              (307.2 MHz)             clock/width
                                                                                           converters
```

- **Reader** (`DDR4Streamer`, DDR4 clock domain). Requests 4 KiB read bursts and keeps the FIFO
  topped up. A burst is only requested when FIFO space for it is already reserved, so the FIFO
  cannot overflow and the DDR4 read channel is never stalled.
- **Short buffer** (`awg_async_fifo`). Absorbs the DDR4 controller's refresh and calibration
  pauses. 2048 words by default (`FIFO_AW = 11`, about 30 block RAMs per channel).
- **Gearbox** (stream clock domain). Drains the FIFO at an exactly constant rate and formats the
  samples for the existing 512 -> 256 bit converter chain and the selected interpolation.
- **Start.** `enable` prefills the buffer and raises `ready`. A software or external trigger then
  starts playback a fixed number of clock cycles later, because the first samples are already
  waiting in the buffer.
- **End.** One-shot, N passes, or loop forever. Loops are seamless: the reader wraps to the start
  address without a gap.
- **Faults.** A sticky `underflow` flag is raised if the buffer ever runs dry, so a waveform with
  a gap cannot go unnoticed.

Buffer sizing, from the model with a full buffer of 2048 words:

| Interpolation | Longest DDR4 pause survived |
| --- | --- |
| 2x | 12 us |
| 4x | 25 us |
| 8x | 50 us |
| 16x | 100 us |

A DDR4 refresh pause is below 1 us. The pauses of the real controller have not been measured
yet; read the `underflow` flag on hardware and raise `FIFO_AW` if it ever sets.

## Files

| File | Purpose |
| --- | --- |
| `src/rtl/DDR4Streamer.v` | reader, short buffer, gearbox, trigger |
| `src/rtl/awg_async_fifo.v` | dual-clock FIFO |
| `src/rtl/DacSourceMux.v`, `src/rtl/DDR4Player.v` | BRAM/DDR4 source mux and block-design wrapper |
| `src/bd/add_ddr4_player.tcl` | adds DDR4 controller, SmartConnect, players and GPIOs to the block design |
| `src/constraints/ddr4_pl.xdc` | DDR4 pins (from the RFSoC4x2 base overlay), trigger pin, CDC constraints |
| `../hdl/tb_DDR4Streamer.v` | self-checking Verilog testbench |
| `../hdl/model/ddr4_player_model.py` | register-transfer model of the RTL |
| `../firmware/ddr4_player.py` | Python driver |
| `../tests/test_ddr4_player*.py` | model and driver tests |

## What is verified

| Item | State |
| --- | --- |
| Control logic: every played sample equals the stored sample, all four rates, lengths 1 to 50000 words, counted and endless loops, abort and re-arm, back-pressure, irregular DDR4 timing, fixed trigger latency, underflow detection | 45 tests pass on the Python register-transfer model (`pytest tests/test_ddr4_player_model.py`) |
| Python driver | 11 tests pass against fake GPIOs (`pytest tests/test_ddr4_player.py`) |
| Verilog source | **not compiled or simulated**: no simulator was available where this was written. The model mirrors the RTL register by register, but it cannot catch a typo in the Verilog |
| Block design script, constraints | **not run through Vivado** |
| Hardware | **not tested** |

## Bring-up order

1. Simulate:
   ```bash
   iverilog -g2012 -o tb_ddr4 hdl/tb_DDR4Streamer.v fpga/src/rtl/DDR4Streamer.v \
       fpga/src/rtl/awg_async_fifo.v && vvp tb_ddr4
   ```
   Expected last line: `PASS: 11 playback tests, no errors`.
2. Build with the DDR4 design enabled: `RFSOC_AWG_DDR4=1 ./build_all.sh`. Without the variable
   the reference design is built exactly as before.
3. On the board, check in this order: DDR4 calibration completes (PS can write and read back the
   window at `0x10_0000_0000`), `probe_ddr4_bytes` result, a short one-shot at 16x with a scope,
   then longer and faster.

Points that need a decision or a check on the first Vivado run:

- **Device package.** The project targets `xczu48dr-fsvg1517-2-e` with the ZCU208 board part. The
  RFSoC4x2 base overlay uses `xczu48dr-ffvg1517-2-e` with `realdigital.org:rfsoc4x2:part0:1.0`.
  The DDR4 controller uses package trace delays, so build the DDR4 variant for the `ffvg` part.
- **Module interfaces.** Check that Vivado infers `m_axi` (read-only AXI4 master), `s_axis` and
  `m_axis` on `DDR4Player` as intended.
- **Runtime interpolation change.** The driver switches the RFDC block with
  `InterpolationFactor` and `FabWrVldWords` while the stream clock stays at 614.4 MHz. Confirm on
  hardware that the block accepts this with the IP generated for 1x; if not, generate the IP with
  interpolation enabled and keep the BRAM player in a separate bitstream.
- **Multi-tile sync.** Re-run `sync_dac_tiles()` after changing the interpolation.
- **DDR4 size.** The board is documented with 4 GB of PL DDR4, the base overlay maps an 8 GiB
  window. The driver probes for aliasing and uses what is really there.
- **Timing at 307.2 MHz.** The FIFO read path is block RAM -> 4:1 mux -> register. If it fails
  timing, add a register stage after the FIFO.
- **Trigger pin.** `ext_trigger` is on PMOD0 pin 0, LVCMOS18 at the FPGA. Check the board manual
  for the voltage allowed at the connector.

## Use

```python
import numpy as np
from firmware import OverlayController

ol = OverlayController("rfsocawg_ddr4.bit")
player = ol.ddr4["dac0"]                 # empty dict when the bitstream has no DDR4

player.set_interpolation(8)              # 1228.8 MS/s input, DC to ~491 MHz
fs = 1228.8e6

def chirp(seconds, f0, f1, chunk=1 << 22):
    """80 -> 120 MHz chirp, generated in pieces so it never has to fit in RAM."""
    total = int(seconds * fs)
    for start in range(0, total, chunk):
        t = (start + np.arange(min(chunk, total - start))) / fs
        phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / seconds * t * t)
        yield (30000 * np.sin(phase)).astype(np.int16)

segment = player.load(chirp(1.5, 80e6, 120e6))
player.arm()                             # prefetch; arm(external_trigger=True) waits for the pin
player.trigger()
player.wait_done()                       # raises if the output had a gap
```

Loading is the slow part for GiB-size waveforms: samples are computed on the ARM core and
written through the PS AXI port. The time has not been measured.

## Limits of this first version

- Samples are stored as 16 bits although the DAC has 14. Packing would add 14 % of length.
- One segment per arm. A segment table (jump, repeat, wait for trigger between segments) is the
  natural next step and needs no change to the buffer or gearbox.
- After an underflow, playback continues late; it is not resynchronised.
- Interpolation factors are limited to 2, 4, 8 and 16, the ones that give a whole number of
  samples per stream beat.
