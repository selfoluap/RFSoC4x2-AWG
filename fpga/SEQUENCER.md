# Sequence mode for the on-chip-memory player

Plays stored blocks with repeat counts, at the full 9.8304 GS/s, on both DAC channels at once,
started by a software or external trigger.

**Status: logic verified on a register-transfer model, not yet simulated in an HDL simulator,
not yet built in Vivado, not yet run on a board.**

## What it does

The waveform memory holds `n_blocks` blocks of equal length. Each block is played `repeats`
times, then the next block starts on a sample-exact boundary:

```
block 0 x repeats -> block 1 x repeats -> ... -> last block x repeats -> stop | hold | loop
```

| End mode | After the last block |
| --- | --- |
| `stop` | output returns to mid-scale, `done` is set |
| `hold` | the last block keeps repeating until `stop()` |
| `loop` | starts again at block 0 |

| Limit | Value |
| --- | --- |
| Waveform memory per channel | 262,144 samples (8,192 rows of 32 samples, 16 UltraRAM), doubled from 131,072 |
| Block length | any multiple of 32 samples, same for all blocks |
| Blocks | up to 65,535, as long as they fit the memory |
| Repeats per block | 1 to 4,294,967,295 |
| Start | software trigger or rising edge on PMOD0 pin 0, fixed delay to the first sample |

Uses:

- **Doppler staircase** (design note, option B): 120 blocks of 1,952 samples, 168 repeats each,
  = 4.003 ms from 457.5 KiB.
- **Waveform A, then B:** two blocks, `repeats=1`, `end="hold"` or `"stop"`.
- **Triggered loop:** one block, `end="loop"`.

Compared with the DDR4 player: the sequencer runs on both channels at once, uploads in about a
millisecond and can run for any length of time, but the waveform has to be made of repeating
blocks. The DDR4 player plays arbitrary waveforms (0.25 s at full rate, one channel).

## Compatibility

`DACSeqStreamer` replaces `DACRAMstreamer` under the same cell name. With sequence mode off it
behaves like the old player: `enable` loops the first `numSamples` samples immediately. The
existing `ol.dac0` / `ol.dac2` API, notebooks and web app keep working, with one difference:
the larger memory holds 262,144 samples but the legacy length register is 18 bits, so the legacy
API cannot loop a waveform of exactly 262,144 samples. Use `ol.seq[...]` for that.

The output stream now carries mid-scale samples while nothing plays, where the old player stopped
the stream. This keeps the trigger-to-output delay fixed.

## Use

```python
from firmware import OverlayController, signals

ol = OverlayController("rfsocawg_seq.bit")
player = ol.seq["dac0"]                  # empty dict when the bitstream has no sequencer

blocks, info = signals.serrodyne_staircase(
    ratios=[1, 5, 3], freqs_hz=[-1.33e9, 0.0, 0.87e9],
    pattern_samples=1952, n_steps=120,
    amplitude=30000, sample_rate=9.8304e9,
    step_sign=+1,                        # -1 sweeps downwards
    predistort=None,                     # your per-block predistortion function goes here
)
print(info["grid_hz"], info["snapped_freqs_hz"], info["sweep_hz"])   # 5.036 MHz steps, 599.3 MHz

player.load_blocks(blocks)
player.arm(repeats=168, end="stop", external_trigger=True)   # waits for the TTL edge
player.wait_done(timeout=10)
```

`player.trigger()` starts it from software. `player.status()` reports `armed`, `running`, `done`,
`holding`, the current `block`, and `stall`.

## Files

| File | Purpose |
| --- | --- |
| `src/rtl/DACSeqStreamer.v` | the player |
| `src/bd/add_sequencer.tcl` | swaps it into the block design, adds GPIOs, enlarges the memory |
| `src/constraints/sequencer.xdc`, `ext_trigger.xdc` | CDC constraints, trigger pin |
| `../hdl/tb_DACSeqStreamer.v` | self-checking Verilog testbench |
| `../hdl/model/seq_streamer_model.py` | register-transfer model |
| `../firmware/sequencer.py`, `signals.serrodyne_staircase` | driver and staircase generator |
| `../tests/test_seq_streamer_model.py`, `test_sequencer.py` | tests |

## What is verified

| Item | State |
| --- | --- |
| Sequencing logic: block order, repeat counts, all end modes, gap-free output, the full 120 x 168 x 61-row staircase, abort and re-arm, fixed trigger delay, legacy mode | 18 tests pass on the Python model |
| Driver, run against that model, and the staircase generator against the design note's numbers | 9 tests pass |
| Verilog source, block design script, constraints | **not compiled, simulated or built** |
| Hardware | **not tested** |

## Bring-up

1. Simulate:
   ```bash
   iverilog -g2012 -o tb_seq hdl/tb_DACSeqStreamer.v fpga/src/rtl/DACSeqStreamer.v && vvp tb_seq
   ```
   Expected last line: `PASS: sequencer tests, no errors`.
2. Build: `RFSOC_AWG_SEQ=1 ./build_all.sh` (add `RFSOC_AWG_DDR4=1` for both).

Points to check on the first run:

- **Memory read latency.** The player assumes data appears 3 clock cycles after the address
  (`READ_LATENCY`, matching the block memory's setting). If it is off by one, the first row of a
  sequence is missing or a stale row appears: check the start of a one-shot on a scope and
  adjust the parameter.
- **Memory size.** The script enlarges both address windows to 512 KiB and moves them; confirm
  the block memory generator follows (16 UltraRAM per channel) and timing still closes.
- **Back-pressure.** The player expects the stream behind it to take one word per clock. The
  `stall` flag reports if it ever does not.
- **Trigger delay and jitter** are fixed in the model; the absolute delay through the converter
  has to be measured.

## Limits of this first version

- All blocks have the same length and the same repeat count.
- No per-step wait-for-trigger or jumps. A segment table would add them.
- Finer Doppler steps need longer patterns and more memory (see the design note); 2.5 MHz steps
  need 1.78 MiB per channel and do not fit.
