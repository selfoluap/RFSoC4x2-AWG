"""Register-transfer model of fpga/src/rtl/DACSeqStreamer.v (single clock domain).

Same purpose and same caveat as ddr4_player_model.py: it checks the sequencing logic register by
register, not the Verilog text. Run hdl/tb_DACSeqStreamer.v in a real simulator as well.
"""

from __future__ import annotations

from dataclasses import dataclass

S_IDLE, S_ARMED, S_RUN, S_DONE = 0, 1, 2, 3
E_STOP, E_HOLD, E_LOOP = 0, 1, 2

CTRL_ARM = 1 << 0
CTRL_SW_TRIGGER = 1 << 1
CTRL_EXT_TRIG_EN = 1 << 2
CTRL_END_SHIFT = 3
CTRL_SEQ_MODE = 1 << 5


def row_value(row: int) -> int:
    """Content of memory row `row`; any non-zero value that identifies the row."""
    return (row * 2654435761 + 0x9E3779B9) & ((1 << 512) - 1) | 1


@dataclass
class Inputs:
    seq_ctrl: int = 0
    seq_lens: int = 0
    seq_repeats: int = 0
    ext_trigger: int = 0
    enable: int = 0          # legacy
    num_samples: int = 0     # legacy
    tready: int = 1


class SeqStreamerModel:
    def __init__(self, mem_size_bytes: int = 524288, read_latency: int = 3):
        self.ROWS = mem_size_bytes // 64
        self.RW = (self.ROWS - 1).bit_length()
        self.L = read_latency
        self.inp = Inputs()

        self.arm_1 = self.arm_2 = self.en_1 = self.en_2 = self.sm_1 = self.sm_2 = 0
        self.swt_1 = self.swt_2 = self.swt_3 = self.ext_1 = self.ext_2 = self.ext_3 = 0
        self.len_l, self.len_m1, self.nblk_m1, self.reps_m1 = 1, 0, 0, 0
        self.emode, self.seq_l = E_LOOP, 0
        self.state = S_IDLE
        self.row = self.row_left = self.base = 0
        self.rep_left = self.blk_left = self.blk = 0
        self.holding = self.stall_q = 0
        self.vd = [0] * (self.L + 1)
        self.addr = 0
        self.axis_tdata = 0

        # memory read pipeline (part of the test bench, not of the DUT)
        self.mem_pipe = [0] * self.L

    @property
    def rdata(self) -> int:
        return self.mem_pipe[-1]

    @property
    def status(self) -> dict:
        return {
            "armed": int(self.state == S_ARMED),
            "running": int(self.state == S_RUN),
            "done": int(self.state == S_DONE),
            "holding": self.holding,
            "stall": self.stall_q,
            "seq_mode": self.seq_l,
            "block": self.blk & 0x1FFF,
        }

    def tick(self, aresetn: int = 1) -> None:
        i = self.inp
        mask = (1 << self.RW) - 1
        state, row, row_left, base = self.state, self.row, self.row_left, self.base
        rep_left, blk_left, blk, holding = self.rep_left, self.blk_left, self.blk, self.holding
        arm_2, en_2, sm_2 = self.arm_2, self.en_2, self.sm_2
        active = arm_2 if sm_2 else en_2
        trig = (self.swt_2 and not self.swt_3) or (
            (i.seq_ctrl >> 2 & 1) and self.ext_2 and not self.ext_3
        )
        legacy_rows_w = (i.num_samples + 31) >> 5
        legacy_rows = self.ROWS if (i.num_samples == 0 or legacy_rows_w > self.ROWS) else legacy_rows_w
        lens16 = i.seq_lens & 0xFFFF
        seq_rows = 1 if lens16 == 0 else self.ROWS if lens16 > self.ROWS else lens16
        rdata, vd = self.rdata, list(self.vd)

        # memory model: data appears READ_LATENCY edges after the address register
        self.mem_pipe = [row_value(self.addr >> 6)] + self.mem_pipe[:-1]

        self.arm_1, self.arm_2 = i.seq_ctrl & 1, self.arm_1
        self.en_1, self.en_2 = i.enable, self.en_1
        self.sm_1, self.sm_2 = i.seq_ctrl >> 5 & 1, self.sm_1
        self.swt_1, self.swt_2, self.swt_3 = i.seq_ctrl >> 1 & 1, self.swt_1, self.swt_2
        self.ext_1, self.ext_2, self.ext_3 = i.ext_trigger, self.ext_1, self.ext_2

        self.addr = ((base + row) & mask) << 6
        self.vd = [int(state == S_RUN and bool(active))] + vd[:-1]
        self.axis_tdata = rdata if vd[self.L] else 0

        if (state == S_RUN or any(vd)) and not i.tready:
            self.stall_q = 1

        if not aresetn or not active:
            self.state = S_IDLE
            self.holding = 0
            if not aresetn:
                self.vd = [0] * (self.L + 1)
                self.stall_q = 0
        elif state == S_IDLE:
            rows = seq_rows if sm_2 else legacy_rows
            self.seq_l = sm_2
            self.len_l = rows
            self.len_m1 = rows - 1
            nblk = i.seq_lens >> 16 & 0xFFFF
            self.nblk_m1 = 0 if (not sm_2 or nblk == 0) else nblk - 1
            self.reps_m1 = 0 if (not sm_2 or i.seq_repeats == 0) else i.seq_repeats - 1
            self.emode = (i.seq_ctrl >> CTRL_END_SHIFT & 3) if sm_2 else E_LOOP
            self.stall_q = 0
            self.state = S_ARMED
        elif state == S_ARMED:
            self.row = 0
            self.row_left = self.len_m1
            self.base = 0
            self.rep_left = self.reps_m1
            self.blk_left = self.nblk_m1
            self.blk = 0
            self.holding = 0
            if not self.seq_l or trig:
                self.state = S_RUN
        elif state == S_RUN:
            if row_left != 0:
                self.row = row + 1
                self.row_left = row_left - 1
            else:
                self.row = 0
                self.row_left = self.len_m1
                if holding:
                    pass
                elif rep_left != 0:
                    self.rep_left = rep_left - 1
                else:
                    self.rep_left = self.reps_m1
                    if blk_left != 0:
                        self.base = (base + self.len_l) & mask
                        self.blk_left = blk_left - 1
                        self.blk = blk + 1
                    elif self.emode == E_STOP:
                        self.state = S_DONE
                    elif self.emode == E_HOLD:
                        self.holding = 1
                    else:
                        self.base = 0
                        self.blk_left = self.nblk_m1
                        self.blk = 0


def expected_rows(block_rows: int, n_blocks: int, repeats: int) -> list[int]:
    """Row order of one pass through the sequence."""
    rows = []
    for block in range(n_blocks):
        rows += list(range(block * block_rows, (block + 1) * block_rows)) * repeats
    return rows
