"""Register-transfer model of fpga/src/rtl/ADCCapture.v. Same caveat as the other models."""

from __future__ import annotations

from dataclasses import dataclass, field

C_IDLE, C_ARMED, C_DELAY, C_REC, C_DONE = range(5)

CTRL_ARM = 1 << 0
CTRL_SW_TRIGGER = 1 << 1
CTRL_SRC_SHIFT = 2
CTRL_ADC_SHIFT = 5


@dataclass
class Inputs:
    ctrl: int = 0
    delay: int = 0
    nrows: int = 0
    ext_trigger: int = 0
    dac0_status: int = 0
    dac2_status: int = 0
    tdata: list = field(default_factory=lambda: [0, 0, 0, 0])
    tvalid: list = field(default_factory=lambda: [0, 0, 0, 0])


class AdcCaptureModel:
    def __init__(self, mem_size_bytes: int = 524288):
        self.ROWS = mem_size_bytes // 32
        self.inp = Inputs()
        self.d_r = self.v_r = 0
        self.arm_1 = self.arm_2 = self.swt_1 = self.swt_2 = self.swt_3 = 0
        self.ext_1 = self.ext_2 = self.ext_3 = self.d0_1 = self.d2_1 = 0
        self.state = C_IDLE
        self.delay_left = self.rows_left = self.wr_row = 0
        self.mem_we = self.mem_addr = self.mem_wdata = 0
        self.memory: dict[int, int] = {}          # row -> word (test bench side)

    @property
    def status(self) -> dict:
        return {
            "armed": int(self.state == C_ARMED),
            "busy": int(self.state in (C_DELAY, C_REC)),
            "done": int(self.state == C_DONE),
            "rows": self.wr_row,
        }

    def tick(self, aresetn: int = 1) -> None:
        i = self.inp
        # memory write port (test bench side), using the outputs valid before this edge
        if self.mem_we:
            self.memory[self.mem_addr >> 5] = self.mem_wdata

        state, d_r, v_r = self.state, self.d_r, self.v_r
        src = i.ctrl >> CTRL_SRC_SHIFT & 7
        trig = (
            (self.swt_2 and not self.swt_3)
            or (src == 1 and self.ext_2 and not self.ext_3)
            or (src == 2 and (i.dac0_status >> 1 & 1) and not self.d0_1)
            or (src == 3 and (i.dac2_status >> 1 & 1) and not self.d2_1)
        )
        rows_cfg = self.ROWS if (i.nrows == 0 or i.nrows > self.ROWS) else i.nrows
        arm_2 = self.arm_2
        delay_left, rows_left, wr_row = self.delay_left, self.rows_left, self.wr_row

        sel = i.ctrl >> CTRL_ADC_SHIFT & 3
        self.d_r, self.v_r = i.tdata[sel], i.tvalid[sel]
        self.arm_1, self.arm_2 = i.ctrl & 1, self.arm_1
        self.swt_1, self.swt_2, self.swt_3 = i.ctrl >> 1 & 1, self.swt_1, self.swt_2
        self.ext_1, self.ext_2, self.ext_3 = i.ext_trigger, self.ext_1, self.ext_2
        self.d0_1, self.d2_1 = i.dac0_status >> 1 & 1, i.dac2_status >> 1 & 1
        self.mem_we = 0

        if not aresetn or not arm_2:
            self.state = C_IDLE
        elif state == C_IDLE:
            self.delay_left, self.rows_left, self.wr_row = i.delay, rows_cfg, 0
            self.state = C_ARMED
        elif state == C_ARMED:
            if trig:
                self.state = C_REC if delay_left == 0 else C_DELAY
        elif state == C_DELAY:
            if v_r:
                self.delay_left = delay_left - 1
                if delay_left == 1:
                    self.state = C_REC
        elif state == C_REC:
            if v_r:
                self.mem_wdata, self.mem_addr, self.mem_we = d_r, wr_row << 5, 1
                self.wr_row, self.rows_left = wr_row + 1, rows_left - 1
                if rows_left == 1:
                    self.state = C_DONE
