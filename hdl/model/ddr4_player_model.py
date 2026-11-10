"""Register-transfer model of fpga/src/rtl/DDR4Streamer.v and awg_async_fifo.v.

Every register of the RTL has a field of the same name here and is updated with the same
equation, one clock edge at a time, in two independent clock domains. The model exists so the
control logic (credit accounting, prefill, trigger, gearbox, loop/last handling, abort) can be
exercised without an HDL simulator, and so the short-buffer depth can be sized against DDR4
controller stalls. It is a check of the algorithm, not of the Verilog text: run
hdl/tb_DDR4Streamer.v in a real simulator as well.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

MASK512 = (1 << 512) - 1
MASK256 = (1 << 256) - 1

W_IDLE, W_RUN, W_DRAIN = 0, 1, 2
R_WAIT, R_READY, R_RUN, R_DONE = 0, 1, 2, 3

SAMPLES_PER_WORD = 32
#: rate_sel -> (RFDC interpolation, samples per axis_clk cycle, axis_clk cycles per FIFO word)
RATE_MODES = {0: (2, 16, 2), 1: (4, 8, 4), 2: (8, 4, 8), 3: (16, 2, 16)}


def sample_value(global_index: int) -> int:
    """Deterministic 16-bit test pattern that depends on the absolute sample position."""
    return (global_index * 40503 + 12345) & 0xFFFF


def memory_word(word_index: int) -> int:
    value = 0
    base = word_index * SAMPLES_PER_WORD
    for i in range(SAMPLES_PER_WORD):
        value |= sample_value(base + i) << (16 * i)
    return value


def gray(x: int) -> int:
    return x ^ (x >> 1)


def gray2bin(g: int) -> int:
    b = 0
    while g:
        b ^= g
        g >>= 1
    return b


def gear(x: int, sel: int) -> int:
    half_bits = 8 * RATE_MODES[sel][1]
    lo = x & ((1 << half_bits) - 1)
    hi = (x >> half_bits) & ((1 << half_bits) - 1)
    return (hi << 256) | lo


def advance(x: int, sel: int) -> int:
    return x >> (16 * RATE_MODES[sel][1])


@dataclass
class Controls:
    enable: int = 0
    sw_trigger: int = 0
    ext_trigger: int = 0
    ext_trig_en: int = 0
    loop_forever: int = 0
    rate_sel: int = 2
    loop_count: int = 1
    base_page: int = 0
    num_words: int = 0


@dataclass
class MigProfile:
    """Behaviour of the DDR4 controller's AXI read channel, in ui_clk cycles."""

    latency: int = 24            # AR accepted -> first R beat
    max_pending: int = 4         # read bursts the controller queues
    refresh_period: int = 2340   # 7.8 us at 300 MHz
    refresh_stall: int = 240     # 0.8 us: precharge + tRFC + re-activate
    long_stall_period: int = 0   # optional rare long stall (calibration), 0 = off
    long_stall: int = 0
    gap_probability: float = 0.0  # random single-cycle rvalid gaps
    ar_gap_probability: float = 0.0


class MigModel:
    def __init__(self, profile: MigProfile, seed: int = 1):
        self.p = profile
        self.rng = random.Random(seed)
        self.cycle = 0
        self.queue: list[list[int]] = []   # [word_index, beats_left, ready_cycle]
        self.rvalid = 0
        self.rdata = 0
        self.arready = 0
        self.beats_served = 0
        self.max_queue = 0
        self.bad_address = False
        self.words: dict[int, int] = {}

    def word(self, index: int) -> int:
        if index not in self.words:
            self.words[index] = memory_word(index)
        return self.words[index]

    def stalled(self, cycle: int) -> bool:
        p = self.p
        if p.refresh_period and (cycle % p.refresh_period) < p.refresh_stall:
            return True
        if p.long_stall_period and (cycle % p.long_stall_period) < p.long_stall:
            return True
        return False

    def tick(self, arvalid: int, araddr: int, arlen: int, rready: int) -> None:
        """Advance one ui_clk edge using the DUT outputs that were valid before the edge."""
        if arvalid and self.arready:
            if araddr % 64 or (araddr % 4096) + (arlen + 1) * 64 > 4096:
                self.bad_address = True   # unaligned or crosses a 4 KiB boundary
            self.queue.append([araddr // 64, arlen + 1, self.cycle + self.p.latency])
            self.max_queue = max(self.max_queue, len(self.queue))
        if self.rvalid and rready:
            self.beats_served += 1
            head = self.queue[0]
            head[0] += 1
            head[1] -= 1
            if head[1] == 0:
                self.queue.pop(0)
            self.rvalid = 0

        self.cycle += 1
        stalled = self.stalled(self.cycle)

        self.arready = int(
            len(self.queue) < self.p.max_pending
            and not stalled
            and self.rng.random() >= self.p.ar_gap_probability
        )
        if not self.rvalid:
            if (
                self.queue
                and self.queue[0][2] <= self.cycle
                and not stalled
                and self.rng.random() >= self.p.gap_probability
            ):
                self.rvalid = 1
                self.rdata = self.word(self.queue[0][0])


class AsyncFifo:
    def __init__(self, aw: int):
        self.aw = aw
        self.depth = 1 << aw
        self.pmask = (1 << (aw + 1)) - 1
        self.mem = [0] * self.depth
        self.wbin = self.wgray = self.rbin = self.rgray = 0
        self.rgray_w1 = self.rgray_w2 = self.wgray_r1 = self.wgray_r2 = 0
        self.rd_valid = 0
        self.rd_data = 0
        self.overflow = False

    @property
    def wr_fill(self) -> int:
        return (self.wbin - gray2bin(self.rgray_w2)) & self.pmask

    @property
    def rd_fill(self) -> int:
        return ((gray2bin(self.wgray_r2) - self.rbin) & self.pmask) + self.rd_valid

    @property
    def true_fill(self) -> int:
        return ((self.wbin - self.rbin) & self.pmask) + self.rd_valid

    def wr_tick(self, wr_rst: int, wr_en: int, wr_data: int) -> None:
        wbin, rgray_w1, rgray = self.wbin, self.rgray_w1, self.rgray
        if wr_en:
            if not wr_rst and ((self.wbin - self.rbin) & self.pmask) >= self.depth:
                self.overflow = True
            self.mem[wbin & (self.depth - 1)] = wr_data
        if wr_rst:
            self.wbin = self.wgray = self.rgray_w1 = self.rgray_w2 = 0
        else:
            if wr_en:
                self.wbin = (wbin + 1) & self.pmask
                self.wgray = gray(self.wbin)
            self.rgray_w1 = rgray
            self.rgray_w2 = rgray_w1

    def rd_tick(self, rd_rst: int, rd_take: int) -> None:
        rd_empty = self.rgray == self.wgray_r2
        rd_fetch = (not rd_empty) and (not self.rd_valid or rd_take)
        rbin, wgray_r1, wgray = self.rbin, self.wgray_r1, self.wgray
        if rd_fetch:
            self.rd_data = self.mem[rbin & (self.depth - 1)]
        if rd_rst:
            self.rbin = self.rgray = 0
            self.rd_valid = 0
            self.wgray_r1 = self.wgray_r2 = 0
        else:
            self.wgray_r1 = wgray
            self.wgray_r2 = wgray_r1
            if rd_fetch:
                self.rbin = (rbin + 1) & self.pmask
                self.rgray = gray(self.rbin)
                self.rd_valid = 1
            elif rd_take:
                self.rd_valid = 0


class DDR4StreamerModel:
    def __init__(self, fifo_aw: int = 10, burst_beats: int = 64, margin: int = 16):
        self.FIFO_AW = fifo_aw
        self.BURST = burst_beats
        self.MARGIN = margin
        self.DEPTH = 1 << fifo_aw
        self.PREFILL = self.DEPTH - margin - 2 * burst_beats
        self.fifo = AsyncFifo(fifo_aw)
        self.ctl = Controls()

        # reader (m_axi_aclk domain)
        self.en_w1 = self.en_w2 = 0
        self.wstate = W_IDLE
        self.idle_cnt = 0
        self.base_r = self.addr_r = 0
        self.nwords_r = self.words_left = 0
        self.loops_left = 0
        self.forever_r = self.final_issued = self.final_acc = 0
        self.outstanding = 0
        self.pend_len = 0
        self.gap = 0
        self.credit_ok = 0
        self.arvalid_r = 0
        self.araddr_r = 0
        self.arlen_r = 0
        self.rd_err_q = 0
        self.wrun_q = self.ready_q = self.busy_q = 0

        # player (axis_clk domain)
        self.en_r1 = self.en_r2 = 0
        self.wrun_r1 = self.wrun_r2 = 0
        self.ready_r1 = self.ready_r2 = 0
        self.swt_r1 = self.swt_r2 = self.swt_r3 = 0
        self.ext_r1 = self.ext_r2 = self.ext_r3 = 0
        self.hold = 64
        self.rstate = R_WAIT
        self.cnt = 0
        self.sh = 0
        self.last_r = 0
        self.underflow_q = self.trig_early_q = 0
        self.m_axis_tdata = 0
        self.m_axis_tvalid = 0

        # instrumentation only (not in the RTL)
        self.tdata_is_data = 0

    # ------------------------------------------------------------------ outputs
    @property
    def m_axi_rready(self) -> int:
        return int(self.wstate != W_IDLE)

    @property
    def rd_active(self) -> int:
        return int(self.en_r2 and self.wrun_r2 and self.hold == 0)

    @property
    def status(self) -> dict:
        active = self.rd_active
        return {
            "busy": self.busy_q,
            "ready": int(self.rstate == R_READY and active),
            "running": int(self.rstate == R_RUN and active),
            "done": int(self.rstate == R_DONE and active),
            "underflow": self.underflow_q,
            "read_error": self.rd_err_q,
            "trig_early": self.trig_early_q,
            "armed": active,
        }

    # ------------------------------------------------------------------ m_axi_aclk edge
    def wr_tick(self, arready: int, rvalid: int, rdata: int, rresp: int = 0, aresetn: int = 1) -> None:
        c = self.ctl
        f = self.fifo
        wstate, outstanding, gap = self.wstate, self.outstanding, self.gap
        arvalid_r, final_issued, final_acc = self.arvalid_r, self.final_issued, self.final_acc
        words_left, loops_left, addr_r = self.words_left, self.loops_left, self.addr_r
        en_w2, idle_cnt, credit_ok = self.en_w2, self.idle_cnt, self.credit_ok

        ar_hs = arvalid_r and arready
        r_beat = rvalid and (wstate != W_IDLE)
        blen = self.BURST if words_left >= self.BURST else (words_left & 0xFF)
        all_rx = final_acc and outstanding == 0
        wr_fill = f.wr_fill
        need = wr_fill + outstanding + self.BURST + self.MARGIN

        fifo_wr_rst = int(wstate == W_IDLE)
        fifo_wr_en = int(bool(r_beat) and wstate == W_RUN)
        fifo_wr_data = (int(bool(final_acc) and outstanding == 1) << 512) | (rdata & MASK512)

        # non-blocking assignments, in RTL order
        self.en_w1, self.en_w2 = c.enable, self.en_w1
        self.credit_ok = int(need <= self.DEPTH)
        self.wrun_q = int(wstate == W_RUN)
        self.busy_q = int(wstate != W_IDLE)
        self.ready_q = int(wstate == W_RUN and (wr_fill >= self.PREFILL or all_rx))
        self.outstanding = outstanding + (self.pend_len if ar_hs else 0) - (1 if r_beat else 0)
        if r_beat and (rresp & 2):
            self.rd_err_q = 1

        if not aresetn:
            self.wstate = W_IDLE
            self.idle_cnt = 0
            self.arvalid_r = 0
            self.outstanding = 0
            self.final_issued = self.final_acc = 0
            self.gap = 0
            self.rd_err_q = 0
        elif wstate == W_IDLE:
            if idle_cnt != 31:
                self.idle_cnt = idle_cnt + 1
            if en_w2 and idle_cnt == 31 and c.num_words != 0:
                self.base_r = self.addr_r = c.base_page << 12
                self.nwords_r = self.words_left = c.num_words
                self.loops_left = 1 if c.loop_count == 0 else c.loop_count
                self.forever_r = c.loop_forever
                self.final_issued = self.final_acc = 0
                self.outstanding = 0
                self.gap = 3
                self.rd_err_q = 0
                self.wstate = W_RUN
        elif wstate == W_RUN:
            if gap != 0:
                self.gap = gap - 1
            if arvalid_r:
                if arready:
                    self.arvalid_r = 0
                    self.gap = 3
                    if final_issued:
                        self.final_acc = 1
            elif not en_w2:
                self.wstate = W_DRAIN
            elif gap == 0 and credit_ok and not final_issued:
                self.arvalid_r = 1
                self.araddr_r = addr_r
                self.arlen_r = blen - 1
                self.pend_len = blen
                if words_left == blen:
                    if self.forever_r or loops_left != 1:
                        self.addr_r = self.base_r
                        self.words_left = self.nwords_r
                        if not self.forever_r:
                            self.loops_left = loops_left - 1
                    else:
                        self.words_left = 0
                        self.final_issued = 1
                else:
                    self.addr_r = addr_r + (blen << 6)
                    self.words_left = words_left - blen
        elif wstate == W_DRAIN:
            if outstanding == 0:
                self.wstate = W_IDLE
                self.idle_cnt = 0

        f.wr_tick(fifo_wr_rst, fifo_wr_en, fifo_wr_data)

    # ------------------------------------------------------------------ axis_clk edge
    def rd_tick(self, tready: int = 1, aresetn: int = 1) -> None:
        c = self.ctl
        f = self.fifo
        rstate, cnt, sh, last_r = self.rstate, self.cnt, self.sh, self.last_r
        rd_active = self.rd_active
        trig = (self.swt_r2 and not self.swt_r3) or (c.ext_trig_en and self.ext_r2 and not self.ext_r3)
        sel = c.rate_sel
        rd_valid, rd_data = f.rd_valid, f.rd_data
        fifo_rd_rst = int(not rd_active)
        fifo_rd_take = int(rstate == R_RUN and tready and cnt == 0 and rd_valid)

        en_r2 = self.en_r2
        self.en_r1, self.en_r2 = c.enable, self.en_r1
        self.wrun_r1, self.wrun_r2 = self.wrun_q, self.wrun_r1
        self.ready_r1, self.ready_r2, ready_r2 = self.ready_q, self.ready_r1, self.ready_r2
        self.swt_r1, self.swt_r2, self.swt_r3 = c.sw_trigger, self.swt_r1, self.swt_r2
        self.ext_r1, self.ext_r2, self.ext_r3 = c.ext_trigger, self.ext_r1, self.ext_r2

        if not en_r2:
            self.hold = 64
        elif self.hold != 0:
            self.hold -= 1

        self.m_axis_tvalid = int(bool(aresetn))

        if not aresetn or not rd_active:
            self.rstate = R_WAIT
            self.cnt = 0
            self.last_r = 0
            self.underflow_q = self.trig_early_q = 0
            self.m_axis_tdata = 0
            self.tdata_is_data = 0
        elif rstate == R_WAIT:
            if trig:
                self.trig_early_q = 1
            if ready_r2:
                self.rstate = R_READY
        elif rstate == R_READY:
            if trig:
                self.rstate = R_RUN
                self.cnt = 0
        elif rstate == R_RUN:
            if tready:
                if cnt == 0:
                    if rd_valid:
                        self.m_axis_tdata = gear(rd_data & MASK512, sel)
                        self.sh = advance(rd_data & MASK512, sel)
                        self.last_r = (rd_data >> 512) & 1
                        self.cnt = RATE_MODES[sel][2] - 1
                        self.tdata_is_data = 1
                    else:
                        self.m_axis_tdata = 0
                        self.underflow_q = 1
                        self.tdata_is_data = 0
                else:
                    self.m_axis_tdata = gear(sh, sel)
                    self.sh = advance(sh, sel)
                    self.cnt = cnt - 1
                    self.tdata_is_data = 1
                    if cnt == 1 and last_r:
                        self.rstate = R_DONE
        else:
            if tready:
                self.m_axis_tdata = 0
                self.tdata_is_data = 0

        f.rd_tick(fifo_rd_rst, fifo_rd_take)


@dataclass
class Bench:
    """Two free-running clocks, a DDR4 controller model, and an output recorder."""

    dut: DDR4StreamerModel = field(default_factory=DDR4StreamerModel)
    mig: MigModel = field(default_factory=lambda: MigModel(MigProfile()))
    ui_period_ps: int = 3333          # 300 MHz DDR4 controller clock
    axis_period_ps: int = 3255        # 307.2 MHz stream clock
    tready_gap_probability: float = 0.0
    seed: int = 7

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.t_ui = 1000
        self.t_axis = 1700
        self.ui_cycles = 0
        self.axis_cycles = 0
        self.samples: list[int] = []
        self.first_data_cycle: int | None = None
        self.min_fill_running: int | None = None
        self.max_true_fill = 0
        self.record = True

    def step(self) -> None:
        dut, mig = self.dut, self.mig
        if self.t_ui <= self.t_axis:
            arvalid, araddr, arlen, rready = dut.arvalid_r, dut.araddr_r, dut.arlen_r, dut.m_axi_rready
            dut.wr_tick(mig.arready, mig.rvalid, mig.rdata)
            mig.tick(arvalid, araddr, arlen, rready)
            self.t_ui += self.ui_period_ps
            self.ui_cycles += 1
            self.max_true_fill = max(self.max_true_fill, dut.fifo.true_fill)
        else:
            tready = int(self.rng.random() >= self.tready_gap_probability)
            if tready and dut.m_axis_tvalid and dut.tdata_is_data:
                if self.first_data_cycle is None:
                    self.first_data_cycle = self.axis_cycles
                if self.record:
                    per_cycle = RATE_MODES[dut.ctl.rate_sel][1]
                    for half in (dut.m_axis_tdata & MASK256, dut.m_axis_tdata >> 256):
                        for i in range(per_cycle // 2):
                            self.samples.append((half >> (16 * i)) & 0xFFFF)
            if dut.rstate == R_RUN:
                fill = dut.fifo.true_fill
                if self.min_fill_running is None or fill < self.min_fill_running:
                    self.min_fill_running = fill
            dut.rd_tick(tready)
            self.t_axis += self.axis_period_ps
            self.axis_cycles += 1

    def run_axis_cycles(self, n: int) -> None:
        target = self.axis_cycles + n
        while self.axis_cycles < target:
            self.step()

    def run_until(self, predicate, max_axis_cycles: int = 2_000_000) -> bool:
        limit = self.axis_cycles + max_axis_cycles
        while self.axis_cycles < limit:
            if predicate():
                return True
            self.step()
        return predicate()


def expected_samples(base_page: int, num_words: int, passes: int) -> list[int]:
    first = (base_page << 12) // 64 * SAMPLES_PER_WORD
    one = [sample_value(first + i) for i in range(num_words * SAMPLES_PER_WORD)]
    return one * passes
