"""Register-transfer model of fpga/src/rtl/AnalogTrigger.v. Same caveat as the other models."""

from __future__ import annotations

from dataclasses import dataclass, field

CTRL_ENABLE = 1 << 0
CTRL_ADC_SHIFT = 1
CTRL_MODE_SHIFT = 3
CTRL_CLEAR = 1 << 5
MODE_ENVELOPE, MODE_RISING, MODE_FALLING = 0, 1, 2


def signed16(x: int) -> int:
    x &= 0xFFFF
    return x - 0x10000 if x & 0x8000 else x


def word_from_samples(samples) -> int:
    value = 0
    for i, sample in enumerate(samples):
        value |= (int(sample) & 0xFFFF) << (16 * i)
    return value


@dataclass
class Inputs:
    ctrl: int = 0
    thr: int = 0
    release_words: int = 0
    tdata: list = field(default_factory=lambda: [0, 0, 0, 0])
    tvalid: list = field(default_factory=lambda: [0, 0, 0, 0])


class AnalogTriggerModel:
    def __init__(self):
        self.inp = Inputs()
        self.en_1 = self.en_2 = self.clr_1 = self.clr_2 = 0
        self.d_r = self.v_r = 0
        self.on_r = self.off_r = self.v1_r = 0
        self.pk4 = [0, 0, 0, 0]
        self.rel_cnt = self.events = self.peak = self.pk_word = self.v2_r = 0
        self.active = 0

    @property
    def status(self) -> dict:
        return {"active": self.active, "peak": self.peak, "events": self.events}

    def tick(self, aresetn: int = 1) -> None:
        i = self.inp
        mode = i.ctrl >> CTRL_MODE_SHIFT & 3
        thr_on, thr_off = i.thr & 0xFFFF, i.thr >> 16 & 0xFFFF

        # combinational stage on d_r
        mags, hit_on, hit_off = [], 0, 0
        for k in range(16):
            s = signed16(self.d_r >> (16 * k))
            m = 0x7FFF if s == -32768 else abs(s)
            mags.append(m)
            if mode == MODE_ENVELOPE:
                on, off = m >= thr_on, m >= thr_off
            elif mode == MODE_RISING:
                on, off = s >= signed16(thr_on), s >= signed16(thr_off)
            else:
                on, off = s <= signed16(thr_on), s <= signed16(thr_off)
            hit_on |= on
            hit_off |= off

        en_2, clr_2 = self.en_2, self.clr_2
        v1_r, v2_r, on_r, off_r = self.v1_r, self.v2_r, self.on_r, self.off_r
        active, rel_cnt, pk4, pk_word, peak = self.active, self.rel_cnt, self.pk4, self.pk_word, self.peak

        self.en_1, self.en_2 = i.ctrl & 1, self.en_1
        self.clr_1, self.clr_2 = i.ctrl >> 5 & 1, self.clr_1
        sel = i.ctrl >> CTRL_ADC_SHIFT & 3
        self.d_r, v_r, self.v_r = i.tdata[sel], self.v_r, i.tvalid[sel]
        self.on_r, self.off_r, self.v1_r = int(hit_on), int(hit_off), v_r
        self.pk4 = [max(mags[4 * g : 4 * g + 4]) for g in range(4)]
        self.pk_word, self.v2_r = max(pk4), v1_r

        if clr_2:
            self.peak = self.events = 0
        else:
            if v2_r and pk_word > peak:
                self.peak = pk_word
            if en_2 and v1_r and on_r and not active:
                self.events = (self.events + 1) & 0xFFFF

        if not aresetn or not en_2:
            self.active, self.rel_cnt = 0, 0
        elif v1_r:
            if on_r:
                self.active, self.rel_cnt = 1, i.release_words & 0xFFFF
            elif active:
                if off_r:
                    self.rel_cnt = i.release_words & 0xFFFF
                elif rel_cnt == 0:
                    self.active = 0
                else:
                    self.rel_cnt = rel_cnt - 1
