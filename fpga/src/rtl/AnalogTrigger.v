// -------------------------------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
//
// AnalogTrigger - activity detector on one ADC input, used as a trigger source.
//
// Looks at every 16-sample word (3.26 ns) of the selected ADC stream:
//
//   mode 0  envelope: any |sample| >= threshold     (RF burst present; works for any carrier)
//   mode 1  rising:   any  sample  >= threshold     (signed level)
//   mode 2  falling:  any  sample  <= threshold     (signed level)
//
// `active` goes high with the first word that meets the "on" threshold. It stays high while
// words keep meeting the "off" threshold (hysteresis) and drops after `release` consecutive
// words that do not. `release` bridges the zero crossings of a slow carrier: use at least one
// carrier period. The rising edge of `active` is the trigger event.
//
// The detection delay is fixed: 3 clock cycles from the word to `active`.
//
//   ctrl     [0] enable   [2:1] ADC input select   [4:3] mode   [5] clear peak and event count
//   thr      [15:0] on threshold   [31:16] off threshold   (ADC counts; signed in modes 1, 2)
//   release  [15:0] words below the off threshold before `active` drops
//   status   [0] active   [15:1] largest |sample| seen since the last clear (0..32767)
//            [31:16] trigger events since the last clear (wraps)
//
// The ADC inputs of the RFSoC 4x2 are AC-coupled, so a slow level does not reach the detector.
// -------------------------------------------------------------------------------------------------
`timescale 1ns / 1ps

module AnalogTrigger (
  input  wire         clk,
  input  wire         aresetn,
  input  wire [255:0] d0, d1, d2, d3,
  input  wire         v0, v1, v2, v3,
  input  wire [31:0]  ctrl,
  input  wire [31:0]  thr,
  input  wire [31:0]  release_words,
  output reg          active = 1'b0,
  output wire [31:0]  status
);

  (* ASYNC_REG = "TRUE" *) reg en_1 = 1'b0, en_2 = 1'b0;
  (* ASYNC_REG = "TRUE" *) reg clr_1 = 1'b0, clr_2 = 1'b0;

  wire [1:0]         mode    = ctrl[4:3];
  wire [15:0]        thr_on  = thr[15:0];
  wire [15:0]        thr_off = thr[31:16];

  // ------------------------------------------------------------ stage 0: input select
  reg [255:0] d_r = 256'd0;
  reg         v_r = 1'b0;

  // ------------------------------------------------------------ stage 1: per-sample compare
  reg         on_r = 1'b0, off_r = 1'b0, v1_r = 1'b0;
  reg [14:0]  pk4 [0:3];                 // largest |sample| of each group of four
  reg [15:0]  hit_on, hit_off;
  reg [14:0]  mag [0:15];
  reg signed [15:0] s;
  reg [14:0]  m;
  integer i, g;

  function [14:0] max2;
    input [14:0] a, b;
    begin max2 = (a > b) ? a : b; end
  endfunction

  always @(*) begin
    for (i = 0; i < 16; i = i + 1) begin
      s = d_r[16*i +: 16];
      // |s| saturated to 15 bits (-32768 becomes 32767)
      m = s[15] ? ((s == -16'sd32768) ? 15'h7FFF : (~s[14:0] + 15'd1)) : s[14:0];
      mag[i] = m;
      case (mode)
        2'd0: begin
          hit_on[i]  = ({1'b0, m} >= thr_on);
          hit_off[i] = ({1'b0, m} >= thr_off);
        end
        2'd1: begin
          hit_on[i]  = (s >= $signed(thr_on));
          hit_off[i] = (s >= $signed(thr_off));
        end
        default: begin
          hit_on[i]  = (s <= $signed(thr_on));
          hit_off[i] = (s <= $signed(thr_off));
        end
      endcase
    end
  end

  // ------------------------------------------------------------ stage 2: state
  reg [15:0] rel_cnt = 16'd0;
  reg [15:0] events  = 16'd0;
  reg [14:0] peak    = 15'd0;
  reg [14:0] pk_word = 15'd0;
  reg        v2_r    = 1'b0;

  always @(posedge clk) begin
    en_1  <= ctrl[0];  en_2  <= en_1;
    clr_1 <= ctrl[5];  clr_2 <= clr_1;

    case (ctrl[2:1])
      2'd0:    begin d_r <= d0; v_r <= v0; end
      2'd1:    begin d_r <= d1; v_r <= v1; end
      2'd2:    begin d_r <= d2; v_r <= v2; end
      default: begin d_r <= d3; v_r <= v3; end
    endcase

    on_r  <= |hit_on;
    off_r <= |hit_off;
    v1_r  <= v_r;
    for (g = 0; g < 4; g = g + 1)
      pk4[g] <= max2(max2(mag[4*g], mag[4*g+1]), max2(mag[4*g+2], mag[4*g+3]));

    pk_word <= max2(max2(pk4[0], pk4[1]), max2(pk4[2], pk4[3]));
    v2_r    <= v1_r;

    if (clr_2) begin
      peak   <= 15'd0;
      events <= 16'd0;
    end else begin
      if (v2_r && pk_word > peak)
        peak <= pk_word;
      if (en_2 && v1_r && on_r && !active)
        events <= events + 16'd1;
    end

    if (!aresetn || !en_2) begin
      active  <= 1'b0;
      rel_cnt <= 16'd0;
    end else if (v1_r) begin
      if (on_r) begin
        active  <= 1'b1;
        rel_cnt <= release_words[15:0];
      end else if (active) begin
        if (off_r)
          rel_cnt <= release_words[15:0];
        else if (rel_cnt == 16'd0)
          active <= 1'b0;
        else
          rel_cnt <= rel_cnt - 16'd1;
      end
    end
  end

  assign status = {events, peak, active};

endmodule
