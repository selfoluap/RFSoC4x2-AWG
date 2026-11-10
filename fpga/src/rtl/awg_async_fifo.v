// -------------------------------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
//
// awg_async_fifo - dual-clock first-word-fall-through FIFO (gray-coded pointers).
//
// This is the "short buffer" between the PL DDR4 and the DAC stream. It is written in the
// DDR4 controller clock domain and read in the DAC stream clock domain.
//
// There is deliberately no "full" flag: the writer (DDR4Streamer) only requests DDR4 data it
// has reserved space for, using wr_fill, so the FIFO can never overflow and the DDR4 read
// channel never has to be stalled.
//
// Reset protocol: wr_rst and rd_rst are synchronous to their own clock. Both must be asserted
// together (overlapping) before the FIFO is reused. DDR4Streamer guarantees this.
// -------------------------------------------------------------------------------------------------
`timescale 1ns / 1ps

module awg_async_fifo #(
  parameter integer WIDTH = 513,
  parameter integer AW    = 10            // depth = 2**AW words
) (
  input  wire             wr_clk,
  input  wire             wr_rst,
  input  wire             wr_en,
  input  wire [WIDTH-1:0] wr_data,
  output wire [AW:0]      wr_fill,        // occupancy seen by the writer (never under-estimates)

  input  wire             rd_clk,
  input  wire             rd_rst,
  output reg              rd_valid,       // rd_data holds a word
  output reg  [WIDTH-1:0] rd_data,
  input  wire             rd_take,        // consume rd_data this cycle
  output wire [AW:0]      rd_fill         // occupancy seen by the reader (never over-estimates)
);

  (* ram_style = "block" *) reg [WIDTH-1:0] mem [0:(1<<AW)-1];

  reg [AW:0] wbin  = {(AW+1){1'b0}};
  reg [AW:0] wgray = {(AW+1){1'b0}};
  reg [AW:0] rbin  = {(AW+1){1'b0}};
  reg [AW:0] rgray = {(AW+1){1'b0}};

  (* ASYNC_REG = "TRUE" *) reg [AW:0] rgray_w1 = {(AW+1){1'b0}};
  (* ASYNC_REG = "TRUE" *) reg [AW:0] rgray_w2 = {(AW+1){1'b0}};
  (* ASYNC_REG = "TRUE" *) reg [AW:0] wgray_r1 = {(AW+1){1'b0}};
  (* ASYNC_REG = "TRUE" *) reg [AW:0] wgray_r2 = {(AW+1){1'b0}};

  function [AW:0] gray2bin;
    input [AW:0] g;
    integer i;
    begin
      gray2bin[AW] = g[AW];
      for (i = AW - 1; i >= 0; i = i - 1)
        gray2bin[i] = gray2bin[i+1] ^ g[i];
    end
  endfunction

  // ---------------------------------------------------------------- write side
  wire [AW:0] wbin_next = wbin + 1'b1;

  always @(posedge wr_clk) begin
    if (wr_en)
      mem[wbin[AW-1:0]] <= wr_data;

    if (wr_rst) begin
      wbin     <= {(AW+1){1'b0}};
      wgray    <= {(AW+1){1'b0}};
      rgray_w1 <= {(AW+1){1'b0}};
      rgray_w2 <= {(AW+1){1'b0}};
    end else begin
      if (wr_en) begin
        wbin  <= wbin_next;
        wgray <= wbin_next ^ (wbin_next >> 1);
      end
      rgray_w1 <= rgray;
      rgray_w2 <= rgray_w1;
    end
  end

  assign wr_fill = wbin - gray2bin(rgray_w2);

  // ---------------------------------------------------------------- read side
  wire [AW:0] rbin_next = rbin + 1'b1;
  wire        rd_empty  = (rgray == wgray_r2);
  wire        rd_fetch  = !rd_empty && (!rd_valid || rd_take);

  always @(posedge rd_clk) begin
    if (rd_fetch)
      rd_data <= mem[rbin[AW-1:0]];

    if (rd_rst) begin
      rbin     <= {(AW+1){1'b0}};
      rgray    <= {(AW+1){1'b0}};
      rd_valid <= 1'b0;
      wgray_r1 <= {(AW+1){1'b0}};
      wgray_r2 <= {(AW+1){1'b0}};
    end else begin
      wgray_r1 <= wgray;
      wgray_r2 <= wgray_r1;
      if (rd_fetch) begin
        rbin     <= rbin_next;
        rgray    <= rbin_next ^ (rbin_next >> 1);
        rd_valid <= 1'b1;
      end else if (rd_take) begin
        rd_valid <= 1'b0;
      end
    end
  end

  assign rd_fill = (gray2bin(wgray_r2) - rbin) + {{AW{1'b0}}, rd_valid};

  // ---------------------------------------------------------------- simulation checks
  // synthesis translate_off
  always @(posedge wr_clk)
    if (!wr_rst && wr_en && wr_fill[AW]) begin
      $display("ERROR %m: write into a full FIFO at %0t", $time);
      $stop;
    end
  // synthesis translate_on

endmodule
