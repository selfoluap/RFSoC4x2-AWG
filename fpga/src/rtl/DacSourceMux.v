// -------------------------------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
//
// DacSourceMux - selects which player feeds one DAC channel.
//   sel = 0 : s0 (BRAM/URAM player, DACRAMstreamer, full sample rate, short waveforms)
//   sel = 1 : s1 (DDR4 player, DDR4Streamer, reduced input rate, long waveforms)
//
// `sel` is quasi-static: change it only while both players are disabled.
// The unselected source sees tready = 1 so it can never stall.
// -------------------------------------------------------------------------------------------------
`timescale 1ns / 1ps

module DacSourceMux #( parameter integer DWIDTH = 512 ) (
  input  wire              aclk,
  input  wire              sel,

  input  wire [DWIDTH-1:0] s0_axis_tdata,
  input  wire              s0_axis_tvalid,
  output wire              s0_axis_tready,

  input  wire [DWIDTH-1:0] s1_axis_tdata,
  input  wire              s1_axis_tvalid,
  output wire              s1_axis_tready,

  output wire [DWIDTH-1:0] m_axis_tdata,
  output wire              m_axis_tvalid,
  input  wire              m_axis_tready
);

  (* ASYNC_REG = "TRUE" *) reg sel_1 = 1'b0;
  (* ASYNC_REG = "TRUE" *) reg sel_2 = 1'b0;

  always @(posedge aclk) begin
    sel_1 <= sel;
    sel_2 <= sel_1;
  end

  assign m_axis_tdata   = sel_2 ? s1_axis_tdata  : s0_axis_tdata;
  assign m_axis_tvalid  = sel_2 ? s1_axis_tvalid : s0_axis_tvalid;
  assign s0_axis_tready = sel_2 ? 1'b1 : m_axis_tready;
  assign s1_axis_tready = sel_2 ? m_axis_tready : 1'b1;

endmodule
