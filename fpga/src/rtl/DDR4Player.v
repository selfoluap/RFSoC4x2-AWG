// -------------------------------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
//
// DDR4Player - block-design wrapper for one DAC channel.
//
//   s_axis (BRAM player) --+
//                          +--> DacSourceMux --> m_axis --> axis_clock_converter --> ... --> RFDC
//   PL DDR4 --> DDR4Streamer
//
// The 32-bit control word comes straight from an AXI GPIO:
//   [0]     enable         arm the DDR4 player (0 = abort, hold low >= 1 ms)
//   [1]     sw_trigger     rising edge starts playback
//   [2]     ext_trig_en    also start on a rising edge of ext_trigger
//   [3]     loop_forever
//   [5:4]   rate_sel       0: 2x  1: 4x  2: 8x  3: 16x RFDC interpolation
//   [6]     source_sel     0: BRAM player drives the DAC   1: DDR4 player drives the DAC
//   [31:8]  loop_count     passes through the waveform (0 is treated as 1)
// -------------------------------------------------------------------------------------------------
`timescale 1ns / 1ps

module DDR4Player #(
  parameter integer ADDR_WIDTH = 33,
  parameter integer FIFO_AW    = 11
) (
  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 m_axi_aclk CLK" *)
  (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF m_axi, ASSOCIATED_RESET m_axi_aresetn" *)
  input  wire                    m_axi_aclk,
  (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 m_axi_aresetn RST" *)
  input  wire                    m_axi_aresetn,
  output wire [ADDR_WIDTH-1:0]   m_axi_araddr,
  output wire [7:0]              m_axi_arlen,
  output wire [2:0]              m_axi_arsize,
  output wire [1:0]              m_axi_arburst,
  output wire                    m_axi_arlock,
  output wire [3:0]              m_axi_arcache,
  output wire [2:0]              m_axi_arprot,
  output wire [3:0]              m_axi_arqos,
  output wire                    m_axi_arvalid,
  input  wire                    m_axi_arready,
  input  wire [511:0]            m_axi_rdata,
  input  wire [1:0]              m_axi_rresp,
  input  wire                    m_axi_rlast,
  input  wire                    m_axi_rvalid,
  output wire                    m_axi_rready,

  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 axis_clk CLK" *)
  (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axis:m_axis, ASSOCIATED_RESET axis_aresetn" *)
  input  wire                    axis_clk,
  (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 axis_aresetn RST" *)
  input  wire                    axis_aresetn,

  input  wire [511:0]            s_axis_tdata,
  input  wire                    s_axis_tvalid,
  output wire                    s_axis_tready,

  output wire [511:0]            m_axis_tdata,
  output wire                    m_axis_tvalid,
  input  wire                    m_axis_tready,

  input  wire [31:0]             ctrl,
  input  wire [ADDR_WIDTH-13:0]  base_page,
  input  wire [27:0]             num_words,
  input  wire                    ext_trigger,
  output wire [31:0]             status
);

  wire [511:0] ddr_tdata;
  wire         ddr_tvalid;
  wire         ddr_tready;

  DDR4Streamer #( .ADDR_WIDTH(ADDR_WIDTH), .FIFO_AW(FIFO_AW) ) u_streamer (
    .m_axi_aclk    (m_axi_aclk),
    .m_axi_aresetn (m_axi_aresetn),
    .m_axi_araddr  (m_axi_araddr),
    .m_axi_arlen   (m_axi_arlen),
    .m_axi_arsize  (m_axi_arsize),
    .m_axi_arburst (m_axi_arburst),
    .m_axi_arlock  (m_axi_arlock),
    .m_axi_arcache (m_axi_arcache),
    .m_axi_arprot  (m_axi_arprot),
    .m_axi_arqos   (m_axi_arqos),
    .m_axi_arvalid (m_axi_arvalid),
    .m_axi_arready (m_axi_arready),
    .m_axi_rdata   (m_axi_rdata),
    .m_axi_rresp   (m_axi_rresp),
    .m_axi_rlast   (m_axi_rlast),
    .m_axi_rvalid  (m_axi_rvalid),
    .m_axi_rready  (m_axi_rready),
    .axis_clk      (axis_clk),
    .axis_aresetn  (axis_aresetn),
    .m_axis_tdata  (ddr_tdata),
    .m_axis_tvalid (ddr_tvalid),
    .m_axis_tready (ddr_tready),
    .enable        (ctrl[0]),
    .sw_trigger    (ctrl[1]),
    .ext_trigger   (ext_trigger),
    .ext_trig_en   (ctrl[2]),
    .loop_forever  (ctrl[3]),
    .rate_sel      (ctrl[5:4]),
    .loop_count    (ctrl[31:8]),
    .base_page     (base_page),
    .num_words     (num_words),
    .status        (status)
  );

  DacSourceMux #( .DWIDTH(512) ) u_mux (
    .aclk           (axis_clk),
    .sel            (ctrl[6]),
    .s0_axis_tdata  (s_axis_tdata),
    .s0_axis_tvalid (s_axis_tvalid),
    .s0_axis_tready (s_axis_tready),
    .s1_axis_tdata  (ddr_tdata),
    .s1_axis_tvalid (ddr_tvalid),
    .s1_axis_tready (ddr_tready),
    .m_axis_tdata   (m_axis_tdata),
    .m_axis_tvalid  (m_axis_tvalid),
    .m_axis_tready  (m_axis_tready)
  );

endmodule
