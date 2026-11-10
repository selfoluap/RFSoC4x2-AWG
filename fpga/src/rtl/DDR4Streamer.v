// -------------------------------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
//
// DDR4Streamer - plays a long waveform out of the PL DDR4 into the DAC stream.
//
//   PL DDR4 (MIG AXI4, 512 bit) --> AXI read master --> short FIFO --> rate gearbox --> AXIS 512 bit
//        ui_clk domain (300 MHz)                         |        axis_clk domain (307.2 MHz)
//
// The DDR4 cannot be read sample-synchronously: the controller stalls for refresh and
// calibration. The reader therefore runs ahead of the DAC and keeps a short FIFO topped up; the
// DAC side drains that FIFO at an exactly constant rate. Read bursts are only requested when
// FIFO space for them is already reserved, so the FIFO can never overflow and never has to
// back-pressure the DDR4.
//
// Waveform layout in memory: little-endian int16 samples, 32 samples per 64-byte word, starting
// at a 4 KiB aligned address. Length is given in 64-byte words (pad the tail with zeros).
//
// Output format: the existing chain behind this block (axis_clock_converter -> 512:256
// axis_dwidth_converter -> RFDC at 614.4 MHz) splits each 512-bit word into two 256-bit beats.
// With RFDC interpolation N the converter only takes the low 16/N samples of each beat, so the
// gearbox places 16/N samples in the low end of each 256-bit half:
//
//   rate_sel   RFDC interpolation   DAC input rate   samples per axis_clk cycle   cycles per FIFO word
//      0              2x             4915.2 MS/s               16                        2
//      1              4x             2457.6 MS/s                8                        4
//      2              8x             1228.8 MS/s                4                        8
//      3             16x              614.4 MS/s                2                       16
//
// Control inputs are quasi-static levels (AXI GPIO). They must be stable before `enable` rises.
//   enable       1 = arm: reader starts prefetching.  0 = abort / idle (hold low >= 1 ms).
//   sw_trigger   rising edge starts playback once status `ready` is set.
//   ext_trigger  rising edge starts playback when ext_trig_en = 1 (asynchronous pin is fine).
//
// Trigger to first sample is a fixed number of axis_clk cycles because the data is already
// waiting in the FIFO; the only uncertainty is the one-cycle trigger synchroniser.
// -------------------------------------------------------------------------------------------------
`timescale 1ns / 1ps

module DDR4Streamer #(
  parameter integer ADDR_WIDTH  = 33,     // RFSoC4x2 PL DDR4 MIG AXI address width
  parameter integer FIFO_AW     = 11,     // short buffer depth = 2**FIFO_AW 64-byte words (2048 = 131 KiB)
  parameter integer BURST_BEATS = 64,     // 64 beats x 64 bytes = 4 KiB, never crosses a 4 KiB boundary
  parameter integer MARGIN      = 16      // FIFO words kept free as slack
) (
  // ------------------------------------------------------------ DDR4 controller side (read only)
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

  // ------------------------------------------------------------ DAC stream side
  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 axis_clk CLK" *)
  (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF m_axis, ASSOCIATED_RESET axis_aresetn" *)
  input  wire                    axis_clk,
  (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 axis_aresetn RST" *)
  input  wire                    axis_aresetn,
  output reg  [511:0]            m_axis_tdata,
  output reg                     m_axis_tvalid,
  input  wire                    m_axis_tready,

  // ------------------------------------------------------------ control (quasi-static)
  input  wire                    enable,
  input  wire                    sw_trigger,
  input  wire                    ext_trigger,
  input  wire                    ext_trig_en,
  input  wire                    loop_forever,
  input  wire [1:0]              rate_sel,
  input  wire [23:0]             loop_count,     // number of passes, 0 is treated as 1
  input  wire [ADDR_WIDTH-13:0]  base_page,      // start address / 4096
  input  wire [27:0]             num_words,      // waveform length in 64-byte words

  // ------------------------------------------------------------ status (mixed domains, levels)
  //  [0] busy        reader not idle (wait for 0 before re-arming)
  //  [1] ready       buffer prefilled, waiting for a trigger
  //  [2] running
  //  [3] done        one-shot / counted playback finished
  //  [4] underflow   sticky: the short buffer ran dry during playback
  //  [5] read_error  sticky: DDR4 AXI read returned an error response
  //  [6] trig_early  sticky: a trigger arrived before `ready`
  //  [7] armed       DAC side out of reset
  //  [16+FIFO_AW:16]  short-buffer fill level seen by the DAC side (debug, not sampled coherently)
  output wire [31:0]             status
);

  localparam integer DEPTH   = 1 << FIFO_AW;
  // prefill to (almost) the steady-state level, so stall tolerance is the same from the first sample
  localparam integer PREFILL = DEPTH - MARGIN - 2 * BURST_BEATS;
  localparam integer CW      = FIFO_AW + 3;

  localparam [1:0] W_IDLE = 2'd0, W_RUN = 2'd1, W_DRAIN = 2'd2;
  localparam [1:0] R_WAIT = 2'd0, R_READY = 2'd1, R_RUN = 2'd2, R_DONE = 2'd3;

  // ================================================================ FIFO
  wire               fifo_wr_rst;
  wire               fifo_wr_en;
  wire [512:0]       fifo_wr_data;
  wire [FIFO_AW:0]   fifo_wr_fill;
  wire               fifo_rd_rst;
  wire               fifo_rd_valid;
  wire [512:0]       fifo_rd_data;
  wire               fifo_rd_take;
  wire [FIFO_AW:0]   fifo_rd_fill;

  awg_async_fifo #( .WIDTH(513), .AW(FIFO_AW) ) u_fifo (
    .wr_clk  (m_axi_aclk),
    .wr_rst  (fifo_wr_rst),
    .wr_en   (fifo_wr_en),
    .wr_data (fifo_wr_data),
    .wr_fill (fifo_wr_fill),
    .rd_clk  (axis_clk),
    .rd_rst  (fifo_rd_rst),
    .rd_valid(fifo_rd_valid),
    .rd_data (fifo_rd_data),
    .rd_take (fifo_rd_take),
    .rd_fill (fifo_rd_fill)
  );

  // ================================================================ reader (m_axi_aclk domain)
  (* ASYNC_REG = "TRUE" *) reg en_w1 = 1'b0;
  (* ASYNC_REG = "TRUE" *) reg en_w2 = 1'b0;

  reg [1:0]             wstate       = W_IDLE;
  reg [4:0]             idle_cnt     = 5'd0;
  reg [ADDR_WIDTH-1:0]  base_r       = {ADDR_WIDTH{1'b0}};
  reg [ADDR_WIDTH-1:0]  addr_r       = {ADDR_WIDTH{1'b0}};
  reg [27:0]            nwords_r     = 28'd0;
  reg [27:0]            words_left   = 28'd0;
  reg [23:0]            loops_left   = 24'd0;
  reg                   forever_r    = 1'b0;
  reg                   final_issued = 1'b0;   // last read burst of the last pass has been built
  reg                   final_acc    = 1'b0;   // ... and accepted by the DDR4 controller
  reg [CW-1:0]          outstanding  = {CW{1'b0}};  // beats requested but not yet received
  reg [7:0]             pend_len     = 8'd0;
  reg [1:0]             gap          = 2'd0;
  reg                   credit_ok    = 1'b0;
  reg                   arvalid_r    = 1'b0;
  reg [ADDR_WIDTH-1:0]  araddr_r     = {ADDR_WIDTH{1'b0}};
  reg [7:0]             arlen_r      = 8'd0;
  reg                   rd_err_q     = 1'b0;
  reg                   wrun_q       = 1'b0;   // level to the DAC side: FIFO write side is live
  reg                   ready_q      = 1'b0;   // level to the DAC side: prefill reached
  reg                   busy_q       = 1'b0;

  wire        ar_hs    = arvalid_r && m_axi_arready;
  wire        r_beat   = m_axi_rvalid && m_axi_rready;
  wire [7:0]  blen     = (words_left >= BURST_BEATS) ? BURST_BEATS[7:0] : words_left[7:0];
  wire        all_rx   = final_acc && (outstanding == {CW{1'b0}});

  wire [CW:0] need     = {{(CW-FIFO_AW){1'b0}}, fifo_wr_fill} + {1'b0, outstanding}
                         + BURST_BEATS[CW:0] + MARGIN[CW:0];

  assign m_axi_araddr  = araddr_r;
  assign m_axi_arlen   = arlen_r;
  assign m_axi_arsize  = 3'b110;             // 64 bytes per beat
  assign m_axi_arburst = 2'b01;              // INCR
  assign m_axi_arlock  = 1'b0;
  assign m_axi_arcache = 4'b0011;
  assign m_axi_arprot  = 3'b000;
  assign m_axi_arqos   = 4'b0000;
  assign m_axi_arvalid = arvalid_r;
  assign m_axi_rready  = (wstate != W_IDLE);

  assign fifo_wr_rst   = (wstate == W_IDLE);
  assign fifo_wr_en    = r_beat && (wstate == W_RUN);
  assign fifo_wr_data  = {final_acc && (outstanding == {{(CW-1){1'b0}}, 1'b1}), m_axi_rdata};

  always @(posedge m_axi_aclk) begin
    en_w1 <= enable;
    en_w2 <= en_w1;

    credit_ok <= (need <= DEPTH[CW:0]);
    wrun_q    <= (wstate == W_RUN);
    busy_q    <= (wstate != W_IDLE);
    ready_q   <= (wstate == W_RUN) && ((fifo_wr_fill >= PREFILL[FIFO_AW:0]) || all_rx);

    outstanding <= outstanding + (ar_hs ? {{(CW-8){1'b0}}, pend_len} : {CW{1'b0}})
                               - (r_beat ? {{(CW-1){1'b0}}, 1'b1} : {CW{1'b0}});

    if (r_beat && m_axi_rresp[1])
      rd_err_q <= 1'b1;

    if (!m_axi_aresetn) begin
      wstate       <= W_IDLE;
      idle_cnt     <= 5'd0;
      arvalid_r    <= 1'b0;
      outstanding  <= {CW{1'b0}};
      final_issued <= 1'b0;
      final_acc    <= 1'b0;
      gap          <= 2'd0;
      rd_err_q     <= 1'b0;
    end else begin
      case (wstate)
        // ------------------------------------------------------------------------------------
        W_IDLE: begin
          if (idle_cnt != 5'd31)
            idle_cnt <= idle_cnt + 5'd1;

          if (en_w2 && idle_cnt == 5'd31 && num_words != 28'd0) begin
            base_r       <= {base_page, 12'd0};
            addr_r       <= {base_page, 12'd0};
            nwords_r     <= num_words;
            words_left   <= num_words;
            loops_left   <= (loop_count == 24'd0) ? 24'd1 : loop_count;
            forever_r    <= loop_forever;
            final_issued <= 1'b0;
            final_acc    <= 1'b0;
            outstanding  <= {CW{1'b0}};
            gap          <= 2'd3;
            rd_err_q     <= 1'b0;
            wstate       <= W_RUN;
          end
        end

        // ------------------------------------------------------------------------------------
        W_RUN: begin
          if (gap != 2'd0)
            gap <= gap - 2'd1;

          if (arvalid_r) begin
            // AXI: a request must be held until it is accepted, even when aborting
            if (m_axi_arready) begin
              arvalid_r <= 1'b0;
              gap       <= 2'd3;        // let credit_ok catch up with the new `outstanding`
              if (final_issued)
                final_acc <= 1'b1;
            end
          end else if (!en_w2) begin
            wstate <= W_DRAIN;
          end else if (gap == 2'd0 && credit_ok && !final_issued) begin
            arvalid_r <= 1'b1;
            araddr_r  <= addr_r;
            arlen_r   <= blen - 8'd1;
            pend_len  <= blen;

            if (words_left == {20'd0, blen}) begin
              // this burst completes a pass through the waveform
              if (forever_r || loops_left != 24'd1) begin
                addr_r     <= base_r;
                words_left <= nwords_r;
                if (!forever_r)
                  loops_left <= loops_left - 24'd1;
              end else begin
                words_left   <= 28'd0;
                final_issued <= 1'b1;
              end
            end else begin
              addr_r     <= addr_r + {{(ADDR_WIDTH-14){1'b0}}, blen, 6'd0};
              words_left <= words_left - {20'd0, blen};
            end
          end
        end

        // ------------------------------------------------------------------------------------
        W_DRAIN: begin
          // data already requested still arrives and is discarded (fifo_wr_en is off)
          if (outstanding == {CW{1'b0}}) begin
            wstate   <= W_IDLE;
            idle_cnt <= 5'd0;
          end
        end

        default: wstate <= W_IDLE;
      endcase
    end
  end

  // ================================================================ player (axis_clk domain)
  (* ASYNC_REG = "TRUE" *) reg en_r1    = 1'b0;
  (* ASYNC_REG = "TRUE" *) reg en_r2    = 1'b0;
  (* ASYNC_REG = "TRUE" *) reg wrun_r1  = 1'b0;
  (* ASYNC_REG = "TRUE" *) reg wrun_r2  = 1'b0;
  (* ASYNC_REG = "TRUE" *) reg ready_r1 = 1'b0;
  (* ASYNC_REG = "TRUE" *) reg ready_r2 = 1'b0;
  (* ASYNC_REG = "TRUE" *) reg swt_r1   = 1'b0;
  (* ASYNC_REG = "TRUE" *) reg swt_r2   = 1'b0;
  (* ASYNC_REG = "TRUE" *) reg ext_r1   = 1'b0;
  (* ASYNC_REG = "TRUE" *) reg ext_r2   = 1'b0;
  reg swt_r3 = 1'b0;
  reg ext_r3 = 1'b0;

  reg [6:0]    hold         = 7'd64;   // keeps the DAC side in reset after `enable` was low
  reg [1:0]    rstate       = R_WAIT;
  reg [3:0]    cnt          = 4'd0;
  reg [511:0]  sh           = 512'd0;
  reg          last_r       = 1'b0;
  reg          underflow_q  = 1'b0;
  reg          trig_early_q = 1'b0;

  wire rd_active = en_r2 && wrun_r2 && (hold == 7'd0);
  wire trig      = (swt_r2 && !swt_r3) || (ext_trig_en && ext_r2 && !ext_r3);

  assign fifo_rd_rst  = !rd_active;
  assign fifo_rd_take = (rstate == R_RUN) && m_axis_tready && (cnt == 4'd0) && fifo_rd_valid;

  // samples for this cycle: half go to the low end of tdata[255:0], half to tdata[511:256]
  function [511:0] gear;
    input [511:0] x;
    input [1:0]   sel;
    begin
      case (sel)
        2'd0:    gear = {128'd0, x[255:128], 128'd0, x[127:0]};
        2'd1:    gear = {192'd0, x[127:64],  192'd0, x[63:0]};
        2'd2:    gear = {224'd0, x[63:32],   224'd0, x[31:0]};
        default: gear = {240'd0, x[31:16],   240'd0, x[15:0]};
      endcase
    end
  endfunction

  function [511:0] advance;
    input [511:0] x;
    input [1:0]   sel;
    begin
      case (sel)
        2'd0:    advance = {256'd0, x[511:256]};
        2'd1:    advance = {128'd0, x[511:128]};
        2'd2:    advance = {64'd0,  x[511:64]};
        default: advance = {32'd0,  x[511:32]};
      endcase
    end
  endfunction

  function [3:0] last_cycle;          // cycles per FIFO word, minus one
    input [1:0] sel;
    begin
      case (sel)
        2'd0:    last_cycle = 4'd1;
        2'd1:    last_cycle = 4'd3;
        2'd2:    last_cycle = 4'd7;
        default: last_cycle = 4'd15;
      endcase
    end
  endfunction

  always @(posedge axis_clk) begin
    en_r1    <= enable;
    en_r2    <= en_r1;
    wrun_r1  <= wrun_q;
    wrun_r2  <= wrun_r1;
    ready_r1 <= ready_q;
    ready_r2 <= ready_r1;
    swt_r1   <= sw_trigger;
    swt_r2   <= swt_r1;
    swt_r3   <= swt_r2;
    ext_r1   <= ext_trigger;
    ext_r2   <= ext_r1;
    ext_r3   <= ext_r2;

    if (!en_r2)
      hold <= 7'd64;
    else if (hold != 7'd0)
      hold <= hold - 7'd1;

    m_axis_tvalid <= axis_aresetn;

    if (!axis_aresetn || !rd_active) begin
      rstate       <= R_WAIT;
      cnt          <= 4'd0;
      last_r       <= 1'b0;
      underflow_q  <= 1'b0;
      trig_early_q <= 1'b0;
      m_axis_tdata <= 512'd0;
    end else begin
      case (rstate)
        R_WAIT: begin
          if (trig)
            trig_early_q <= 1'b1;
          if (ready_r2)
            rstate <= R_READY;
        end

        R_READY: begin
          if (trig) begin
            rstate <= R_RUN;
            cnt    <= 4'd0;
          end
        end

        R_RUN: begin
          if (m_axis_tready) begin
            if (cnt == 4'd0) begin
              if (fifo_rd_valid) begin
                m_axis_tdata <= gear(fifo_rd_data[511:0], rate_sel);
                sh           <= advance(fifo_rd_data[511:0], rate_sel);
                last_r       <= fifo_rd_data[512];
                cnt          <= last_cycle(rate_sel);
              end else begin
                // short buffer ran dry: output mid-scale and flag it, playback is now late
                m_axis_tdata <= 512'd0;
                underflow_q  <= 1'b1;
              end
            end else begin
              m_axis_tdata <= gear(sh, rate_sel);
              sh           <= advance(sh, rate_sel);
              cnt          <= cnt - 4'd1;
              if (cnt == 4'd1 && last_r)
                rstate <= R_DONE;
            end
          end
        end

        default: begin   // R_DONE
          if (m_axis_tready)
            m_axis_tdata <= 512'd0;
        end
      endcase
    end
  end

  // ================================================================ status
  assign status = {{(15-FIFO_AW){1'b0}}, fifo_rd_fill, 8'd0,
                   rd_active, trig_early_q, rd_err_q, underflow_q,
                   (rstate == R_DONE)  && rd_active,
                   (rstate == R_RUN)   && rd_active,
                   (rstate == R_READY) && rd_active,
                   busy_q};

endmodule
