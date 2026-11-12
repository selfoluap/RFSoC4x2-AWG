// -------------------------------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
//
// DACSeqStreamer - on-chip-memory player with sequence mode. Drop-in replacement for
// DACRAMstreamer (same BRAM and AXI4-Stream interfaces, same legacy `enable` / `numSamples`).
//
// The waveform memory holds n_blocks blocks of equal length, one after the other. Each block is
// played `repeats` times before the next one starts; switching happens on a 512-bit row boundary,
// so it is sample-exact and gap-free at the full DAC rate:
//
//   block 0 x repeats -> block 1 x repeats -> ... -> block n-1 x repeats -> end_mode
//
//   end_mode 0  stop: output returns to mid-scale
//            1  hold: keep repeating the last block until disarmed
//            2  loop: start again at block 0
//
// A Doppler staircase is n_blocks patterns that differ by one frequency step; a plain looped
// waveform is one block with end_mode = loop; "waveform A, then B" is two blocks with repeats = 1.
//
// Sequence mode (seq_ctrl[5] = 1): `arm` latches the configuration, then a software or external
// trigger starts playback a fixed number of clock cycles later.
// Legacy mode   (seq_ctrl[5] = 0): behaves like DACRAMstreamer: `enable` loops the first
// numSamples samples (rounded up to a row) immediately, no trigger.
//
// Inputs from AXI GPIOs are quasi-static levels: write the configuration before `arm`.
//
//   seq_ctrl    [0] arm   [1] sw_trigger (rising edge)   [2] ext_trig_en   [4:3] end_mode
//               [5] seq_mode
//   seq_lens    [15:0]  block length in 512-bit rows (32 samples each), 1 .. memory rows
//               [31:16] number of blocks (0 is treated as 1)
//   seq_repeats repeats per block (0 is treated as 1)
//   seq_status  [0] armed (waiting for a trigger)   [1] running   [2] done   [3] holding
//               [4] stall: the stream was back-pressured while playing (sticky; samples are then
//                   not guaranteed)   [5] sequence mode   [28:16] current block
//
// The output stream is always valid; it carries mid-scale (zero) samples when nothing plays, so
// the delay from trigger to the first sample is fixed. Back-pressure is not supported while
// playing: the chain behind this block consumes exactly one word per clock.
// -------------------------------------------------------------------------------------------------
`timescale 1ns / 1ps

module DACSeqStreamer #(
  parameter integer DWIDTH         = 512,
  parameter integer MEM_SIZE_BYTES = 524288,
  parameter integer READ_LATENCY   = 3        // clock cycles from address to data of the memory
) (
  (* X_INTERFACE_PARAMETER = "MASTER_TYPE BRAM_CTRL, READ_WRITE_MODE READ, MEM_SIZE 524288, MEM_WIDTH 512" *)

  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_A DIN" *)
  output wire [DWIDTH-1:0]   portA_cpu_wdata,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_A WE" *)
  output wire [DWIDTH/8-1:0] portA_we,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_A EN" *)
  output wire                portA_en,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_A DOUT" *)
  input  wire [DWIDTH-1:0]   portA_cpu_rdata,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_A ADDR" *)
  output reg  [31:0]         portAcpu_addr,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_A CLK" *)
  output wire                portA_clk,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_A RST" *)
  output wire                portA_rst,

  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 axis_clk CLK" *)
  (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF AXIS, ASSOCIATED_RESET axis_aresetn" *)
  input  wire                axis_clk,
  (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 axis_aresetn RST" *)
  input  wire                axis_aresetn,
  output reg  [DWIDTH-1:0]   axis_tdata,
  input  wire                axis_tready,
  output reg                 axis_tvalid,

  // legacy control (as DACRAMstreamer)
  input  wire [17:0]         numSamples,
  input  wire                enable,

  // sequence mode control
  input  wire [31:0]         seq_ctrl,
  input  wire [31:0]         seq_lens,
  input  wire [31:0]         seq_repeats,
  input  wire                ext_trigger,
  output wire [31:0]         seq_status
);

  localparam integer ROWS = MEM_SIZE_BYTES / (DWIDTH / 8);
  localparam integer RW   = $clog2(ROWS);
  localparam integer SHIFT = $clog2(DWIDTH / 8);

  localparam [1:0] S_IDLE = 2'd0, S_ARMED = 2'd1, S_RUN = 2'd2, S_DONE = 2'd3;
  localparam [1:0] E_STOP = 2'd0, E_HOLD = 2'd1, E_LOOP = 2'd2;

  assign portA_cpu_wdata = {DWIDTH{1'b0}};
  assign portA_we        = {(DWIDTH/8){1'b0}};
  assign portA_en        = 1'b1;
  assign portA_clk       = axis_clk;
  assign portA_rst       = ~axis_aresetn;

  // ------------------------------------------------------------ synchronisers
  (* ASYNC_REG = "TRUE" *) reg arm_1 = 1'b0, arm_2 = 1'b0;
  (* ASYNC_REG = "TRUE" *) reg en_1  = 1'b0, en_2  = 1'b0;
  (* ASYNC_REG = "TRUE" *) reg sm_1  = 1'b0, sm_2  = 1'b0;
  (* ASYNC_REG = "TRUE" *) reg swt_1 = 1'b0, swt_2 = 1'b0;
  (* ASYNC_REG = "TRUE" *) reg ext_1 = 1'b0, ext_2 = 1'b0;
  reg swt_3 = 1'b0, ext_3 = 1'b0;

  wire active = sm_2 ? arm_2 : en_2;
  wire trig   = (swt_2 && !swt_3) || (seq_ctrl[2] && ext_2 && !ext_3);

  // ------------------------------------------------------------ configuration, latched at arm
  // legacy: numSamples rounded up to rows; 0 or too long means the whole memory
  wire [18:0] legacy_rows_w = ({1'b0, numSamples} + 19'd31) >> 5;
  wire        legacy_full   = (numSamples == 18'd0) || (legacy_rows_w > ROWS);
  wire [RW:0] legacy_rows   = legacy_full ? ROWS[RW:0] : legacy_rows_w[RW:0];
  wire [RW:0] seq_rows      = (seq_lens[15:0] == 16'd0)  ? {{RW{1'b0}}, 1'b1} :
                              (seq_lens[15:0] > ROWS)    ? ROWS[RW:0] : seq_lens[RW:0];

  reg [RW:0]   len_l     = 1;          // block length in rows
  reg [RW-1:0] len_m1    = 0;
  reg [15:0]   nblk_m1   = 16'd0;
  reg [31:0]   reps_m1   = 32'd0;
  reg [1:0]    emode     = E_LOOP;
  reg          seq_l     = 1'b0;

  // ------------------------------------------------------------ sequencer
  reg [1:0]    state     = S_IDLE;
  reg [RW-1:0] row       = 0;          // row inside the block
  reg [RW-1:0] row_left  = 0;
  reg [RW-1:0] base      = 0;          // first row of the current block
  reg [31:0]   rep_left  = 32'd0;
  reg [15:0]   blk_left  = 16'd0;
  reg [15:0]   blk       = 16'd0;
  reg          holding   = 1'b0;
  reg          stall_q   = 1'b0;

  // valid flag travelling with the address through the memory's read latency
  reg [READ_LATENCY:0] vd = {(READ_LATENCY+1){1'b0}};

  always @(posedge axis_clk) begin
    arm_1 <= seq_ctrl[0];  arm_2 <= arm_1;
    en_1  <= enable;       en_2  <= en_1;
    sm_1  <= seq_ctrl[5];  sm_2  <= sm_1;
    swt_1 <= seq_ctrl[1];  swt_2 <= swt_1;  swt_3 <= swt_2;
    ext_1 <= ext_trigger;  ext_2 <= ext_1;  ext_3 <= ext_2;

    // address and data path: no reset needed, `vd` gates the output
    portAcpu_addr <= {{(32-RW-SHIFT){1'b0}}, base + row, {SHIFT{1'b0}}};
    vd            <= {vd[READ_LATENCY-1:0], (state == S_RUN) && active};
    axis_tdata    <= vd[READ_LATENCY] ? portA_cpu_rdata : {DWIDTH{1'b0}};
    axis_tvalid   <= axis_aresetn;

    if ((state == S_RUN || vd != 0) && !axis_tready)
      stall_q <= 1'b1;

    if (!axis_aresetn || !active) begin
      state   <= S_IDLE;
      holding <= 1'b0;
      if (!axis_aresetn) begin
        vd      <= {(READ_LATENCY+1){1'b0}};
        stall_q <= 1'b0;
      end
    end else begin
      case (state)
        S_IDLE: begin
          seq_l    <= sm_2;
          len_l    <= sm_2 ? seq_rows : legacy_rows;
          len_m1   <= (sm_2 ? seq_rows : legacy_rows) - 1'b1;
          nblk_m1  <= (!sm_2 || seq_lens[31:16] == 16'd0) ? 16'd0 : seq_lens[31:16] - 16'd1;
          reps_m1  <= (!sm_2 || seq_repeats == 32'd0)     ? 32'd0 : seq_repeats - 32'd1;
          emode    <= sm_2 ? seq_ctrl[4:3] : E_LOOP;
          stall_q  <= 1'b0;
          state    <= S_ARMED;
        end

        S_ARMED: begin
          row      <= 0;
          row_left <= len_m1;
          base     <= 0;
          rep_left <= reps_m1;
          blk_left <= nblk_m1;
          blk      <= 16'd0;
          holding  <= 1'b0;
          if (!seq_l || trig)
            state <= S_RUN;
        end

        S_RUN: begin
          if (row_left != 0) begin
            row      <= row + 1'b1;
            row_left <= row_left - 1'b1;
          end else begin
            // last row of the block: decide what follows it
            row      <= 0;
            row_left <= len_m1;
            if (holding) begin
              // keep repeating the last block
            end else if (rep_left != 32'd0) begin
              rep_left <= rep_left - 32'd1;
            end else begin
              rep_left <= reps_m1;
              if (blk_left != 16'd0) begin
                base     <= base + len_l[RW-1:0];
                blk_left <= blk_left - 16'd1;
                blk      <= blk + 16'd1;
              end else begin
                case (emode)
                  E_STOP:  state   <= S_DONE;
                  E_HOLD:  holding <= 1'b1;
                  default: begin
                    base     <= 0;
                    blk_left <= nblk_m1;
                    blk      <= 16'd0;
                  end
                endcase
              end
            end
          end
        end

        default: ;   // S_DONE: wait for disarm
      endcase
    end
  end

  assign seq_status = {3'd0, blk[12:0], 10'd0, seq_l, stall_q, holding,
                       (state == S_DONE), (state == S_RUN), (state == S_ARMED)};

endmodule
