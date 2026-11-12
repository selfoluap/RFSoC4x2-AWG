// -------------------------------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
//
// ADCCapture - triggered recording of one ADC input into on-chip memory.
//
//   4 ADC streams (256 bit = 16 samples per word, 307.2 MHz) -> input select -> capture memory
//
// After a trigger the block skips `delay` words, then writes `nrows` consecutive words to the
// memory, which the processor reads back through the other memory port. The ADCs run at
// 4915.2 MS/s, exactly half the DAC rate and from the same reference, so a capture is coherent
// with playback.
//
// Inputs from AXI GPIOs are quasi-static levels: write the configuration before `arm`.
//
//   ctrl    [0] arm   [1] sw_trigger (rising edge, always accepted)
//           [4:2] extra trigger source: 0 none, 1 external pin, 2 DAC0 playback start,
//                 3 DAC2 playback start, 4 analog trigger (AnalogTrigger, see below)
//           [6:5] ADC input select (stream 0..3)
//   delay   words (16 samples, 3.26 ns) to skip between trigger and first recorded word
//   nrows   words to record (0 or more than the memory holds = whole memory)
//   status  [0] armed (waiting for a trigger)   [1] busy (delay or recording)   [2] done
//           [30:16] words recorded
//
// The ADC streams are never back-pressured (tready is constant 1).
//
// The block also holds the analog trigger detector (AnalogTrigger.v), which watches one of the
// same four streams. Its `analog_trig` output is a trigger source here and for the DAC players.
// -------------------------------------------------------------------------------------------------
`timescale 1ns / 1ps

module ADCCapture #(
  parameter integer DWIDTH         = 256,
  parameter integer MEM_SIZE_BYTES = 524288
) (
  (* X_INTERFACE_PARAMETER = "MASTER_TYPE BRAM_CTRL, MEM_SIZE 524288, MEM_WIDTH 256" *)
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_A DIN" *)
  output reg  [DWIDTH-1:0]   mem_wdata,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_A WE" *)
  output reg  [DWIDTH/8-1:0] mem_we,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_A EN" *)
  output wire                mem_en,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_A DOUT" *)
  input  wire [DWIDTH-1:0]   mem_rdata,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_A ADDR" *)
  output reg  [31:0]         mem_addr,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_A CLK" *)
  output wire                mem_clk,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_A RST" *)
  output wire                mem_rst,

  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 axis_clk CLK" *)
  (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s0_axis:s1_axis:s2_axis:s3_axis, ASSOCIATED_RESET axis_aresetn" *)
  input  wire                axis_clk,
  (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 axis_aresetn RST" *)
  input  wire                axis_aresetn,

  input  wire [DWIDTH-1:0]   s0_axis_tdata,
  input  wire                s0_axis_tvalid,
  output wire                s0_axis_tready,
  input  wire [DWIDTH-1:0]   s1_axis_tdata,
  input  wire                s1_axis_tvalid,
  output wire                s1_axis_tready,
  input  wire [DWIDTH-1:0]   s2_axis_tdata,
  input  wire                s2_axis_tvalid,
  output wire                s2_axis_tready,
  input  wire [DWIDTH-1:0]   s3_axis_tdata,
  input  wire                s3_axis_tvalid,
  output wire                s3_axis_tready,

  input  wire [31:0]         ctrl,
  input  wire [31:0]         delay,
  input  wire [31:0]         nrows,
  input  wire                ext_trigger,
  input  wire [31:0]         dac0_status,     // seq_status of the DAC0 player (bit 1 = running)
  input  wire [31:0]         dac2_status,
  output wire [31:0]         status,

  // analog trigger (see AnalogTrigger.v for the register layout)
  input  wire [31:0]         trig_ctrl,
  input  wire [31:0]         trig_thr,
  input  wire [31:0]         trig_release,
  output wire [31:0]         trig_status,
  output wire                analog_trig
);

  localparam integer ROWS  = MEM_SIZE_BYTES / (DWIDTH / 8);
  localparam integer RW    = $clog2(ROWS);
  localparam integer SHIFT = $clog2(DWIDTH / 8);

  localparam [2:0] C_IDLE = 3'd0, C_ARMED = 3'd1, C_DELAY = 3'd2, C_REC = 3'd3, C_DONE = 3'd4;

  assign mem_en  = 1'b1;
  assign mem_clk = axis_clk;
  assign mem_rst = ~axis_aresetn;

  assign s0_axis_tready = 1'b1;
  assign s1_axis_tready = 1'b1;
  assign s2_axis_tready = 1'b1;
  assign s3_axis_tready = 1'b1;

  // ------------------------------------------------------------ input select (registered)
  reg [DWIDTH-1:0] d_r = {DWIDTH{1'b0}};
  reg              v_r = 1'b0;

  always @(posedge axis_clk) begin
    case (ctrl[6:5])
      2'd0:    begin d_r <= s0_axis_tdata; v_r <= s0_axis_tvalid; end
      2'd1:    begin d_r <= s1_axis_tdata; v_r <= s1_axis_tvalid; end
      2'd2:    begin d_r <= s2_axis_tdata; v_r <= s2_axis_tvalid; end
      default: begin d_r <= s3_axis_tdata; v_r <= s3_axis_tvalid; end
    endcase
  end

  // ------------------------------------------------------------ triggers
  (* ASYNC_REG = "TRUE" *) reg arm_1 = 1'b0, arm_2 = 1'b0;
  (* ASYNC_REG = "TRUE" *) reg swt_1 = 1'b0, swt_2 = 1'b0;
  (* ASYNC_REG = "TRUE" *) reg ext_1 = 1'b0, ext_2 = 1'b0;
  reg swt_3 = 1'b0, ext_3 = 1'b0, d0_1 = 1'b0, d2_1 = 1'b0, at_1 = 1'b0;

  AnalogTrigger u_atrig (
    .clk           (axis_clk),
    .aresetn       (axis_aresetn),
    .d0 (s0_axis_tdata), .d1 (s1_axis_tdata), .d2 (s2_axis_tdata), .d3 (s3_axis_tdata),
    .v0 (s0_axis_tvalid), .v1 (s1_axis_tvalid), .v2 (s2_axis_tvalid), .v3 (s3_axis_tvalid),
    .ctrl          (trig_ctrl),
    .thr           (trig_thr),
    .release_words (trig_release),
    .active        (analog_trig),
    .status        (trig_status)
  );

  wire trig = (swt_2 && !swt_3)
           || (ctrl[4:2] == 3'd1 && ext_2 && !ext_3)
           || (ctrl[4:2] == 3'd2 && dac0_status[1] && !d0_1)
           || (ctrl[4:2] == 3'd3 && dac2_status[1] && !d2_1)
           || (ctrl[4:2] == 3'd4 && analog_trig && !at_1);

  // ------------------------------------------------------------ recorder
  reg [2:0]    state      = C_IDLE;
  reg [31:0]   delay_left = 32'd0;
  reg [RW:0]   rows_left  = 0;
  reg [RW:0]   wr_row     = 0;

  wire [RW:0] rows_cfg = (nrows == 32'd0 || nrows > ROWS) ? ROWS[RW:0] : nrows[RW:0];

  always @(posedge axis_clk) begin
    arm_1 <= ctrl[0];      arm_2 <= arm_1;
    swt_1 <= ctrl[1];      swt_2 <= swt_1;  swt_3 <= swt_2;
    ext_1 <= ext_trigger;  ext_2 <= ext_1;  ext_3 <= ext_2;
    d0_1  <= dac0_status[1];
    d2_1  <= dac2_status[1];
    at_1  <= analog_trig;

    mem_we <= {(DWIDTH/8){1'b0}};

    if (!axis_aresetn || !arm_2) begin
      state <= C_IDLE;
    end else begin
      case (state)
        C_IDLE: begin
          delay_left <= delay;
          rows_left  <= rows_cfg;
          wr_row     <= 0;
          state      <= C_ARMED;
        end

        C_ARMED: begin
          if (trig)
            state <= (delay_left == 32'd0) ? C_REC : C_DELAY;
        end

        C_DELAY: begin
          if (v_r) begin
            delay_left <= delay_left - 32'd1;
            if (delay_left == 32'd1)
              state <= C_REC;
          end
        end

        C_REC: begin
          if (v_r) begin
            mem_wdata <= d_r;
            mem_addr  <= {{(32-RW-1-SHIFT){1'b0}}, wr_row, {SHIFT{1'b0}}};
            mem_we    <= {(DWIDTH/8){1'b1}};
            wr_row    <= wr_row + 1'b1;
            rows_left <= rows_left - 1'b1;
            if (rows_left == 1)
              state <= C_DONE;
          end
        end

        default: ;   // C_DONE: wait for disarm
      endcase
    end
  end

  assign status = {{(15-RW){1'b0}}, wr_row, 13'd0,
                   (state == C_DONE), (state == C_DELAY || state == C_REC), (state == C_ARMED)};

  // mem_rdata is unused: this port only writes
  wire unused = &{1'b0, mem_rdata};

endmodule
