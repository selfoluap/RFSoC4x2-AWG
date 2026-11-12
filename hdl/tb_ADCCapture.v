// -------------------------------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
//
// Self-checking testbench for ADCCapture.
//
//   iverilog -g2012 -o tb_adc hdl/tb_ADCCapture.v fpga/src/rtl/ADCCapture.v \
//            fpga/src/rtl/AnalogTrigger.v && vvp tb_adc
//
// Stream s carries the word {s, n} for its n-th valid word, with random gaps. Every word written
// to the memory port must come from the selected stream and continue the count without a skip.
// The same scenarios run on the Python model in tests/test_adc_capture_model.py.
// -------------------------------------------------------------------------------------------------
`timescale 1ns / 1ps

module tb_ADCCapture;

  reg clk = 1'b0;
  always #1.6275 clk = ~clk;

  reg          aresetn = 1'b0;
  reg  [31:0]  ctrl = 32'd0, delay = 32'd0, nrows = 32'd0;
  reg          ext_trigger = 1'b0;
  reg  [31:0]  dac0_status = 32'd0, dac2_status = 32'd0;
  wire [31:0]  status;

  reg  [255:0] td [0:3];
  reg          tv [0:3];
  reg  [31:0]  n  [0:3];
  integer s;

  initial for (s = 0; s < 4; s = s + 1) begin td[s] = 256'd0; tv[s] = 1'b0; n[s] = 32'd0; end

  always @(posedge clk)
    for (s = 0; s < 4; s = s + 1) begin
      if ($urandom % 4 != 0) begin
        tv[s] <= 1'b1;
        td[s] <= {216'd0, s[7:0], n[s]};
        n[s]  <= n[s] + 32'd1;
      end else begin
        tv[s] <= 1'b0;
        td[s] <= {256{1'b1}};
      end
    end

  wire [255:0] wdata;
  wire [31:0]  we;
  wire [31:0]  addr;

  ADCCapture uut (
    .mem_wdata (wdata), .mem_we (we), .mem_en (), .mem_rdata (256'd0), .mem_addr (addr),
    .mem_clk (), .mem_rst (),
    .axis_clk (clk), .axis_aresetn (aresetn),
    .s0_axis_tdata (td[0]), .s0_axis_tvalid (tv[0]), .s0_axis_tready (),
    .s1_axis_tdata (td[1]), .s1_axis_tvalid (tv[1]), .s1_axis_tready (),
    .s2_axis_tdata (td[2]), .s2_axis_tvalid (tv[2]), .s2_axis_tready (),
    .s3_axis_tdata (td[3]), .s3_axis_tvalid (tv[3]), .s3_axis_tready (),
    .ctrl (ctrl), .delay (delay), .nrows (nrows), .ext_trigger (ext_trigger),
    .dac0_status (dac0_status), .dac2_status (dac2_status), .status (status),
    .trig_ctrl (32'd0), .trig_thr (32'd0), .trig_release (32'd0), .trig_status (), .analog_trig ()
  );

  // ------------------------------------------------------------ memory port checker
  integer errors = 0;
  integer written = 0;
  reg [31:0] first_n, prev_n;
  reg [7:0]  want_adc;

  always @(posedge clk)
    if (we != 32'd0) begin
      if (we !== 32'hFFFFFFFF || addr !== written * 32) begin
        errors = errors + 1; $display("ERROR: bad write strobe or address %h at %0t", addr, $time);
      end
      if (wdata[39:32] !== want_adc) begin
        errors = errors + 1; $display("ERROR: word from stream %0d, expected %0d", wdata[39:32], want_adc);
      end
      if (written == 0) first_n = wdata[31:0];
      else if (wdata[31:0] !== prev_n + 1) begin
        errors = errors + 1; $display("ERROR: skipped from %0d to %0d", prev_n, wdata[31:0]);
      end
      prev_n = wdata[31:0];
      written = written + 1;
    end

  task run;
    input [1:0]   adc;
    input integer rows;
    input integer dly;
    input [2:0]   source;      // 0 software, 1 external, 2 DAC0 start
    reg [31:0]    n_at_trigger;
    begin
      ctrl = 32'd0;
      repeat (6) @(posedge clk);
      written = 0; want_adc = {6'd0, adc};
      nrows = rows; delay = dly;
      ctrl = (adc << 5) | (source << 2);
      repeat (3) @(posedge clk);
      ctrl[0] = 1'b1;
      wait (status[0]);
      repeat (20) @(posedge clk);
      if (written != 0) begin errors = errors + 1; $display("ERROR: write before trigger"); end
      n_at_trigger = n[adc];
      case (source)
        3'd0: begin ctrl[1] = 1'b1; repeat (4) @(posedge clk); ctrl[1] = 1'b0; end
        3'd1: begin #4.3 ext_trigger = 1'b1; end
        default: dac0_status = 32'd2;
      endcase
      wait (status[2]);
      repeat (4) @(posedge clk);
      ext_trigger = 1'b0; dac0_status = 32'd0;
      if (written != rows || status[30:16] != rows) begin
        errors = errors + 1; $display("ERROR: %0d words written, status says %0d, expected %0d", written, status[30:16], rows);
      end
      if (first_n - n_at_trigger < dly || first_n - n_at_trigger > dly + 12) begin
        errors = errors + 1; $display("ERROR: first word %0d words after the trigger, delay was %0d", first_n - n_at_trigger, dly);
      end
      $display("  adc %0d, %0d words, delay %0d, source %0d: ok so far (%0d errors)", adc, rows, dly, source, errors);
    end
  endtask

  initial begin
    repeat (10) @(posedge clk);
    aresetn = 1'b1;
    repeat (10) @(posedge clk);
    run(2'd0, 1, 0, 3'd0);
    run(2'd1, 100, 0, 3'd0);
    run(2'd2, 300, 500, 3'd0);
    run(2'd3, 64, 0, 3'd1);
    run(2'd0, 64, 7, 3'd2);
    run(2'd1, 16384, 0, 3'd0);
    if (errors == 0) $display("PASS: capture tests, no errors");
    else $display("FAIL: %0d errors", errors);
    $finish;
  end

  initial begin #2000000; $display("FAIL: global timeout"); $finish; end

endmodule
