// -------------------------------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
//
// Self-checking testbench for AnalogTrigger.
//
//   iverilog -g2012 -o tb_atrig hdl/tb_AnalogTrigger.v fpga/src/rtl/AnalogTrigger.v && vvp tb_atrig
//
// Input 2 carries bursts of a square-ish carrier; the other inputs are held at full scale and
// must be ignored. Checks: one event per burst, `active` rises on the third clock edge after the
// first word over threshold, drops after the release time, peak reading, disabled = silent.
// The same behaviour is tested on the Python model in tests/test_analog_trigger_model.py.
// -------------------------------------------------------------------------------------------------
`timescale 1ns / 1ps

module tb_AnalogTrigger;

  reg clk = 1'b0;
  always #1.6275 clk = ~clk;

  reg          aresetn = 1'b0;
  reg  [255:0] d = 256'd0;
  reg  [31:0]  ctrl = 32'd0, thr = 32'd0, release_words = 32'd0;
  wire         active;
  wire [31:0]  status;

  AnalogTrigger uut (
    .clk (clk), .aresetn (aresetn),
    .d0 ({16{16'h7FFF}}), .d1 ({16{16'h7FFF}}), .d2 (d), .d3 ({16{16'h7FFF}}),
    .v0 (1'b1), .v1 (1'b1), .v2 (1'b1), .v3 (1'b1),
    .ctrl (ctrl), .thr (thr), .release_words (release_words),
    .active (active), .status (status)
  );

  integer errors = 0;
  integer rises = 0;
  integer cycle = 0;
  integer rise_cycle = 0;
  reg     active_d = 1'b0;

  always @(posedge clk) begin
    cycle <= cycle + 1;
    active_d <= active;
    if (active && !active_d) begin
      rises = rises + 1;
      rise_cycle = cycle;
    end
  end

  // one word with all 16 samples at +/- amplitude, alternating sign every 4 samples
  function [255:0] carrier;
    input integer amplitude;
    integer k;
    reg signed [15:0] v;
    begin
      for (k = 0; k < 16; k = k + 1) begin
        v = ((k / 4) % 2) ? -amplitude : amplitude;
        carrier[16*k +: 16] = v;
      end
    end
  endfunction

  task burst;                 // `words` words of carrier, then `gap` words of silence
    input integer amplitude;
    input integer words;
    input integer gap;
    begin
      repeat (words) begin d <= carrier(amplitude); @(posedge clk); end
      repeat (gap)   begin d <= 256'd0;             @(posedge clk); end
    end
  endtask

  integer start_cycle;

  initial begin
    repeat (8) @(posedge clk);
    aresetn = 1'b1;

    // disabled: nothing fires, but the peak is measured
    ctrl = (2 << 1);          thr = (32'd1500 << 16) | 32'd3000;   release_words = 32'd4;
    repeat (4) @(posedge clk);
    burst(8000, 20, 20);
    if (rises != 0 || status[31:16] != 0) begin errors = errors + 1; $display("ERROR: fired while disabled"); end
    if (status[15:1] != 15'd8000) begin errors = errors + 1; $display("ERROR: peak %0d, expected 8000", status[15:1]); end

    // enabled, envelope mode: one event per burst
    ctrl = (2 << 1) | 1;
    repeat (6) @(posedge clk);
    start_cycle = cycle + 1;                 // cycle index at which the first burst word is sampled
    burst(8000, 30, 40);
    if (rises != 1) begin errors = errors + 1; $display("ERROR: %0d rising edges, expected 1", rises); end
    // 3 clock edges in the design; the bench's own cycle counting is good to +/- 1
    if (rise_cycle - start_cycle < 2 || rise_cycle - start_cycle > 4) begin
      errors = errors + 1; $display("ERROR: active rose %0d cycles after the burst, expected 3", rise_cycle - start_cycle);
    end
    if (active) begin errors = errors + 1; $display("ERROR: still active after the gap"); end
    burst(8000, 30, 40);
    burst(2000, 30, 40);                     // below the on threshold: no event
    burst(8000, 30, 40);
    if (rises != 3 || status[31:16] != 16'd3) begin
      errors = errors + 1; $display("ERROR: %0d rises, %0d events, expected 3", rises, status[31:16]);
    end

    // clear
    ctrl[5] = 1'b1; repeat (6) @(posedge clk); ctrl[5] = 1'b0; repeat (6) @(posedge clk);
    if (status[31:1] != 0) begin errors = errors + 1; $display("ERROR: clear did not reset peak and events"); end

    // rising-level mode with a negative-going input: no event; with a positive one: event
    ctrl = (1 << 3) | (2 << 1) | 1;          thr = (32'd3000 << 16) | 32'd3000;
    repeat (6) @(posedge clk);
    rises = 0;
    d <= {16{16'hE000}}; repeat (10) @(posedge clk);       // -8192
    d <= 256'd0;         repeat (10) @(posedge clk);
    if (rises != 0) begin errors = errors + 1; $display("ERROR: rising mode fired on a negative level"); end
    d <= {16{16'h2000}}; repeat (10) @(posedge clk);       // +8192
    d <= 256'd0;         repeat (20) @(posedge clk);
    if (rises != 1) begin errors = errors + 1; $display("ERROR: rising mode did not fire once"); end

    if (errors == 0) $display("PASS: analog trigger tests, no errors");
    else $display("FAIL: %0d errors", errors);
    $finish;
  end

  initial begin #200000; $display("FAIL: global timeout"); $finish; end

endmodule
