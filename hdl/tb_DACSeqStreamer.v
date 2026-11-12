// -------------------------------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
//
// Self-checking testbench for DACSeqStreamer.
//
//   iverilog -g2012 -o tb_seq hdl/tb_DACSeqStreamer.v fpga/src/rtl/DACSeqStreamer.v && vvp tb_seq
//
// The waveform memory is modelled as "row r holds the value r + 1" with the same read latency as
// the block memory. Every non-zero output word is compared with the row the sequence must play
// at that position, and the output must be gap-free from first to last word.
// The same scenarios run on the Python model in tests/test_seq_streamer_model.py.
// -------------------------------------------------------------------------------------------------
`timescale 1ns / 1ps

module tb_DACSeqStreamer;

  localparam integer LAT = 3;

  reg clk = 1'b0;
  always #1.6275 clk = ~clk;              // 307.2 MHz

  reg          aresetn = 1'b0;
  reg  [31:0]  seq_ctrl = 32'd0;
  reg  [31:0]  seq_lens = 32'd0;
  reg  [31:0]  seq_repeats = 32'd0;
  reg          ext_trigger = 1'b0;
  reg          enable = 1'b0;
  reg  [17:0]  numSamples = 18'd0;
  wire [31:0]  status;
  wire [31:0]  addr;
  wire [511:0] tdata;
  wire         tvalid;

  // memory model: rdata(t) = content of the row addressed LAT cycles earlier
  reg [511:0] pipe [0:LAT-1];
  integer p;
  always @(posedge clk) begin
    pipe[0] <= {480'd0, addr >> 6} + 512'd1;
    for (p = 1; p < LAT; p = p + 1)
      pipe[p] <= pipe[p-1];
  end

  DACSeqStreamer #( .MEM_SIZE_BYTES(524288), .READ_LATENCY(LAT) ) uut (
    .portA_cpu_wdata (),
    .portA_we        (),
    .portA_en        (),
    .portA_cpu_rdata (pipe[LAT-1]),
    .portAcpu_addr   (addr),
    .portA_clk       (),
    .portA_rst       (),
    .axis_clk        (clk),
    .axis_aresetn    (aresetn),
    .axis_tdata      (tdata),
    .axis_tready     (1'b1),
    .axis_tvalid     (tvalid),
    .numSamples      (numSamples),
    .enable          (enable),
    .seq_ctrl        (seq_ctrl),
    .seq_lens        (seq_lens),
    .seq_repeats     (seq_repeats),
    .ext_trigger     (ext_trigger),
    .seq_status      (status)
  );

  // ------------------------------------------------------------ checker
  integer errors = 0;
  integer count = 0;                      // non-zero words seen in the current test
  integer gap = 0;                        // set when a zero word follows data
  integer gap_errors = 0;
  integer exp_rows, exp_blocks, exp_reps; // sequence under test
  integer exp_hold = 0;                   // 1: after one pass the last block repeats (hold mode)
  integer per_block, blk_i, row_i, want;

  always @(posedge clk) begin
    if (tvalid && tdata != 512'd0) begin
      if (gap) begin
        gap_errors = gap_errors + 1;
        if (gap_errors < 5) $display("ERROR: gap inside the sequence at %0t", $time);
      end
      per_block = exp_rows * exp_reps;
      if (exp_hold && count >= per_block * exp_blocks) begin
        blk_i = exp_blocks - 1;
        row_i = (count - per_block * exp_blocks) % exp_rows;
      end else begin
        blk_i = (count / per_block) % exp_blocks;
        row_i = (count % per_block) % exp_rows;
      end
      want  = blk_i * exp_rows + row_i + 1;
      if (tdata !== {480'd0, want[31:0]}) begin
        errors = errors + 1;
        if (errors < 10)
          $display("ERROR: word %0d is row %0d, expected row %0d at %0t", count, tdata[31:0] - 1, want - 1, $time);
      end
      count = count + 1;
    end else if (count != 0)
      gap = 1;
  end

  task expect_sequence;
    input integer rows;
    input integer blocks;
    input integer reps;
    begin
      exp_rows = rows; exp_blocks = blocks; exp_reps = reps;
      count = 0; gap = 0; exp_hold = 0;
    end
  endtask

  task arm;
    input integer rows;
    input integer blocks;
    input integer reps;
    input [1:0]   end_mode;
    input         ext_en;
    begin
      seq_ctrl = 32'd0 | (1 << 5);
      repeat (12) @(posedge clk);         // disarmed: pipeline drains
      expect_sequence(rows, blocks, reps);
      seq_lens    = (blocks << 16) | rows;
      seq_repeats = reps;
      seq_ctrl    = (1 << 5) | (end_mode << 3) | (ext_en << 2);
      repeat (3) @(posedge clk);
      seq_ctrl[0] = 1'b1;
      wait (status[0]);
      repeat (10) @(posedge clk);
      if (count != 0) begin
        errors = errors + 1;
        $display("ERROR: output before the trigger at %0t", $time);
      end
    end
  endtask

  task sw_trigger;
    begin
      @(posedge clk); seq_ctrl[1] = 1'b1;
      repeat (4) @(posedge clk); seq_ctrl[1] = 1'b0;
    end
  endtask

  task run_stop_test;
    input integer rows;
    input integer blocks;
    input integer reps;
    begin
      arm(rows, blocks, reps, 2'd0, 1'b0);
      sw_trigger;
      wait (status[2]);
      repeat (12) @(posedge clk);
      if (count != rows * blocks * reps) begin
        errors = errors + 1;
        $display("ERROR: %0d rows played, expected %0d", count, rows * blocks * reps);
      end
      if (status[4]) begin errors = errors + 1; $display("ERROR: stall flag set"); end
      $display("  stop: %0d rows x %0d blocks x %0d repeats: %0d rows checked", rows, blocks, reps, count);
    end
  endtask

  initial begin
    repeat (10) @(posedge clk);
    aresetn = 1'b1;
    repeat (10) @(posedge clk);

    $display("end_mode = stop");
    run_stop_test(1, 1, 1);
    run_stop_test(1, 5, 3);
    run_stop_test(61, 3, 4);
    run_stop_test(7, 9, 1);
    run_stop_test(4096, 2, 1);
    run_stop_test(61, 120, 168);          // the Doppler staircase of the design note: 4.003 ms

    $display("end_mode = loop");
    arm(5, 3, 2, 2'd2, 1'b0);
    sw_trigger;
    repeat (500) @(posedge clk);
    if (!status[1] || count < 400) begin errors = errors + 1; $display("ERROR: loop not running"); end

    $display("end_mode = hold, external trigger");
    arm(4, 3, 2, 2'd1, 1'b1);
    exp_hold = 1;
    #5.1 ext_trigger = 1'b1;              // deliberately asynchronous
    wait (status[3]);
    repeat (100) @(posedge clk);          // the checker now expects the last block to repeat
    if (count < 24 + 80) begin errors = errors + 1; $display("ERROR: hold sequence too short"); end
    ext_trigger = 1'b0;
    repeat (4) @(posedge clk);
    if (!status[1] || status[18:16] != 3'd2) begin
      errors = errors + 1; $display("ERROR: not holding on block 2");
    end

    $display("legacy mode (as DACRAMstreamer)");
    seq_ctrl = 32'd0;
    repeat (12) @(posedge clk);
    expect_sequence(32, 1, 1);
    numSamples = 18'd1000;                // 32 rows
    enable = 1'b1;
    repeat (200) @(posedge clk);
    if (count < 150) begin errors = errors + 1; $display("ERROR: legacy loop not running"); end
    enable = 1'b0;
    repeat (12) @(posedge clk);

    if (errors == 0 && gap_errors == 0) $display("PASS: sequencer tests, no errors");
    else $display("FAIL: %0d data/status errors, %0d gaps", errors, gap_errors);
    $finish;
  end

  initial begin
    #40000000;
    $display("FAIL: global timeout");
    $finish;
  end

endmodule
