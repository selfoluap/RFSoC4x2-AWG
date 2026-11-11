// -------------------------------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
//
// Self-checking testbench for DDR4Streamer + awg_async_fifo.
//
//   iverilog -g2012 -o tb_ddr4 hdl/tb_DDR4Streamer.v fpga/src/rtl/DDR4Streamer.v \
//            fpga/src/rtl/awg_async_fifo.v && vvp tb_ddr4
//
// The DDR4 controller is replaced by a small AXI read slave whose memory content is a function
// of the address, with a first-beat latency and periodic "refresh" stalls. Every sample that
// leaves the streamer is compared against the content of the address it must have come from.
// The same scenarios are covered by tests/test_ddr4_player_model.py on the Python model.
// -------------------------------------------------------------------------------------------------
`timescale 1ns / 1ps

module tb_DDR4Streamer;

  localparam integer ADDR_WIDTH = 33;
  localparam integer FIFO_AW    = 10;
  localparam integer LATENCY    = 24;     // AR accepted -> first R beat, in ui_clk cycles
  localparam integer REF_PERIOD = 2340;   // refresh every 7.8 us
  localparam integer REF_STALL  = 75;     // ... for 0.25 us (3 % dead time; 14-bit full rate tolerates 10 %)

  reg ui_clk   = 1'b0;
  reg axis_clk = 1'b0;
  always #1.6665 ui_clk   = ~ui_clk;      // 300 MHz
  always #1.6275 axis_clk = ~axis_clk;    // 307.2 MHz

  reg ui_aresetn   = 1'b0;
  reg axis_aresetn = 1'b0;

  // ------------------------------------------------------------ DUT connections
  wire [ADDR_WIDTH-1:0] araddr;
  wire [7:0]            arlen;
  wire [2:0]            arsize;
  wire [1:0]            arburst;
  wire                  arvalid;
  reg                   arready = 1'b0;
  reg  [511:0]          rdata   = 512'd0;
  reg                   rlast   = 1'b0;
  reg                   rvalid  = 1'b0;
  wire                  rready;

  wire [511:0]          tdata;
  wire                  tvalid;
  reg                   tready = 1'b1;

  reg                   enable       = 1'b0;
  reg                   sw_trigger   = 1'b0;
  reg                   ext_trigger  = 1'b0;
  reg                   ext_trig_en  = 1'b0;
  reg                   loop_forever = 1'b0;
  reg  [2:0]            mode         = 3'd0;
  reg  [23:0]           loop_count   = 24'd1;
  reg  [ADDR_WIDTH-13:0] base_page   = 0;
  reg  [27:0]           num_words    = 28'd0;
  wire [31:0]           status;

  wire st_busy      = status[0];
  wire st_ready     = status[1];
  wire st_running   = status[2];
  wire st_done      = status[3];
  wire st_underflow = status[4];
  wire st_rd_err    = status[5];

  DDR4Streamer #( .ADDR_WIDTH(ADDR_WIDTH), .FIFO_AW(FIFO_AW) ) uut (
    .m_axi_aclk    (ui_clk),
    .m_axi_aresetn (ui_aresetn),
    .m_axi_araddr  (araddr),
    .m_axi_arlen   (arlen),
    .m_axi_arsize  (arsize),
    .m_axi_arburst (arburst),
    .m_axi_arlock  (),
    .m_axi_arcache (),
    .m_axi_arprot  (),
    .m_axi_arqos   (),
    .m_axi_arvalid (arvalid),
    .m_axi_arready (arready),
    .m_axi_rdata   (rdata),
    .m_axi_rresp   (2'b00),
    .m_axi_rlast   (rlast),
    .m_axi_rvalid  (rvalid),
    .m_axi_rready  (rready),
    .axis_clk      (axis_clk),
    .axis_aresetn  (axis_aresetn),
    .m_axis_tdata  (tdata),
    .m_axis_tvalid (tvalid),
    .m_axis_tready (tready),
    .enable        (enable),
    .sw_trigger    (sw_trigger),
    .ext_trigger   (ext_trigger),
    .ext_trig_en   (ext_trig_en),
    .loop_forever  (loop_forever),
    .mode          (mode),
    .loop_count    (loop_count),
    .base_page     (base_page),
    .num_words     (num_words),
    .status        (status)
  );

  // ------------------------------------------------------------ memory content
  function [15:0] pattern;                 // sample value at an absolute sample index
    input [63:0] index;
    reg [63:0] v;
    begin
      v = index * 64'd40503 + 64'd12345;
      pattern = v[15:0];
    end
  endfunction

  function [511:0] mem_word;               // 64-byte word at an absolute word index
    input [63:0] word_index;
    integer i;
    begin
      for (i = 0; i < 32; i = i + 1)
        mem_word[16*i +: 16] = pattern(word_index * 64'd32 + i);
    end
  endfunction

  // ------------------------------------------------------------ DDR4 controller model
  integer        ui_cycle = 0;
  reg [63:0]     q_word [0:3];
  integer        q_len  [0:3];
  integer        q_time [0:3];
  integer        q_head = 0;
  integer        q_tail = 0;
  integer        q_count = 0;
  integer        axi_errors = 0;
  wire           stalled_next = ((ui_cycle + 1) % REF_PERIOD) < REF_STALL;

  always @(posedge ui_clk) begin : mig_model
    integer count_next;
    integer head_next;
    reg     rvalid_keep;

    count_next  = q_count;
    head_next   = q_head;
    rvalid_keep = rvalid;

    if (arvalid && arready) begin
      if (araddr[5:0] != 6'd0 || arsize != 3'b110 || arburst != 2'b01 ||
          ({20'd0, araddr[11:0]} + (arlen + 1) * 64) > 4096) begin
        $display("ERROR: illegal read burst addr=%h len=%0d at %0t", araddr, arlen, $time);
        axi_errors = axi_errors + 1;
      end
      q_word[q_tail] = araddr >> 6;
      q_len[q_tail]  = arlen + 1;
      q_time[q_tail] = ui_cycle + LATENCY;
      q_tail         = (q_tail + 1) % 4;
      count_next     = count_next + 1;
    end

    if (rvalid && rready) begin
      q_word[head_next] = q_word[head_next] + 1;
      q_len[head_next]  = q_len[head_next] - 1;
      if (q_len[head_next] == 0) begin
        head_next  = (head_next + 1) % 4;
        count_next = count_next - 1;
      end
      rvalid_keep = 1'b0;
    end

    q_head   = head_next;
    q_count  = count_next;
    ui_cycle = ui_cycle + 1;

    arready <= (count_next < 4) && !stalled_next && ui_aresetn;

    if (!rvalid_keep) begin
      if (count_next > 0 && q_time[head_next] <= ui_cycle && !stalled_next) begin
        rvalid <= 1'b1;
        rdata  <= mem_word(q_word[head_next]);
        rlast  <= (q_len[head_next] == 1);
      end else begin
        rvalid <= 1'b0;
      end
    end
  end

  // ------------------------------------------------------------ output checker
  reg            is_data = 1'b0;           // the beat currently on tdata is waveform data
  integer        errors = 0;
  integer        checked = 0;              // samples compared in the current test
  reg [63:0]     exp_first = 0;            // absolute index of the first sample of the waveform
  reg [63:0]     exp_len   = 1;            // waveform length in samples
  integer        per_half;                 // samples per 256-bit half for the current rate_sel
  integer        k;
  reg [15:0]     got;
  reg [15:0]     want;

  integer        sample_bits;              // stored bits per sample for the current mode

  always @(*) begin
    case (mode)
      3'd0:    per_half = 8;
      3'd1:    per_half = 4;
      3'd2:    per_half = 2;
      3'd3:    per_half = 1;
      default: per_half = 16;              // full rate
    endcase
    case (mode)
      3'd4:    sample_bits = 14;
      3'd5:    sample_bits = 12;
      3'd6:    sample_bits = 8;
      default: sample_bits = 16;
    endcase
  end

  // Expected output for stored sample j of the waveform: the memory read as a stream of
  // sample_bits-wide samples, MSB-aligned in 16 bits.
  function [15:0] expected_sample;
    input [63:0] j;
    reg [63:0] bit_index;
    reg [15:0] word16;
    integer b;
    begin
      expected_sample = 16'd0;
      for (b = 0; b < sample_bits; b = b + 1) begin
        bit_index = exp_first * 64'd16 + j * sample_bits + b;
        word16    = pattern(bit_index >> 4);
        expected_sample[16 - sample_bits + b] = word16[bit_index[3:0]];
      end
    end
  endfunction

  always @(posedge axis_clk) begin
    if (tvalid && tready && is_data) begin
      for (k = 0; k < 2 * per_half; k = k + 1) begin
        if (k < per_half)
          got = tdata[16*k +: 16];
        else
          got = tdata[256 + 16*(k - per_half) +: 16];
        want = expected_sample(checked % exp_len);
        if (got !== want) begin
          errors = errors + 1;
          if (errors < 10)
            $display("ERROR: sample %0d is %h, expected %h at %0t", checked, got, want, $time);
        end
        checked = checked + 1;
      end
      // everything above the valid samples must be zero
      if ((tdata[255:0] >> (16 * per_half)) != 0 || (tdata[511:256] >> (16 * per_half)) != 0) begin
        errors = errors + 1;
        $display("ERROR: unused sample slots are not zero at %0t", $time);
      end
    end

    // mirror of how the DUT loads m_axis_tdata
    if (!uut.rd_active)
      is_data <= 1'b0;
    else if (mode[2]) begin
      if (tready)
        is_data <= uut.pk_v;
    end else if (tready && uut.rstate == 2'd2)
      is_data <= (uut.cnt != 4'd0) || uut.fifo_rd_valid;
    else if (tready && uut.rstate == 2'd3)
      is_data <= 1'b0;
  end

  // ------------------------------------------------------------ test sequencing
  integer tests_run = 0;
  integer timeout;

  task wait_status;
    input integer bit_index;
    input         value;
    input integer max_cycles;
    begin
      timeout = 0;
      while (status[bit_index] !== value && timeout < max_cycles) begin
        @(posedge axis_clk);
        timeout = timeout + 1;
      end
      if (status[bit_index] !== value) begin
        errors = errors + 1;
        $display("ERROR: timeout waiting for status[%0d] = %0d at %0t", bit_index, value, $time);
      end
    end
  endtask

  task arm;
    input [2:0]  sel;
    input [20:0] page;
    input [27:0] words;
    input [23:0] loops;
    input        forever_mode;
    begin
      mode         = sel;
      base_page    = page;
      num_words    = words;
      loop_count   = loops;
      loop_forever = forever_mode;
      exp_first    = {43'd0, page} * 64'd2048;     // 4096 bytes = 2048 samples per page
      checked      = 0;
      repeat (4) @(posedge axis_clk);              // `mode` settles, sample_bits follows
      exp_len      = {36'd0, words} * 64'd512 / sample_bits;
      enable = 1'b1;
      wait_status(1, 1'b1, 200000);                // ready
    end
  endtask

  task fire;
    begin
      @(posedge axis_clk);
      sw_trigger = 1'b1;
      repeat (4) @(posedge axis_clk);
      sw_trigger = 1'b0;
    end
  endtask

  task disarm;
    begin
      enable = 1'b0;
      wait_status(0, 1'b0, 200000);                // not busy
      repeat (200) @(posedge axis_clk);
      if (tdata !== 512'd0) begin
        errors = errors + 1;
        $display("ERROR: output is not zero while idle at %0t", $time);
      end
    end
  endtask

  task play_once;
    input [2:0]  sel;
    input [20:0] page;
    input [27:0] words;
    input [23:0] loops;
    integer      want_samples;
    begin
      arm(sel, page, words, loops, 1'b0);
      fire;
      wait_status(3, 1'b1, 4000000);               // done
      repeat (40) @(posedge axis_clk);
      want_samples = words * 512 / sample_bits * loops;
      if (checked != want_samples) begin
        errors = errors + 1;
        $display("ERROR: %0d samples played, expected %0d", checked, want_samples);
      end
      if (st_underflow || st_rd_err) begin
        errors = errors + 1;
        $display("ERROR: underflow=%0d read_error=%0d", st_underflow, st_rd_err);
      end
      tests_run = tests_run + 1;
      $display("  mode=%0d page=%0d words=%0d loops=%0d: %0d samples checked, %0d errors so far",
               sel, page, words, loops, checked, errors);
      disarm;
    end
  endtask

  initial begin
    repeat (20) @(posedge ui_clk);
    ui_aresetn = 1'b1;
    repeat (20) @(posedge axis_clk);
    axis_aresetn = 1'b1;
    repeat (20) @(posedge axis_clk);

    $display("one-shot playback, all rate modes and awkward lengths");
    play_once(3'd0, 21'd5, 28'd1,    24'd1);
    play_once(3'd0, 21'd5, 28'd63,   24'd1);
    play_once(3'd1, 21'd5, 28'd64,   24'd1);
    play_once(3'd2, 21'd5, 28'd65,   24'd1);
    play_once(3'd3, 21'd5, 28'd200,  24'd1);
    play_once(3'd0, 21'd7, 28'd2500, 24'd1);
    play_once(3'd1, 21'd7, 28'd2500, 24'd1);

    $display("counted loops");
    play_once(3'd1, 21'd9, 28'd100,  24'd3);
    play_once(3'd0, 21'd9, 28'd70,   24'd4);
    play_once(3'd2, 21'd9, 28'd1,    24'd5);

    $display("full rate, bit-packed samples");
    play_once(3'd4, 21'd6, 28'd7,    24'd1);       // 14 bit: groups of 7 words = 256 samples
    play_once(3'd4, 21'd6, 28'd259,  24'd1);
    play_once(3'd5, 21'd6, 28'd120,  24'd1);       // 12 bit: groups of 3 words = 128 samples
    play_once(3'd6, 21'd6, 28'd50,   24'd1);       //  8 bit: 1 word = 64 samples
    play_once(3'd4, 21'd3, 28'd21,   24'd3);

    $display("loop forever, abort, then a different waveform");
    arm(3'd0, 21'd2, 28'd130, 24'd1, 1'b1);
    fire;
    repeat (3000) @(posedge axis_clk);
    if (!st_running || checked < 5 * 130 * 32) begin
      errors = errors + 1;
      $display("ERROR: looping playback is not running (checked=%0d)", checked);
    end
    disarm;
    play_once(3'd2, 21'd40, 28'd77, 24'd1);

    $display("external trigger");
    ext_trig_en = 1'b1;
    arm(3'd1, 21'd3, 28'd50, 24'd1, 1'b0);
    #7.3 ext_trigger = 1'b1;                       // deliberately asynchronous
    wait_status(3, 1'b1, 100000);
    repeat (40) @(posedge axis_clk);
    ext_trigger = 1'b0;
    if (checked != 50 * 32) begin
      errors = errors + 1;
      $display("ERROR: external trigger played %0d samples, expected %0d", checked, 50 * 32);
    end
    ext_trig_en = 1'b0;
    disarm;

    if (errors == 0 && axi_errors == 0)
      $display("PASS: %0d playback tests, no errors", tests_run);
    else
      $display("FAIL: %0d data/status errors, %0d AXI protocol errors", errors, axi_errors);
    $finish;
  end

  initial begin
    #200000000;
    $display("FAIL: global timeout");
    $finish;
  end

endmodule
