`timescale 1ns / 1ps

module tb_DACRAMstreamer_vector_count;

  localparam integer DWIDTH = 512;
  localparam integer MEM_SIZE_BYTES = 1024;
  localparam integer WORD_BYTES = DWIDTH / 8;
  localparam integer SAMPLES_PER_WORD = DWIDTH / 16;
  localparam integer DEPTH = MEM_SIZE_BYTES / WORD_BYTES;
  localparam integer ADDR_LSB_BITS = $clog2(WORD_BYTES);
  localparam integer WORD_ADDR_BITS = $clog2(DEPTH);

  reg axis_clk = 1'b0;
  reg axis_aresetn = 1'b0;
  reg axis_tready = 1'b1;
  reg [17:0] numSamples = 18'd0;
  reg enable = 1'b0;

  reg [DWIDTH-1:0] mem [0:DEPTH-1];
  reg [DWIDTH-1:0] portA_cpu_rdata = {DWIDTH{1'b0}};

  wire [DWIDTH-1:0] portA_cpu_wdata;
  wire [DWIDTH/8-1:0] portA_we;
  wire portA_en;
  wire [31:0] portAcpu_addr;
  wire portA_clk;
  wire portA_rst;
  wire [DWIDTH-1:0] axis_tdata;
  wire axis_tvalid;

  wire [WORD_ADDR_BITS-1:0] word_addr;

  integer checks = 0;

  assign word_addr = portAcpu_addr[ADDR_LSB_BITS + WORD_ADDR_BITS - 1 : ADDR_LSB_BITS];

  always #5 axis_clk = ~axis_clk;

  DACRAMstreamer #(
    .DWIDTH(DWIDTH),
    .MEM_SIZE_BYTES(MEM_SIZE_BYTES),
    .USE_VECTOR_COUNT(1)
  ) dut (
    .portA_cpu_wdata(portA_cpu_wdata),
    .portA_we(portA_we),
    .portA_en(portA_en),
    .portA_cpu_rdata(portA_cpu_rdata),
    .portAcpu_addr(portAcpu_addr),
    .portA_clk(portA_clk),
    .portA_rst(portA_rst),
    .axis_clk(axis_clk),
    .axis_aresetn(axis_aresetn),
    .axis_tdata(axis_tdata),
    .axis_tready(axis_tready),
    .axis_tvalid(axis_tvalid),
    .numSamples(numSamples),
    .enable(enable)
  );

  always @(posedge axis_clk) begin
    if (!axis_aresetn)
      portA_cpu_rdata <= {DWIDTH{1'b0}};
    else if (portA_en)
      portA_cpu_rdata <= mem[word_addr];
  end

  task init_memory;
    integer word_idx;
    integer lane_idx;
    reg [15:0] sample;
    begin
      for (word_idx = 0; word_idx < DEPTH; word_idx = word_idx + 1) begin
        mem[word_idx] = {DWIDTH{1'b0}};
        for (lane_idx = 0; lane_idx < SAMPLES_PER_WORD; lane_idx = lane_idx + 1) begin
          sample = (word_idx * SAMPLES_PER_WORD + lane_idx) & 16'hffff;
          mem[word_idx][lane_idx * 16 +: 16] = sample;
        end
      end
    end
  endtask

  task apply_reset;
    begin
      enable = 1'b0;
      axis_tready = 1'b1;
      numSamples = 18'd0;
      axis_aresetn = 1'b0;
      repeat (4) @(posedge axis_clk);
      axis_aresetn = 1'b1;
      repeat (4) @(posedge axis_clk);
    end
  endtask

  task configure_disabled;
    input [17:0] samples;
    begin
      enable = 1'b0;
      numSamples = samples;
      repeat (8) @(posedge axis_clk);
    end
  endtask

  task start_streaming;
    begin
      enable = 1'b1;
      repeat (4) @(posedge axis_clk);
    end
  endtask

  task stop_streaming;
    begin
      enable = 1'b0;
      repeat (4) @(posedge axis_clk);
    end
  endtask

  function integer expected_vectors;
    input integer samples;
    integer rounded;
    begin
      rounded = (samples + SAMPLES_PER_WORD - 1) / SAMPLES_PER_WORD;
      if (samples == 0 || rounded > DEPTH)
        expected_vectors = DEPTH;
      else
        expected_vectors = rounded;
    end
  endfunction

  task expect_addr;
    input [1023:0] tag;
    input integer exp_word_idx;
    integer exp_addr;
    begin
      exp_addr = exp_word_idx * WORD_BYTES;
      if (axis_tvalid !== 1'b1 || portA_en !== 1'b1) begin
        $display("[%0t] ERROR %0s: streamer not valid/enabled", $time, tag);
        $display("  axis_tvalid=%b portA_en=%b", axis_tvalid, portA_en);
        $fatal;
      end
      if (portAcpu_addr !== exp_addr[31:0]) begin
        $display("[%0t] ERROR %0s: address mismatch", $time, tag);
        $display("  expected word=%0d addr=0x%08h", exp_word_idx, exp_addr[31:0]);
        $display("  actual        addr=0x%08h", portAcpu_addr);
        $fatal;
      end
      checks = checks + 1;
    end
  endtask

  task step_expect_addr;
    input [1023:0] tag;
    input integer exp_word_idx;
    begin
      @(posedge axis_clk);
      #1;
      expect_addr(tag, exp_word_idx);
    end
  endtask

  task check_loop;
    input [1023:0] tag;
    input integer samples;
    input integer cycles;
    integer nvec;
    integer cycle;
    integer prev_word_idx;
    integer curr_word_idx;
    integer exp_word_idx;
    begin
      $display("[%0t] Checking %0s: numSamples=%0d", $time, tag, samples);
      configure_disabled(samples[17:0]);
      start_streaming();
      nvec = expected_vectors(samples);

      @(posedge axis_clk);
      #1;
      prev_word_idx = portAcpu_addr / WORD_BYTES;
      if (prev_word_idx >= nvec) begin
        $display("[%0t] ERROR %0s: address outside programmed loop", $time, tag);
        $display("  nvec=%0d word_idx=%0d addr=0x%08h", nvec, prev_word_idx, portAcpu_addr);
        $fatal;
      end
      checks = checks + 1;

      for (cycle = 1; cycle < cycles; cycle = cycle + 1) begin
        @(posedge axis_clk);
        #1;
        curr_word_idx = portAcpu_addr / WORD_BYTES;
        exp_word_idx = (prev_word_idx + 1) % nvec;
        if (axis_tvalid !== 1'b1 || portA_en !== 1'b1) begin
          $display("[%0t] ERROR %0s: streamer not valid/enabled", $time, tag);
          $display("  axis_tvalid=%b portA_en=%b", axis_tvalid, portA_en);
          $fatal;
        end
        if (curr_word_idx !== exp_word_idx) begin
          $display("[%0t] ERROR %0s: address sequence mismatch", $time, tag);
          $display("  nvec=%0d previous=%0d expected=%0d actual=%0d addr=0x%08h", nvec, prev_word_idx, exp_word_idx, curr_word_idx, portAcpu_addr);
          $fatal;
        end
        prev_word_idx = curr_word_idx;
        checks = checks + 1;
      end
      stop_streaming();
    end
  endtask

  task check_ready_stall;
    reg [31:0] held_addr;
    integer held_word_idx;
    begin
      $display("[%0t] Checking axis_tready stall", $time);
      configure_disabled(18'd96);
      start_streaming();
      @(posedge axis_clk);
      #1;
      held_addr = portAcpu_addr;
      held_word_idx = held_addr / WORD_BYTES;
      if (held_word_idx >= 3) begin
        $display("[%0t] ERROR ready_stall: address outside programmed loop", $time);
        $display("  word_idx=%0d addr=0x%08h", held_word_idx, held_addr);
        $fatal;
      end
      checks = checks + 1;
      axis_tready = 1'b0;
      repeat (5) begin
        @(posedge axis_clk);
        #1;
        if (portAcpu_addr !== held_addr) begin
          $display("[%0t] ERROR ready_stall: address advanced while tready low", $time);
          $display("  held=0x%08h actual=0x%08h", held_addr, portAcpu_addr);
          $fatal;
        end
        checks = checks + 1;
      end
      axis_tready = 1'b1;
      step_expect_addr("ready_stall/resume", (held_word_idx + 1) % 3);
      stop_streaming();
    end
  endtask

  task check_runtime_numSamples_ignored;
    integer cycle;
    integer prev_word_idx;
    integer curr_word_idx;
    integer exp_word_idx;
    begin
      $display("[%0t] Checking numSamples change while enabled", $time);
      configure_disabled(18'd320);
      start_streaming();
      numSamples = 18'd32;

      @(posedge axis_clk);
      #1;
      prev_word_idx = portAcpu_addr / WORD_BYTES;
      if (prev_word_idx >= 10) begin
        $display("[%0t] ERROR runtime_change: address outside original programmed loop", $time);
        $display("  word_idx=%0d addr=0x%08h", prev_word_idx, portAcpu_addr);
        $fatal;
      end
      checks = checks + 1;

      for (cycle = 1; cycle < 16; cycle = cycle + 1) begin
        @(posedge axis_clk);
        #1;
        curr_word_idx = portAcpu_addr / WORD_BYTES;
        exp_word_idx = (prev_word_idx + 1) % 10;
        if (curr_word_idx !== exp_word_idx) begin
          $display("[%0t] ERROR runtime_change: enabled numSamples update affected active loop", $time);
          $display("  previous=%0d expected=%0d actual=%0d addr=0x%08h", prev_word_idx, exp_word_idx, curr_word_idx, portAcpu_addr);
          $fatal;
        end
        prev_word_idx = curr_word_idx;
        checks = checks + 1;
      end
      stop_streaming();
    end
  endtask

  initial begin
    $dumpfile("tb_DACRAMstreamer_vector_count.vcd");
    $dumpvars(0, tb_DACRAMstreamer_vector_count);

    init_memory();
    apply_reset();

    if (portA_cpu_wdata !== {DWIDTH{1'b0}}) begin
      $display("ERROR: BRAM write data is not tied low");
      $fatal;
    end
    if (portA_we !== {(DWIDTH/8){1'b0}}) begin
      $display("ERROR: BRAM write enable is not tied low");
      $fatal;
    end
    if (portA_clk !== axis_clk) begin
      $display("ERROR: portA_clk does not follow axis_clk");
      $fatal;
    end
    if (portA_rst !== ~axis_aresetn) begin
      $display("ERROR: portA_rst does not follow axis_aresetn");
      $fatal;
    end
    checks = checks + 4;

    check_loop("one_sample_rounds_to_one_vector", 1, 8);
    check_loop("one_word_exact", 32, 8);
    check_loop("thirty_three_samples_rounds_to_two_vectors", 33, 12);
    check_loop("ten_words_exact", 320, 24);
    check_loop("zero_samples_means_full_bram", 0, 34);
    check_ready_stall();
    check_runtime_numSamples_ignored();

    $display("[%0t] PASS: completed %0d checks", $time, checks);
    $finish;
  end

endmodule
