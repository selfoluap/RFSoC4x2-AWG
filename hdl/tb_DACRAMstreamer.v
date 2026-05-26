`timescale 1ns / 1ps

module tb_DACRAMstreamer;

  localparam integer DWIDTH            = 512;
  localparam integer MEM_SIZE_BYTES    = 262144;
  localparam real    WORD_CLK_PERIOD_NS = 3.368;
  localparam integer WORD_BYTES        = DWIDTH / 8;
  localparam integer DEPTH             = MEM_SIZE_BYTES / WORD_BYTES;
  localparam integer ADDR_LSB_BITS     = $clog2(WORD_BYTES);
  localparam integer WORD_ADDR_BITS    = $clog2(DEPTH);
  localparam integer SAMPLES_PER_WORD  = DWIDTH / 16;
  localparam real    SAMPLE_PERIOD_NS   = WORD_CLK_PERIOD_NS / SAMPLES_PER_WORD;
  localparam real    SAMPLE_CLK_HALF_NS = SAMPLE_PERIOD_NS / 2.0;
  localparam integer ACTIVITY_THRESHOLD = 16'd1000;
  localparam integer LAST_ADDR         = MEM_SIZE_BYTES - WORD_BYTES;
  localparam integer CONTROL_HOLD_CYCLES = 8;
  localparam integer REPLAY_ON_CYCLES    = 64;
  localparam integer REPLAY_OFF_CYCLES   = 16;
  localparam integer REPLAY_TOGGLE_COUNT = 3;

  reg axis_clk;
  reg axis_aresetn;
  reg axis_tready;
  reg cap_activity_active;
  reg sample_view_clk;
  reg [$clog2(SAMPLES_PER_WORD)-1:0] sample_idx;
  reg [$clog2(SAMPLES_PER_WORD)-1:0] sample_idx_view;
  integer sample_count_view;

  integer checks;
  reg [1023:0] memfile;

  reg  [DWIDTH-1:0] mem [0:DEPTH-1];
  reg  [DWIDTH-1:0] portA_cpu_rdata;
  reg  [DWIDTH-1:0] cap_axis_tdata;
  reg               cap_axis_tvalid;
  reg  [DWIDTH-1:0] axis_word_view;
  reg  [DWIDTH-1:0] bram_word_view;
  wire [DWIDTH-1:0] portA_cpu_wdata;
  wire [DWIDTH/8-1:0] portA_we;
  wire                portA_en;
  wire [31:0]         portAcpu_addr;
  wire                portA_clk;
  wire                portA_rst;
  wire [DWIDTH-1:0]   axis_tdata;
  wire                axis_tvalid;
  wire                cap_axis_tready;
  wire                enable;
  wire [WORD_ADDR_BITS-1:0] word_addr;
  wire [DWIDTH-1:0]   current_mem_word;
  reg signed [15:0]   axis_wave_sample;
  reg signed [15:0]   bram_wave_sample;
  integer             axis_wave_sample_dec;
  integer             bram_wave_sample_dec;

  assign word_addr        = portAcpu_addr[ADDR_LSB_BITS + WORD_ADDR_BITS - 1 : ADDR_LSB_BITS];
  assign current_mem_word = mem[word_addr];

  initial begin
    axis_clk = 1'b0;
    // 296.875 MHz AXI clock: 3.368421 ns period, 1.6842105 ns half-period.
    forever #1.684 axis_clk = ~axis_clk;
  end

  initial begin
    sample_view_clk = 1'b0;
    #(WORD_CLK_PERIOD_NS / 2.0 + SAMPLE_PERIOD_NS / 2.0);
    forever #SAMPLE_CLK_HALF_NS sample_view_clk = ~sample_view_clk;
  end

  DACRAMstreamer #(
    .DWIDTH(DWIDTH),
    .MEM_SIZE_BYTES(MEM_SIZE_BYTES)
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
    .enable(enable)
  );

  ActivityDetector #(
    .DWIDTH(DWIDTH),
    .SAMPLE_W(16),
    .THRESHOLD(ACTIVITY_THRESHOLD),
    .HOLD_COUNT(CONTROL_HOLD_CYCLES)
  ) activity_detector (
    .axis_clk(axis_clk),
    .axis_aresetn(axis_aresetn),
    .CAP_AXIS_tdata(cap_axis_tdata),
    .CAP_AXIS_tvalid(cap_axis_tvalid),
    .CAP_AXIS_tready(cap_axis_tready),
    .enable(enable)
  );

  task init_memory;
    integer word_idx;
    integer lane_idx;
    reg [15:0] sample;
    begin
      for (word_idx = 0; word_idx < DEPTH; word_idx = word_idx + 1) begin
        mem[word_idx] = {DWIDTH{1'b0}};
        for (lane_idx = 0; lane_idx < SAMPLES_PER_WORD; lane_idx = lane_idx + 1) begin
          sample = (word_idx * SAMPLES_PER_WORD + lane_idx) & 16'hffff;
          mem[word_idx][(lane_idx * 16) +: 16] = sample;
        end
      end
    end
  endtask

  task load_memory;
    begin
      init_memory();
      if ($value$plusargs("memfile=%s", memfile)) begin
        $display("[%0t] Loading BRAM contents from %0s", $time, memfile);
        $readmemh(memfile, mem);
      end else begin
        $display("[%0t] Using built-in BRAM test pattern", $time);
      end
    end
  endtask

  task apply_reset;
    begin
      axis_aresetn    = 1'b0;
      axis_tready     = 1'b1;
      cap_activity_active = 1'b1;
      cap_axis_tvalid = 1'b1;
      cap_axis_tdata  = {DWIDTH{1'b0}};
      portA_cpu_rdata = {DWIDTH{1'b0}};
      repeat (4) @(posedge axis_clk);
      axis_aresetn = 1'b1;
    end
  endtask

  task set_detector_activity;
    input active;
    begin
      cap_activity_active = active;
      if (active)
        cap_axis_tdata = {{(DWIDTH-16){1'b0}}, ACTIVITY_THRESHOLD + 16'd500};
      else
        cap_axis_tdata = {DWIDTH{1'b0}};
      cap_axis_tvalid = 1'b1;
    end
  endtask

  task check_no_x;
    input [1023:0] tag;
    begin
      if ((axis_tvalid !== 1'b0) && (axis_tvalid !== 1'b1)) begin
        $display("[%0t] ERROR %0s: axis_tvalid is X/Z", $time, tag);
        $fatal;
      end
      if ((portA_en !== 1'b0) && (portA_en !== 1'b1)) begin
        $display("[%0t] ERROR %0s: portA_en is X/Z", $time, tag);
        $fatal;
      end
      if (^axis_tdata === 1'bx) begin
        $display("[%0t] ERROR %0s: axis_tdata contains X", $time, tag);
        $fatal;
      end
      if (^portAcpu_addr === 1'bx) begin
        $display("[%0t] ERROR %0s: portAcpu_addr contains X", $time, tag);
        $fatal;
      end
    end
  endtask

  task expect_full;
    input [1023:0] tag;
    input          exp_tvalid;
    input          exp_portA_en;
    input [31:0]   exp_addr;
    input [DWIDTH-1:0] exp_tdata;
    begin
      check_no_x(tag);
      if (axis_tvalid !== exp_tvalid) begin
        $display("[%0t] ERROR %0s: axis_tvalid mismatch", $time, tag);
        $display("  expected = %b", exp_tvalid);
        $display("  actual   = %b", axis_tvalid);
        $fatal;
      end
      if (portA_en !== exp_portA_en) begin
        $display("[%0t] ERROR %0s: portA_en mismatch", $time, tag);
        $display("  expected = %b", exp_portA_en);
        $display("  actual   = %b", portA_en);
        $fatal;
      end
      if (portAcpu_addr !== exp_addr) begin
        $display("[%0t] ERROR %0s: portAcpu_addr mismatch", $time, tag);
        $display("  expected = 0x%08h", exp_addr);
        $display("  actual   = 0x%08h", portAcpu_addr);
        $fatal;
      end
      if (axis_tdata !== exp_tdata) begin
        $display("[%0t] ERROR %0s: axis_tdata mismatch", $time, tag);
        $display("  expected = %h", exp_tdata);
        $display("  actual   = %h", axis_tdata);
        $fatal;
      end
      checks = checks + 1;
    end
  endtask

  task expect_ctrl;
    input [1023:0] tag;
    input          exp_tvalid;
    input          exp_portA_en;
    input [31:0]   exp_addr;
    begin
      check_no_x(tag);
      if (axis_tvalid !== exp_tvalid) begin
        $display("[%0t] ERROR %0s: axis_tvalid mismatch", $time, tag);
        $display("  expected = %b", exp_tvalid);
        $display("  actual   = %b", axis_tvalid);
        $fatal;
      end
      if (portA_en !== exp_portA_en) begin
        $display("[%0t] ERROR %0s: portA_en mismatch", $time, tag);
        $display("  expected = %b", exp_portA_en);
        $display("  actual   = %b", portA_en);
        $fatal;
      end
      if (portAcpu_addr !== exp_addr) begin
        $display("[%0t] ERROR %0s: portAcpu_addr mismatch", $time, tag);
        $display("  expected = 0x%08h", exp_addr);
        $display("  actual   = 0x%08h", portAcpu_addr);
        $fatal;
      end
      checks = checks + 1;
    end
  endtask

  task step_expect_full;
    input [1023:0] tag;
    input          exp_tvalid;
    input          exp_portA_en;
    input [31:0]   exp_addr;
    input [DWIDTH-1:0] exp_tdata;
    begin
      @(posedge axis_clk);
      #1;
      expect_full(tag, exp_tvalid, exp_portA_en, exp_addr, exp_tdata);
    end
  endtask

  task step_expect_ctrl;
    input [1023:0] tag;
    input          exp_tvalid;
    input          exp_portA_en;
    input [31:0]   exp_addr;
    begin
      @(posedge axis_clk);
      #1;
      expect_ctrl(tag, exp_tvalid, exp_portA_en, exp_addr);
    end
  endtask

  task step_expect_word;
    input [1023:0] tag;
    input integer  word_idx;
    input [31:0]   exp_addr;
    begin
      @(posedge axis_clk);
      #1;
      expect_full(tag, 1'b1, 1'b1, exp_addr, mem[word_idx]);
    end
  endtask

  always @(posedge axis_clk) begin
    if (!axis_aresetn) begin
      portA_cpu_rdata <= {DWIDTH{1'b0}};
      axis_word_view  <= {DWIDTH{1'b0}};
      bram_word_view  <= {DWIDTH{1'b0}};
      if (cap_activity_active)
        cap_axis_tdata <= {{(DWIDTH-16){1'b0}}, ACTIVITY_THRESHOLD + 16'd500};
      else
        cap_axis_tdata <= {DWIDTH{1'b0}};
    end else if (portA_en) begin
      portA_cpu_rdata <= mem[word_addr];
      axis_word_view  <= axis_tvalid ? axis_tdata : {DWIDTH{1'b0}};
      bram_word_view  <= portA_cpu_rdata;
      if (cap_activity_active)
        cap_axis_tdata <= {{(DWIDTH-16){1'b0}}, ACTIVITY_THRESHOLD + 16'd500};
      else
        cap_axis_tdata <= {DWIDTH{1'b0}};
    end else begin
      axis_word_view  <= axis_tvalid ? axis_tdata : {DWIDTH{1'b0}};
      bram_word_view  <= {DWIDTH{1'b0}};
      if (cap_activity_active)
        cap_axis_tdata <= {{(DWIDTH-16){1'b0}}, ACTIVITY_THRESHOLD + 16'd500};
      else
        cap_axis_tdata <= {DWIDTH{1'b0}};
    end
  end

  always @(posedge sample_view_clk or negedge axis_aresetn) begin
    if (!axis_aresetn) begin
      sample_idx        <= {$clog2(SAMPLES_PER_WORD){1'b0}};
      sample_idx_view   <= {$clog2(SAMPLES_PER_WORD){1'b0}};
      sample_count_view <= 0;
      axis_wave_sample  <= 16'sd0;
      bram_wave_sample  <= 16'sd0;
      axis_wave_sample_dec <= 0;
      bram_wave_sample_dec <= 0;
    end else begin
      sample_idx_view  <= sample_idx;
      sample_count_view <= sample_count_view + 1;
      axis_wave_sample <= $signed(axis_word_view[(sample_idx * 16) +: 16]);
      bram_wave_sample <= $signed(bram_word_view[(sample_idx * 16) +: 16]);
      axis_wave_sample_dec <= $signed(axis_word_view[(sample_idx * 16) +: 16]);
      bram_wave_sample_dec <= $signed(bram_word_view[(sample_idx * 16) +: 16]);
      if (sample_idx == SAMPLES_PER_WORD - 1)
        sample_idx <= {$clog2(SAMPLES_PER_WORD){1'b0}};
      else
        sample_idx <= sample_idx + 1'b1;
    end
  end

  initial begin
    $dumpfile("tb_DACRAMstreamer.vcd");
    $dumpvars(0, axis_wave_sample);
    $dumpvars(0, bram_wave_sample);
    $dumpvars(0, sample_count_view);
    $dumpvars(0, sample_idx_view);
    $dumpvars(0, sample_view_clk);
    $dumpvars(0, axis_clk);
    $dumpvars(0, enable);
    $dumpvars(0, cap_axis_tdata);

    checks = 0;
    load_memory();
    apply_reset();

    step_expect_full("post_reset/cycle0", 1'b1, 1'b0, 32'd0, {DWIDTH{1'b0}});
    if (portA_cpu_wdata !== {DWIDTH{1'b0}}) begin
      $display("[%0t] ERROR post_reset/wdata: BRAM write data is not tied low", $time);
      $fatal;
    end
    if (portA_we !== {(DWIDTH/8){1'b0}}) begin
      $display("[%0t] ERROR post_reset/we: BRAM write enable is not tied low", $time);
      $fatal;
    end
    if (portA_clk !== axis_clk) begin
      $display("[%0t] ERROR post_reset/clk: portA_clk does not follow axis_clk", $time);
      $fatal;
    end
    if (portA_rst !== ~axis_aresetn) begin
      $display("[%0t] ERROR post_reset/rst: portA_rst does not follow axis_aresetn", $time);
      $fatal;
    end
    checks = checks + 4;

    step_expect_full("startup/cycle0", 1'b1, 1'b1, 32'd0, {DWIDTH{1'b0}});
    step_expect_full("startup/cycle1", 1'b1, 1'b1, WORD_BYTES, {DWIDTH{1'b0}});
    step_expect_word("startup/word0", 0, 2 * WORD_BYTES);
    step_expect_word("startup/word1", 1, 3 * WORD_BYTES);

    $display("[%0t] Checking address hold while axis_tready is low", $time);
    axis_tready = 1'b0;
    step_expect_ctrl("ready_low/cycle0", 1'b1, 1'b1, 3 * WORD_BYTES);
    step_expect_ctrl("ready_low/cycle1", 1'b1, 1'b1, 3 * WORD_BYTES);
    step_expect_full("ready_low/steady_data", 1'b1, 1'b1, 3 * WORD_BYTES, mem[3]);

    axis_tready = 1'b1;
    step_expect_ctrl("ready_high/resume0", 1'b1, 1'b1, 4 * WORD_BYTES);
    step_expect_ctrl("ready_high/resume1", 1'b1, 1'b1, 5 * WORD_BYTES);

    $display("[%0t] Checking wrap from last BRAM word back to address 0", $time);
    @(negedge axis_clk);
    dut.portAcpu_addr = LAST_ADDR;
    dut.enable_d      = 1'b1;
    @(posedge axis_clk);
    #1;
    expect_ctrl("wrap/cycle0", 1'b1, 1'b1, 32'd0);
    step_expect_ctrl("wrap/cycle1", 1'b1, 1'b1, WORD_BYTES);

    $display("[%0t] Checking disabled output behavior", $time);
    set_detector_activity(1'b0);
    repeat (CONTROL_HOLD_CYCLES + 1) @(posedge axis_clk);
    #1;
    expect_full("disable/cycle0", 1'b1, 1'b0, 32'd0, {DWIDTH{1'b0}});
    repeat (CONTROL_HOLD_CYCLES - 1) begin
      step_expect_full("disable/hold", 1'b1, 1'b0, 32'd0, {DWIDTH{1'b0}});
    end

    $display("[%0t] Checking restart after enable rising edge", $time);
    set_detector_activity(1'b1);
    step_expect_full("enable_rise/cycle0", 1'b1, 1'b0, 32'd0, {DWIDTH{1'b0}});
    step_expect_ctrl("enable_rise/cycle1", 1'b1, 1'b1, 32'd0);
    step_expect_ctrl("enable_rise/cycle2", 1'b1, 1'b1, WORD_BYTES);
    step_expect_full("enable_rise/restart_word0", 1'b1, 1'b1, 2 * WORD_BYTES, mem[0]);
    step_expect_word("enable_rise/restart_word1", 1, 3 * WORD_BYTES);

    $display("[%0t] Replaying waveform with %0d enable bursts (%0d on / %0d off AXI cycles)", $time, REPLAY_TOGGLE_COUNT, REPLAY_ON_CYCLES, REPLAY_OFF_CYCLES);
    repeat (REPLAY_TOGGLE_COUNT) begin
      set_detector_activity(1'b1);
      repeat (REPLAY_ON_CYCLES) @(posedge axis_clk);
      set_detector_activity(1'b0);
      repeat (REPLAY_OFF_CYCLES) @(posedge axis_clk);
    end
    set_detector_activity(1'b1);
    repeat (REPLAY_ON_CYCLES) @(posedge axis_clk);

    $display("[%0t] PASS: completed %0d checks", $time, checks);
    $finish;
  end

endmodule
