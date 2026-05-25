// -------------------------------------------------------------------------------------------------
// Copyright (C) 2023 Advanced Micro Devices, Inc
// SPDX-License-Identifier: MIT
// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- --
`timescale 1ns / 1ps

module DACRAMstreamer #( parameter DWIDTH = 512, parameter MEM_SIZE_BYTES = 262144, parameter USE_VECTOR_COUNT = 0 ) (
  (* X_INTERFACE_PARAMETER = "MASTER_TYPE BRAM_CTRL, READ_WRITE_MODE READ, MEM_SIZE 262144, MEM_WIDTH 512" *)

  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_A DIN" *)
  output wire [DWIDTH-1:0] portA_cpu_wdata, // Data In Bus (optional)

  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_A WE" *)
  output [DWIDTH/8-1:0] portA_we, // Byte Enables (optional)

  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_A EN" *)
  output reg portA_en, // Chip Enable Signal (optional)

  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_A DOUT" *)
  input wire [DWIDTH-1:0] portA_cpu_rdata, // Data Out Bus (optional)

  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_A ADDR" *)
  output reg [31:0] portAcpu_addr, /// Address Signal (required)

  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_A CLK" *)
  output wire portA_clk, // Clock Signal (required)

  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_A RST" *)
  output wire portA_rst, // Reset Signal (required)

  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 axis_clk CLK" *)
  (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF AXIS, ASSOCIATED_RESET axis_aresetn" *)
  input wire axis_clk,
  (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 axis_aresetn RST" *)
  input  wire              axis_aresetn,
  output reg  [DWIDTH-1:0] axis_tdata,       // luckily rest of AXIS is inferred properly
  input  wire              axis_tready,
  output reg               axis_tvalid,

  // Control Input Parameters
  input  [17:0] numSamples,
  input                    enable );

  localparam integer WORD_BYTES   = DWIDTH/8;
  localparam integer SAMPLES_PER_WORD = DWIDTH/16;
  localparam integer VECTOR_COUNT = MEM_SIZE_BYTES/WORD_BYTES;
  localparam integer VCNT_WIDTH   = $clog2(VECTOR_COUNT);
  localparam integer LAST_ADDR    = MEM_SIZE_BYTES - WORD_BYTES; // last valid start byte address

  (* ASYNC_REG = "TRUE" *) reg [17:0] numSamples_meta;
  (* ASYNC_REG = "TRUE" *) reg [17:0] numSamples_axis;
  (* ASYNC_REG = "TRUE" *) reg enable_meta;
  (* ASYNC_REG = "TRUE" *) reg enable_axis;

  reg [VCNT_WIDTH:0] loopVectorCount;
  reg [VCNT_WIDTH:0] vectorsRemaining;

  function [VCNT_WIDTH:0] sample_count_to_vectors;
    input [17:0] sample_count;
    reg [31:0] rounded_vectors;
    begin
      rounded_vectors = (sample_count + SAMPLES_PER_WORD - 1) / SAMPLES_PER_WORD;
      if (sample_count == 0 || rounded_vectors > VECTOR_COUNT)
        sample_count_to_vectors = VECTOR_COUNT;
      else
        sample_count_to_vectors = rounded_vectors[VCNT_WIDTH:0];
    end
  endfunction
  
  assign portA_cpu_wdata = 0;
  assign portA_clk       = axis_clk;
  assign portA_rst       = ~axis_aresetn;
  assign portA_we        = 0;

  wire advance = enable_axis && axis_tvalid && axis_tready;

always @(posedge axis_clk) begin
  if (~axis_aresetn) begin
    axis_tvalid   <= 1'b0;
    axis_tdata    <= {DWIDTH{1'b0}};
    portA_en      <= 1'b0;
    portAcpu_addr <= 32'd0;
    numSamples_meta <= 18'd0;
    numSamples_axis <= 18'd0;
    enable_meta <= 1'b0;
    enable_axis <= 1'b0;
    loopVectorCount <= VECTOR_COUNT;
    vectorsRemaining <= VECTOR_COUNT;
  end else begin
    numSamples_meta <= numSamples;
    numSamples_axis <= numSamples_meta;
    enable_meta <= enable;
    enable_axis <= enable_meta;

    if (!enable_axis) begin
      axis_tvalid   <= 1'b0;
      portA_en      <= 1'b0;
      portAcpu_addr <= 32'd0;
      loopVectorCount <= sample_count_to_vectors(numSamples_axis);
      vectorsRemaining <= sample_count_to_vectors(numSamples_axis);
    end else begin
      // streaming enabled
      portA_en    <= 1'b1;
      axis_tvalid <= 1'b1;

      // drive data from BRAM; BRAM output stays stable if addr doesn't change
      axis_tdata  <= portA_cpu_rdata;

      if (advance) begin
        if (USE_VECTOR_COUNT) begin
          if (vectorsRemaining <= 1) begin
            portAcpu_addr <= 32'd0;
            vectorsRemaining <= loopVectorCount;
          end else begin
            portAcpu_addr <= portAcpu_addr + WORD_BYTES;
            vectorsRemaining <= vectorsRemaining - 1;
          end
        end else begin
          if (portAcpu_addr >= LAST_ADDR)
            portAcpu_addr <= 32'd0;
          else
            portAcpu_addr <= portAcpu_addr + WORD_BYTES;
        end
      end
    end
  end
end
endmodule
