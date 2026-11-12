# -------------------------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
#
# add_adc_capture - enables the four ADCs of the RFSoC 4x2 and adds a triggered capture memory.
#
# Run on the open block design after add_sequencer.tcl (create_project does this when
# RFSOC_AWG_ADC=1; it needs RFSOC_AWG_SEQ=1 for the playback-start trigger).
#
# STATUS: written against the block design sources, NOT yet run through Vivado. The ADC tile
# settings (tiles 0 and 2, 4915.2 MS/s from a 491.52 MHz reference with the tile PLL) follow the
# RFSoC4x2 base overlay of Xilinx/RFSoC-PYNQ. The data-path settings (real samples, no
# decimation, mixer bypassed, 8 samples per 614.4 MHz word) are the ADC counterparts of this
# project's DAC settings: check them in the IP dialog on the first run.
#
# Data path per ADC:
#   RFDC mXX_axis (8 samples, 614.4 MHz) -> width 128 -> 256 -> clock 614.4 -> 307.2 MHz
#   -> ADCCapture input N -> capture memory (16384 words of 16 samples, 16 UltraRAM)
# -------------------------------------------------------------------------------------------------

proc add_adc_capture {} {
  current_bd_instance /

  set rfdc      [get_bd_cells usp_rf_data_converter_1]
  set clk_fast  [get_bd_pins clocktree/clkRF]
  set clk_word  [get_bd_pins clocktree/clkRFdiv2]
  set aresetn   [get_bd_pins clocktree/egress_aresetn]

  # ------------------------------------------------------------------ ADC tiles 0 and 2
  set adc_cfg [list \
    CONFIG.ADC0_Sampling_Rate {4.9152} CONFIG.ADC0_Refclk_Freq {491.520} \
    CONFIG.ADC0_PLL_Enable {true}      CONFIG.ADC0_Outclk_Freq {614.400} \
    CONFIG.ADC2_Sampling_Rate {4.9152} CONFIG.ADC2_Refclk_Freq {491.520} \
    CONFIG.ADC2_PLL_Enable {true}      CONFIG.ADC2_Outclk_Freq {614.400} \
  ]
  foreach slice {00 02 20 22} {
    lappend adc_cfg \
      CONFIG.ADC_Slice${slice}_Enable {true} \
      CONFIG.ADC_Data_Type${slice} {0} \
      CONFIG.ADC_Data_Width${slice} {8} \
      CONFIG.ADC_Decimation_Mode${slice} {1} \
      CONFIG.ADC_Mixer_Type${slice} {1} \
      CONFIG.ADC_Coarse_Mixer_Freq${slice} {3} \
      CONFIG.ADC_Mixer_Mode${slice} {2}
  }
  set_property -dict $adc_cfg $rfdc

  foreach tile {0 2} {
    set port [create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 adc${tile}_clk]
    set_property CONFIG.FREQ_HZ {491520000} $port
    connect_bd_intf_net $port [get_bd_intf_pins usp_rf_data_converter_1/adc${tile}_clk]
    connect_bd_net $clk_fast [get_bd_pins usp_rf_data_converter_1/m${tile}_axis_aclk]
    connect_bd_net $aresetn  [get_bd_pins usp_rf_data_converter_1/m${tile}_axis_aresetn]
  }
  foreach vin {vin0_01 vin0_23 vin2_01 vin2_23} {
    set port [create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 $vin]
    connect_bd_intf_net $port [get_bd_intf_pins usp_rf_data_converter_1/$vin]
  }

  # ------------------------------------------------------------------ capture block and memory
  set capture [create_bd_cell -type module -reference ADCCapture adc_capture_0]
  set_property -dict [list CONFIG.DWIDTH {256} CONFIG.MEM_SIZE_BYTES {524288}] $capture

  set bram_ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 adc_bram_ctrl]
  set_property -dict [list CONFIG.DATA_WIDTH {256} CONFIG.READ_LATENCY {3} CONFIG.SINGLE_PORT_BRAM {1}] $bram_ctrl

  set mem [create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 adc_blk_mem]
  set_property -dict [list \
    CONFIG.Memory_Type {True_Dual_Port_RAM} \
    CONFIG.PRIM_type_to_Implement {URAM} \
    CONFIG.READ_LATENCY_A {3} \
    CONFIG.READ_LATENCY_B {3} \
    CONFIG.Use_RSTB_Pin {true} \
  ] $mem

  connect_bd_intf_net [get_bd_intf_pins adc_bram_ctrl/BRAM_PORTA] [get_bd_intf_pins adc_blk_mem/BRAM_PORTA]
  connect_bd_intf_net [get_bd_intf_pins adc_capture_0/BRAM_A]     [get_bd_intf_pins adc_blk_mem/BRAM_PORTB]
  connect_bd_net $clk_word [get_bd_pins adc_capture_0/axis_clk] [get_bd_pins adc_bram_ctrl/s_axi_aclk]
  connect_bd_net $aresetn  [get_bd_pins adc_capture_0/axis_aresetn] [get_bd_pins adc_bram_ctrl/s_axi_aresetn]

  # ------------------------------------------------------------------ four ADC streams
  set input 0
  foreach slice {00 02 20 22} {
    set dwc [create_bd_cell -type ip -vlnv xilinx.com:ip:axis_dwidth_converter:1.1 adc_dwc_$input]
    set_property -dict [list CONFIG.S_TDATA_NUM_BYTES {16} CONFIG.M_TDATA_NUM_BYTES {32}] $dwc
    set cc [create_bd_cell -type ip -vlnv xilinx.com:ip:axis_clock_converter:1.1 adc_cc_$input]
    set_property -dict [list CONFIG.IS_ACLK_ASYNC {1} CONFIG.TDATA_NUM_BYTES {32}] $cc

    connect_bd_intf_net [get_bd_intf_pins usp_rf_data_converter_1/m${slice}_axis] [get_bd_intf_pins adc_dwc_$input/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins adc_dwc_$input/M_AXIS] [get_bd_intf_pins adc_cc_$input/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins adc_cc_$input/M_AXIS]  [get_bd_intf_pins adc_capture_0/s${input}_axis]
    connect_bd_net $clk_fast [get_bd_pins adc_dwc_$input/aclk] [get_bd_pins adc_cc_$input/s_axis_aclk]
    connect_bd_net $clk_word [get_bd_pins adc_cc_$input/m_axis_aclk]
    connect_bd_net $aresetn  [get_bd_pins adc_dwc_$input/aresetn] \
      [get_bd_pins adc_cc_$input/s_axis_aresetn] [get_bd_pins adc_cc_$input/m_axis_aresetn]
    incr input
  }

  # ------------------------------------------------------------------ triggers
  connect_bd_net [awg_ext_trigger_port] [get_bd_pins adc_capture_0/ext_trigger]
  connect_bd_net [get_bd_pins hier_dac_play/seq_status]  [get_bd_pins adc_capture_0/dac0_status]
  connect_bd_net [get_bd_pins hier_dac2_play/seq_status] [get_bd_pins adc_capture_0/dac2_status]

  # The analog trigger (inside ADCCapture) also starts the DAC players: their trigger input
  # becomes "PMOD pin OR analog trigger". The analog trigger is silent unless it is enabled.
  set trig_or [create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:2.0 trigger_or]
  set_property -dict [list CONFIG.C_OPERATION {or} CONFIG.C_SIZE {1}] $trig_or
  set pin_net [get_bd_nets -of_objects [awg_ext_trigger_port]]
  set player_pins [list [get_bd_pins hier_dac_play/ext_trigger] [get_bd_pins hier_dac2_play/ext_trigger]]
  disconnect_bd_net $pin_net $player_pins
  connect_bd_net [awg_ext_trigger_port] [get_bd_pins trigger_or/Op1]
  connect_bd_net [get_bd_pins adc_capture_0/analog_trig] [get_bd_pins trigger_or/Op2]
  connect_bd_net [get_bd_pins trigger_or/Res] \
    [get_bd_pins hier_dac_play/ext_trigger] [get_bd_pins hier_dac2_play/ext_trigger]

  # analog trigger registers: ctl = control (out) / status (in), cfg = thresholds / release
  set tctl [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_adc_trig_ctl]
  set_property -dict [list CONFIG.C_IS_DUAL {1} CONFIG.C_ALL_OUTPUTS {1} CONFIG.C_GPIO_WIDTH {32} \
    CONFIG.C_ALL_INPUTS_2 {1} CONFIG.C_GPIO2_WIDTH {32}] $tctl
  set tcfg [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_adc_trig_cfg]
  set_property -dict [list CONFIG.C_IS_DUAL {1} CONFIG.C_ALL_OUTPUTS {1} CONFIG.C_GPIO_WIDTH {32} \
    CONFIG.C_ALL_OUTPUTS_2 {1} CONFIG.C_GPIO2_WIDTH {32}] $tcfg
  connect_bd_net [get_bd_pins axi_gpio_adc_trig_ctl/gpio_io_o]  [get_bd_pins adc_capture_0/trig_ctrl]
  connect_bd_net [get_bd_pins adc_capture_0/trig_status]        [get_bd_pins axi_gpio_adc_trig_ctl/gpio2_io_i]
  connect_bd_net [get_bd_pins axi_gpio_adc_trig_cfg/gpio_io_o]  [get_bd_pins adc_capture_0/trig_thr]
  connect_bd_net [get_bd_pins axi_gpio_adc_trig_cfg/gpio2_io_o] [get_bd_pins adc_capture_0/trig_release]

  # ------------------------------------------------------------------ control and status GPIOs
  # ctl: channel 1 = control word (out), channel 2 = status word (in)
  set ctl [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_adc_ctl]
  set_property -dict [list CONFIG.C_IS_DUAL {1} CONFIG.C_ALL_OUTPUTS {1} CONFIG.C_GPIO_WIDTH {32} \
    CONFIG.C_ALL_INPUTS_2 {1} CONFIG.C_GPIO2_WIDTH {32}] $ctl
  # cfg: channel 1 = delay in words, channel 2 = words to record
  set cfg [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_adc_cfg]
  set_property -dict [list CONFIG.C_IS_DUAL {1} CONFIG.C_ALL_OUTPUTS {1} CONFIG.C_GPIO_WIDTH {32} \
    CONFIG.C_ALL_OUTPUTS_2 {1} CONFIG.C_GPIO2_WIDTH {32}] $cfg
  connect_bd_net [get_bd_pins axi_gpio_adc_ctl/gpio_io_o]  [get_bd_pins adc_capture_0/ctrl]
  connect_bd_net [get_bd_pins adc_capture_0/status]        [get_bd_pins axi_gpio_adc_ctl/gpio2_io_i]
  connect_bd_net [get_bd_pins axi_gpio_adc_cfg/gpio_io_o]  [get_bd_pins adc_capture_0/delay]
  connect_bd_net [get_bd_pins axi_gpio_adc_cfg/gpio2_io_o] [get_bd_pins adc_capture_0/nrows]
  awg_add_control_slaves {axi_gpio_adc_ctl axi_gpio_adc_cfg axi_gpio_adc_trig_ctl axi_gpio_adc_trig_cfg}

  # ------------------------------------------------------------------ processor access to the memory
  set ic [get_bd_cells internalRAM_interconnect]
  set index [get_property CONFIG.NUM_MI $ic]
  set_property CONFIG.NUM_MI [expr {$index + 1}] $ic
  set port [format "M%02d" $index]
  connect_bd_intf_net [get_bd_intf_pins internalRAM_interconnect/${port}_AXI] [get_bd_intf_pins adc_bram_ctrl/S_AXI]
  connect_bd_net $clk_word [get_bd_pins internalRAM_interconnect/${port}_ACLK]
  connect_bd_net $aresetn  [get_bd_pins internalRAM_interconnect/${port}_ARESETN]

  assign_bd_address -offset 0xB0200000 -range 0x00080000 \
    -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] \
    [get_bd_addr_segs adc_bram_ctrl/S_AXI/Mem0] -force
  assign_bd_address

  puts "ADC capture added to the block design"
}
