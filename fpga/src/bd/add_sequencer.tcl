# -------------------------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
#
# add_sequencer - replaces DACRAMstreamer by DACSeqStreamer (sequence mode) in both DAC players
# and doubles the waveform memory to 8192 rows (262144 samples) per channel.
#
# Run on the open block design after bd_bd.tcl and before add_ddr4_player.tcl
# (create_project does this when RFSOC_AWG_SEQ=1).
#
# STATUS: written against the block design sources, NOT yet run through Vivado.
#
# The new cell keeps the name DACRAMstreamer_0, so the legacy `enable` / `sample_count` wiring,
# the existing Python DacPlayer and the DDR4 add-on all keep working.
# -------------------------------------------------------------------------------------------------

proc awg_ext_trigger_port {} {
  if {[llength [get_bd_ports -quiet ext_trigger]] == 0} {
    create_bd_port -dir I ext_trigger
  }
  return [get_bd_ports ext_trigger]
}

# Adds AXI-Lite slaves to the control interconnect on its next free master ports.
proc awg_add_control_slaves { cells } {
  set ic [get_bd_cells control_interconnect]
  set index [get_property CONFIG.NUM_MI $ic]
  set_property CONFIG.NUM_MI [expr {$index + [llength $cells]}] $ic
  foreach cell $cells {
    set port [format "M%02d" $index]
    connect_bd_intf_net [get_bd_intf_pins control_interconnect/${port}_AXI] [get_bd_intf_pins $cell/S_AXI]
    connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] \
      [get_bd_pins control_interconnect/${port}_ACLK] [get_bd_pins $cell/s_axi_aclk]
    connect_bd_net [get_bd_pins clocktree/s_axi_aresetn] \
      [get_bd_pins control_interconnect/${port}_ARESETN] [get_bd_pins $cell/s_axi_aresetn]
    incr index
  }
}

proc seq_replace_streamer { hier gpio_prefix } {
  delete_bd_objs [get_bd_intf_nets -quiet $hier/DACRAMstreamer_0_axis] \
                 [get_bd_intf_nets -quiet $hier/DACRAMstreamer_0_BRAM_A]
  delete_bd_objs [get_bd_cells $hier/DACRAMstreamer_0]

  set streamer [create_bd_cell -type module -reference DACSeqStreamer $hier/DACRAMstreamer_0]
  set_property -dict [list CONFIG.DWIDTH {512} CONFIG.MEM_SIZE_BYTES {524288} CONFIG.READ_LATENCY {3}] $streamer

  connect_bd_intf_net -intf_net DACRAMstreamer_0_BRAM_A \
    [get_bd_intf_pins $hier/DACRAMstreamer_0/BRAM_A] [get_bd_intf_pins $hier/blk_mem_gen_0/BRAM_PORTB]
  connect_bd_intf_net -intf_net DACRAMstreamer_0_axis \
    [get_bd_intf_pins $hier/DACRAMstreamer_0/axis] [get_bd_intf_pins $hier/axis_clock_converter_0/S_AXIS]
  connect_bd_net [get_bd_pins $hier/axis_clk]      [get_bd_pins $hier/DACRAMstreamer_0/axis_clk]
  connect_bd_net [get_bd_pins $hier/s_axi_aresetn] [get_bd_pins $hier/DACRAMstreamer_0/axis_aresetn]
  connect_bd_net [get_bd_pins $hier/enable]        [get_bd_pins $hier/DACRAMstreamer_0/enable]
  connect_bd_net [get_bd_pins $hier/sample_count]  [get_bd_pins $hier/DACRAMstreamer_0/numSamples]

  create_bd_pin -dir I -from 31 -to 0 $hier/seq_ctrl
  create_bd_pin -dir I -from 31 -to 0 $hier/seq_lens
  create_bd_pin -dir I -from 31 -to 0 $hier/seq_repeats
  create_bd_pin -dir O -from 31 -to 0 $hier/seq_status
  if {[llength [get_bd_pins -quiet $hier/ext_trigger]] == 0} {
    create_bd_pin -dir I $hier/ext_trigger
  }
  connect_bd_net [get_bd_pins $hier/seq_ctrl]    [get_bd_pins $hier/DACRAMstreamer_0/seq_ctrl]
  connect_bd_net [get_bd_pins $hier/seq_lens]    [get_bd_pins $hier/DACRAMstreamer_0/seq_lens]
  connect_bd_net [get_bd_pins $hier/seq_repeats] [get_bd_pins $hier/DACRAMstreamer_0/seq_repeats]
  connect_bd_net [get_bd_pins $hier/ext_trigger] [get_bd_pins $hier/DACRAMstreamer_0/ext_trigger]
  connect_bd_net [get_bd_pins $hier/DACRAMstreamer_0/seq_status] [get_bd_pins $hier/seq_status]

  # channel 1 = control word (out), channel 2 = status word (in)
  set ctl [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 ${gpio_prefix}_ctl]
  set_property -dict [list CONFIG.C_IS_DUAL {1} CONFIG.C_ALL_OUTPUTS {1} CONFIG.C_GPIO_WIDTH {32} \
    CONFIG.C_ALL_INPUTS_2 {1} CONFIG.C_GPIO2_WIDTH {32}] $ctl
  # channel 1 = {number of blocks, block length in rows}, channel 2 = repeats per block
  set cfg [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 ${gpio_prefix}_cfg]
  set_property -dict [list CONFIG.C_IS_DUAL {1} CONFIG.C_ALL_OUTPUTS {1} CONFIG.C_GPIO_WIDTH {32} \
    CONFIG.C_ALL_OUTPUTS_2 {1} CONFIG.C_GPIO2_WIDTH {32}] $cfg

  connect_bd_net [get_bd_pins ${gpio_prefix}_ctl/gpio_io_o]  [get_bd_pins $hier/seq_ctrl]
  connect_bd_net [get_bd_pins $hier/seq_status]              [get_bd_pins ${gpio_prefix}_ctl/gpio2_io_i]
  connect_bd_net [get_bd_pins ${gpio_prefix}_cfg/gpio_io_o]  [get_bd_pins $hier/seq_lens]
  connect_bd_net [get_bd_pins ${gpio_prefix}_cfg/gpio2_io_o] [get_bd_pins $hier/seq_repeats]
}

proc add_sequencer {} {
  current_bd_instance /

  seq_replace_streamer hier_dac_play  axi_gpio_seq0
  seq_replace_streamer hier_dac2_play axi_gpio_seq2

  connect_bd_net [awg_ext_trigger_port] \
    [get_bd_pins hier_dac_play/ext_trigger] [get_bd_pins hier_dac2_play/ext_trigger]

  awg_add_control_slaves {axi_gpio_seq0_ctl axi_gpio_seq0_cfg axi_gpio_seq2_ctl axi_gpio_seq2_cfg}

  # 512 KiB of waveform memory per channel (16 UltraRAM each, 32 of 80). The two windows no
  # longer fit side by side at their old addresses; software reads the addresses from the .hwh.
  set ps [get_bd_addr_spaces zynq_ultra_ps_e_0/Data]
  assign_bd_address -offset 0xB0100000 -range 0x00080000 -target_address_space $ps \
    [get_bd_addr_segs hier_dac_play/axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0xB0180000 -range 0x00080000 -target_address_space $ps \
    [get_bd_addr_segs hier_dac2_play/axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address

  puts "Sequencer added to the block design"
}
