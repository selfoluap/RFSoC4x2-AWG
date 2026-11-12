# -------------------------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
#
# add_ddr4_player - extends the dual-DAC block design with the PL DDR4 long-waveform player.
#
# Run on the open block design after bd_bd.tcl and, if used, add_sequencer.tcl (create_project
# does this when RFSOC_AWG_DDR4=1). Uses the helper procs of add_sequencer.tcl.
#
# STATUS: written against the block design sources, NOT yet run through Vivado. The DDR4
# controller settings and its clock/reset wiring are the ones of the RFSoC4x2 base overlay
# (Xilinx/RFSoC-PYNQ, boards/RFSoC4x2/base/base.tcl). Expect to fix small things on the first
# run; DDR4_PLAYER.md lists what to check.
#
# What it adds
#   ddr4_0                 DDR4 controller (MIG), 512-bit AXI, ui_clk = 300 MHz
#   ddr4_smc               SmartConnect: PS (M_AXI_HPM0_FPD, loads waveforms) + 2 players -> ddr4_0
#   <hier>/DDR4Player_0    one per DAC, inserted between DACRAMstreamer_0 and axis_clock_converter_0
#   axi_gpio_ddr{0,2}_ctl  control word (out) and status word (in) per DAC
#   axi_gpio_ddr{0,2}_cfg  base address and length per DAC
# -------------------------------------------------------------------------------------------------

proc ddr4_insert_player { hier gpio_prefix } {
  set player [create_bd_cell -type module -reference DDR4Player $hier/DDR4Player_0]

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 $hier/M_AXI_DDR4
  create_bd_pin -dir I -type clk $hier/ddr4_ui_clk
  create_bd_pin -dir I -type rst $hier/ddr4_aresetn
  create_bd_pin -dir I -from 31 -to 0 $hier/ddr4_ctrl
  create_bd_pin -dir I -from 20 -to 0 $hier/ddr4_base_page
  create_bd_pin -dir I -from 27 -to 0 $hier/ddr4_num_words
  if {[llength [get_bd_pins -quiet $hier/ext_trigger]] == 0} {
    create_bd_pin -dir I $hier/ext_trigger
  }
  create_bd_pin -dir O -from 31 -to 0 $hier/ddr4_status

  # put the player's source mux between the BRAM streamer and the clock converter
  delete_bd_objs [get_bd_intf_nets $hier/DACRAMstreamer_0_axis]
  connect_bd_intf_net [get_bd_intf_pins $hier/DACRAMstreamer_0/axis]  [get_bd_intf_pins $hier/DDR4Player_0/s_axis]
  connect_bd_intf_net [get_bd_intf_pins $hier/DDR4Player_0/m_axis]    [get_bd_intf_pins $hier/axis_clock_converter_0/S_AXIS]
  connect_bd_intf_net [get_bd_intf_pins $hier/DDR4Player_0/m_axi]     [get_bd_intf_pins $hier/M_AXI_DDR4]

  connect_bd_net [get_bd_pins $hier/axis_clk]       [get_bd_pins $hier/DDR4Player_0/axis_clk]
  connect_bd_net [get_bd_pins $hier/s_axi_aresetn]  [get_bd_pins $hier/DDR4Player_0/axis_aresetn]
  connect_bd_net [get_bd_pins $hier/ddr4_ui_clk]    [get_bd_pins $hier/DDR4Player_0/m_axi_aclk]
  connect_bd_net [get_bd_pins $hier/ddr4_aresetn]   [get_bd_pins $hier/DDR4Player_0/m_axi_aresetn]
  connect_bd_net [get_bd_pins $hier/ddr4_ctrl]      [get_bd_pins $hier/DDR4Player_0/ctrl]
  connect_bd_net [get_bd_pins $hier/ddr4_base_page] [get_bd_pins $hier/DDR4Player_0/base_page]
  connect_bd_net [get_bd_pins $hier/ddr4_num_words] [get_bd_pins $hier/DDR4Player_0/num_words]
  connect_bd_net [get_bd_pins $hier/ext_trigger]    [get_bd_pins $hier/DDR4Player_0/ext_trigger]
  connect_bd_net [get_bd_pins $hier/DDR4Player_0/status] [get_bd_pins $hier/ddr4_status]

  # control/status GPIO: channel 1 = control word (out), channel 2 = status word (in)
  set ctl [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 ${gpio_prefix}_ctl]
  set_property -dict [list \
    CONFIG.C_IS_DUAL {1} \
    CONFIG.C_ALL_OUTPUTS {1} \
    CONFIG.C_GPIO_WIDTH {32} \
    CONFIG.C_ALL_INPUTS_2 {1} \
    CONFIG.C_GPIO2_WIDTH {32} \
  ] $ctl

  # configuration GPIO: channel 1 = base address / 4096, channel 2 = length in 64-byte words
  set cfg [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 ${gpio_prefix}_cfg]
  set_property -dict [list \
    CONFIG.C_IS_DUAL {1} \
    CONFIG.C_ALL_OUTPUTS {1} \
    CONFIG.C_GPIO_WIDTH {21} \
    CONFIG.C_ALL_OUTPUTS_2 {1} \
    CONFIG.C_GPIO2_WIDTH {28} \
  ] $cfg

  connect_bd_net [get_bd_pins ${gpio_prefix}_ctl/gpio_io_o]  [get_bd_pins $hier/ddr4_ctrl]
  connect_bd_net [get_bd_pins $hier/ddr4_status] [get_bd_pins ${gpio_prefix}_ctl/gpio2_io_i]
  connect_bd_net [get_bd_pins ${gpio_prefix}_cfg/gpio_io_o]  [get_bd_pins $hier/ddr4_base_page]
  connect_bd_net [get_bd_pins ${gpio_prefix}_cfg/gpio2_io_o] [get_bd_pins $hier/ddr4_num_words]
}

proc add_ddr4_player {} {
  current_bd_instance /

  # ------------------------------------------------------------------ external ports
  set ddr4_pl [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddr4_rtl:1.0 ddr4_pl]
  set sys_clk_ddr4 [create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 sys_clk_ddr4]
  set_property CONFIG.FREQ_HZ {200000000} $sys_clk_ddr4

  # ------------------------------------------------------------------ DDR4 controller
  # Settings as in the RFSoC4x2 base overlay: DDR4-2400 (833 ps), 64-bit, 512-bit AXI.
  set ddr4_0 [create_bd_cell -type ip -vlnv xilinx.com:ip:ddr4:2.2 ddr4_0]
  set_property -dict [list \
    CONFIG.C0.BANK_GROUP_WIDTH {1} \
    CONFIG.C0.DDR4_AxiAddressWidth {33} \
    CONFIG.C0.DDR4_AxiDataWidth {512} \
    CONFIG.C0.DDR4_CasLatency {17} \
    CONFIG.C0.DDR4_CasWriteLatency {12} \
    CONFIG.C0.DDR4_DataWidth {64} \
    CONFIG.C0.DDR4_InputClockPeriod {4998} \
    CONFIG.C0.DDR4_MemoryPart {MT40A1G16RC-062E} \
    CONFIG.C0.DDR4_MemoryType {Components} \
    CONFIG.C0.DDR4_TimePeriod {833} \
    CONFIG.System_Clock {No_Buffer} \
  ] $ddr4_0

  set ddr4_clk_ibufds [create_bd_cell -type ip -vlnv xilinx.com:ip:util_ds_buf:2.2 ddr4_clk_ibufds]
  set_property CONFIG.C_BUF_TYPE {IBUFDS} $ddr4_clk_ibufds
  set ddr4_clk_bufg [create_bd_cell -type ip -vlnv xilinx.com:ip:util_ds_buf:2.2 ddr4_clk_bufg]
  set_property CONFIG.C_BUF_TYPE {BUFG} $ddr4_clk_bufg

  set ddr4_rst_inv [create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:2.0 ddr4_rst_inv]
  set_property -dict [list CONFIG.C_OPERATION {not} CONFIG.C_SIZE {1}] $ddr4_rst_inv

  set ddr4_sys_reset [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 ddr4_sys_reset]

  connect_bd_intf_net [get_bd_intf_ports sys_clk_ddr4] [get_bd_intf_pins ddr4_clk_ibufds/CLK_IN_D]
  connect_bd_net [get_bd_pins ddr4_clk_ibufds/IBUF_OUT] [get_bd_pins ddr4_clk_bufg/BUFG_I]
  connect_bd_net [get_bd_pins ddr4_clk_bufg/BUFG_O]     [get_bd_pins ddr4_0/c0_sys_clk_i]
  connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins ddr4_rst_inv/Op1]
  connect_bd_net [get_bd_pins ddr4_rst_inv/Res]         [get_bd_pins ddr4_0/sys_rst]
  connect_bd_intf_net [get_bd_intf_pins ddr4_0/C0_DDR4] [get_bd_intf_ports ddr4_pl]

  connect_bd_net -net ddr4_ui_clk [get_bd_pins ddr4_0/c0_ddr4_ui_clk] \
    [get_bd_pins ddr4_sys_reset/slowest_sync_clk]
  connect_bd_net [get_bd_pins ddr4_0/c0_ddr4_ui_clk_sync_rst] [get_bd_pins ddr4_sys_reset/ext_reset_in]
  connect_bd_net -net ddr4_aresetn [get_bd_pins ddr4_sys_reset/peripheral_aresetn] \
    [get_bd_pins ddr4_0/c0_ddr4_aresetn]

  # ------------------------------------------------------------------ AXI path to the DDR4
  set ddr4_smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 ddr4_smc]
  set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {1} CONFIG.NUM_CLKS {2}] $ddr4_smc

  connect_bd_net -net ddr4_ui_clk  [get_bd_pins ddr4_smc/aclk]
  connect_bd_net -net ddr4_aresetn [get_bd_pins ddr4_smc/aresetn]
  connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins ddr4_smc/aclk1]
  connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] [get_bd_intf_pins ddr4_smc/S00_AXI]
  connect_bd_intf_net [get_bd_intf_pins ddr4_smc/M00_AXI] [get_bd_intf_pins ddr4_0/C0_DDR4_S_AXI]

  # ------------------------------------------------------------------ players and their GPIOs
  ddr4_insert_player hier_dac_play  axi_gpio_ddr0
  ddr4_insert_player hier_dac2_play axi_gpio_ddr2

  connect_bd_intf_net [get_bd_intf_pins hier_dac_play/M_AXI_DDR4]  [get_bd_intf_pins ddr4_smc/S01_AXI]
  connect_bd_intf_net [get_bd_intf_pins hier_dac2_play/M_AXI_DDR4] [get_bd_intf_pins ddr4_smc/S02_AXI]
  connect_bd_net -net ddr4_ui_clk  [get_bd_pins hier_dac_play/ddr4_ui_clk]  [get_bd_pins hier_dac2_play/ddr4_ui_clk]
  connect_bd_net -net ddr4_aresetn [get_bd_pins hier_dac_play/ddr4_aresetn] [get_bd_pins hier_dac2_play/ddr4_aresetn]
  connect_bd_net [awg_ext_trigger_port] [get_bd_pins hier_dac_play/ext_trigger] [get_bd_pins hier_dac2_play/ext_trigger]

  # four more AXI-Lite slaves on the next free ports of the control interconnect
  awg_add_control_slaves {axi_gpio_ddr0_ctl axi_gpio_ddr0_cfg axi_gpio_ddr2_ctl axi_gpio_ddr2_cfg}

  # ------------------------------------------------------------------ addresses
  # PS: 8 GiB window at 0x10_0000_0000, as in the base overlay. Players: DDR4-relative from 0,
  # so base_page written by the driver is simply (offset inside the DDR4) / 4096.
  set ddr4_seg [get_bd_addr_segs ddr4_0/C0_DDR4_MEMORY_MAP/C0_DDR4_ADDRESS_BLOCK]
  assign_bd_address -offset 0x001000000000 -range 0x000200000000 \
    -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] $ddr4_seg -force
  assign_bd_address -offset 0x000000000000 -range 0x000200000000 \
    -target_address_space [get_bd_addr_spaces hier_dac_play/DDR4Player_0/m_axi] $ddr4_seg -force
  assign_bd_address -offset 0x000000000000 -range 0x000200000000 \
    -target_address_space [get_bd_addr_spaces hier_dac2_play/DDR4Player_0/m_axi] $ddr4_seg -force
  assign_bd_address

  puts "DDR4 player added to the block design"
}
