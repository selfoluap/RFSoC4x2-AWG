# -------------------------------------------------------------------------------------------------
# Constraints for the PL DDR4 long-waveform player (only used when RFSOC_AWG_DDR4=1).
# The external trigger pin is in ext_trigger.xdc.
#
# The PL DDR4 pin assignments below are copied unchanged from the RFSoC4x2 base overlay of
# Xilinx/RFSoC-PYNQ (boards/RFSoC4x2/base/vivado/constraints/base.xdc):
#   Copyright (C) 2022 Xilinx, Inc
#   SPDX-License-Identifier: BSD-3-Clause
# -------------------------------------------------------------------------------------------------

## PL ddr4
set_property PACKAGE_PIN G13			[get_ports "sys_clk_ddr4_clk_p"]
set_property IOSTANDARD DIFF_SSTL12		[get_ports "sys_clk_ddr4_clk_p"]
create_clock -period 5.000 -name sys_clk_ddr4	[get_ports "sys_clk_ddr4_clk_p"]

set_property PACKAGE_PIN B13			[get_ports "ddr4_pl_adr[0]"]
set_property PACKAGE_PIN G6			[get_ports "ddr4_pl_adr[1]"]
set_property PACKAGE_PIN A14			[get_ports "ddr4_pl_adr[2]"]
set_property PACKAGE_PIN F10			[get_ports "ddr4_pl_adr[3]"]
set_property PACKAGE_PIN D14			[get_ports "ddr4_pl_adr[4]"]
set_property PACKAGE_PIN F11			[get_ports "ddr4_pl_adr[5]"]
set_property PACKAGE_PIN J7			[get_ports "ddr4_pl_adr[6]"]
set_property PACKAGE_PIN H13			[get_ports "ddr4_pl_adr[7]"]
set_property PACKAGE_PIN A11			[get_ports "ddr4_pl_adr[8]"]
set_property PACKAGE_PIN H6			[get_ports "ddr4_pl_adr[9]"]
set_property PACKAGE_PIN C15			[get_ports "ddr4_pl_adr[10]"]
set_property PACKAGE_PIN G7			[get_ports "ddr4_pl_adr[11]"]
set_property PACKAGE_PIN D13			[get_ports "ddr4_pl_adr[12]"]
set_property PACKAGE_PIN H11			[get_ports "ddr4_pl_adr[13]"]
set_property PACKAGE_PIN K13			[get_ports "ddr4_pl_adr[14]"]
set_property PACKAGE_PIN F14			[get_ports "ddr4_pl_adr[15]"]
set_property PACKAGE_PIN E13			[get_ports "ddr4_pl_adr[16]"]
set_property IOSTANDARD SSTL12_DCI		[get_ports {ddr4_pl_adr[?]}]

set_property PACKAGE_PIN B14			[get_ports {ddr4_pl_act_n}]
set_property IOSTANDARD SSTL12_DCI		[get_ports {ddr4_pl_act_n}]

set_property PACKAGE_PIN A12			[get_ports "ddr4_pl_ba[0]"]
set_property PACKAGE_PIN H10			[get_ports "ddr4_pl_ba[1]"]
set_property IOSTANDARD SSTL12_DCI		[get_ports {ddr4_pl_ba[?]}]
set_property PACKAGE_PIN H12			[get_ports "ddr4_pl_bg[0]"]
set_property IOSTANDARD SSTL12_DCI		[get_ports "ddr4_pl_bg[0]"]

set_property PACKAGE_PIN J10			[get_ports "ddr4_pl_ck_c[0]"]
set_property IOSTANDARD DIFF_SSTL12_DCI		[get_ports "ddr4_pl_ck_c[0]"]
set_property PACKAGE_PIN J11			[get_ports "ddr4_pl_ck_t[0]"]
set_property IOSTANDARD DIFF_SSTL12_DCI		[get_ports "ddr4_pl_ck_t[0]"]

set_property PACKAGE_PIN F12			[get_ports "ddr4_pl_cke[0]"]
set_property IOSTANDARD SSTL12_DCI		[get_ports "ddr4_pl_cke[0]"]

set_property PACKAGE_PIN E11			[get_ports "ddr4_pl_cs_n[0]"]
set_property IOSTANDARD SSTL12_DCI		[get_ports "ddr4_pl_cs_n[0]"]

set_property PACKAGE_PIN J15			[get_ports "ddr4_pl_dm_n[0]"]
set_property PACKAGE_PIN N14			[get_ports "ddr4_pl_dm_n[1]"]
set_property PACKAGE_PIN D18			[get_ports "ddr4_pl_dm_n[2]"]
set_property PACKAGE_PIN G17			[get_ports "ddr4_pl_dm_n[3]"]
set_property PACKAGE_PIN F21			[get_ports "ddr4_pl_dm_n[4]"]
set_property PACKAGE_PIN J23			[get_ports "ddr4_pl_dm_n[5]"]
set_property PACKAGE_PIN C23			[get_ports "ddr4_pl_dm_n[6]"]
set_property PACKAGE_PIN N20			[get_ports "ddr4_pl_dm_n[7]"]
set_property IOSTANDARD POD12_DCI		[get_ports {ddr4_pl_dm_n[?]}]

set_property PACKAGE_PIN K17			[get_ports "ddr4_pl_dq[0]"]
set_property PACKAGE_PIN J16			[get_ports "ddr4_pl_dq[1]"]
set_property PACKAGE_PIN H17			[get_ports "ddr4_pl_dq[2]"]
set_property PACKAGE_PIN H16			[get_ports "ddr4_pl_dq[3]"]
set_property PACKAGE_PIN J18			[get_ports "ddr4_pl_dq[4]"]
set_property PACKAGE_PIN K16			[get_ports "ddr4_pl_dq[5]"]
set_property PACKAGE_PIN J19			[get_ports "ddr4_pl_dq[6]"]
set_property PACKAGE_PIN L17			[get_ports "ddr4_pl_dq[7]"]

set_property PACKAGE_PIN N17			[get_ports "ddr4_pl_dq[8]"]
set_property PACKAGE_PIN N13			[get_ports "ddr4_pl_dq[9]"]
set_property PACKAGE_PIN N15			[get_ports "ddr4_pl_dq[10]"]
set_property PACKAGE_PIN L12			[get_ports "ddr4_pl_dq[11]"]
set_property PACKAGE_PIN M17			[get_ports "ddr4_pl_dq[12]"]
set_property PACKAGE_PIN M13			[get_ports "ddr4_pl_dq[13]"]
set_property PACKAGE_PIN M15			[get_ports "ddr4_pl_dq[14]"]
set_property PACKAGE_PIN M12			[get_ports "ddr4_pl_dq[15]"]

set_property PACKAGE_PIN D16			[get_ports "ddr4_pl_dq[16]"]
set_property PACKAGE_PIN A17			[get_ports "ddr4_pl_dq[17]"]
set_property PACKAGE_PIN C17			[get_ports "ddr4_pl_dq[18]"]
set_property PACKAGE_PIN A19			[get_ports "ddr4_pl_dq[19]"]
set_property PACKAGE_PIN D15			[get_ports "ddr4_pl_dq[20]"]
set_property PACKAGE_PIN C16			[get_ports "ddr4_pl_dq[21]"]
set_property PACKAGE_PIN B19			[get_ports "ddr4_pl_dq[22]"]
set_property PACKAGE_PIN A16			[get_ports "ddr4_pl_dq[23]"]

set_property PACKAGE_PIN G18			[get_ports "ddr4_pl_dq[24]"]
set_property PACKAGE_PIN E16			[get_ports "ddr4_pl_dq[25]"]
set_property PACKAGE_PIN F16			[get_ports "ddr4_pl_dq[26]"]
set_property PACKAGE_PIN G15			[get_ports "ddr4_pl_dq[27]"]
set_property PACKAGE_PIN H18			[get_ports "ddr4_pl_dq[28]"]
set_property PACKAGE_PIN E17			[get_ports "ddr4_pl_dq[29]"]
set_property PACKAGE_PIN E18			[get_ports "ddr4_pl_dq[30]"]
set_property PACKAGE_PIN F15			[get_ports "ddr4_pl_dq[31]"]

set_property PACKAGE_PIN E24			[get_ports "ddr4_pl_dq[32]"]
set_property PACKAGE_PIN D21			[get_ports "ddr4_pl_dq[33]"]
set_property PACKAGE_PIN E22			[get_ports "ddr4_pl_dq[34]"]
set_property PACKAGE_PIN E21			[get_ports "ddr4_pl_dq[35]"]
set_property PACKAGE_PIN E23			[get_ports "ddr4_pl_dq[36]"]
set_property PACKAGE_PIN F20			[get_ports "ddr4_pl_dq[37]"]
set_property PACKAGE_PIN F24			[get_ports "ddr4_pl_dq[38]"]
set_property PACKAGE_PIN G20			[get_ports "ddr4_pl_dq[39]"]

set_property PACKAGE_PIN J21			[get_ports "ddr4_pl_dq[40]"]
set_property PACKAGE_PIN G22			[get_ports "ddr4_pl_dq[41]"]
set_property PACKAGE_PIN K24			[get_ports "ddr4_pl_dq[42]"]
set_property PACKAGE_PIN G23			[get_ports "ddr4_pl_dq[43]"]
set_property PACKAGE_PIN L24			[get_ports "ddr4_pl_dq[44]"]
set_property PACKAGE_PIN H22			[get_ports "ddr4_pl_dq[45]"]
set_property PACKAGE_PIN H23			[get_ports "ddr4_pl_dq[46]"]
set_property PACKAGE_PIN H21			[get_ports "ddr4_pl_dq[47]"]

set_property PACKAGE_PIN C21			[get_ports "ddr4_pl_dq[48]"]
set_property PACKAGE_PIN A24			[get_ports "ddr4_pl_dq[49]"]
set_property PACKAGE_PIN B24			[get_ports "ddr4_pl_dq[50]"]
set_property PACKAGE_PIN A20			[get_ports "ddr4_pl_dq[51]"]
set_property PACKAGE_PIN C22			[get_ports "ddr4_pl_dq[52]"]
set_property PACKAGE_PIN A21			[get_ports "ddr4_pl_dq[53]"]
set_property PACKAGE_PIN C20			[get_ports "ddr4_pl_dq[54]"]
set_property PACKAGE_PIN B20			[get_ports "ddr4_pl_dq[55]"]

set_property PACKAGE_PIN M20			[get_ports "ddr4_pl_dq[56]"]
set_property PACKAGE_PIN L20			[get_ports "ddr4_pl_dq[57]"]
set_property PACKAGE_PIN L22			[get_ports "ddr4_pl_dq[58]"]
set_property PACKAGE_PIN L21			[get_ports "ddr4_pl_dq[59]"]
set_property PACKAGE_PIN N19			[get_ports "ddr4_pl_dq[60]"]
set_property PACKAGE_PIN M19			[get_ports "ddr4_pl_dq[61]"]
set_property PACKAGE_PIN L23			[get_ports "ddr4_pl_dq[62]"]
set_property PACKAGE_PIN L19			[get_ports "ddr4_pl_dq[63]"]

set_property IOSTANDARD POD12_DCI		[get_ports {ddr4_pl_dq[?]}]

set_property PACKAGE_PIN K18			[get_ports "ddr4_pl_dqs_c[0]"]
set_property PACKAGE_PIN L14			[get_ports "ddr4_pl_dqs_c[1]"]
set_property PACKAGE_PIN B17			[get_ports "ddr4_pl_dqs_c[2]"]
set_property PACKAGE_PIN F19			[get_ports "ddr4_pl_dqs_c[3]"]
set_property PACKAGE_PIN D24			[get_ports "ddr4_pl_dqs_c[4]"]
set_property PACKAGE_PIN H20			[get_ports "ddr4_pl_dqs_c[5]"]
set_property PACKAGE_PIN A22			[get_ports "ddr4_pl_dqs_c[6]"]
set_property PACKAGE_PIN K22			[get_ports "ddr4_pl_dqs_c[7]"]

set_property IOSTANDARD DIFF_POD12_DCI		[get_ports {ddr4_pl_dqs_c[?]}]

set_property PACKAGE_PIN K19			[get_ports "ddr4_pl_dqs_t[0]"]
set_property PACKAGE_PIN L15			[get_ports "ddr4_pl_dqs_t[1]"]
set_property PACKAGE_PIN B18			[get_ports "ddr4_pl_dqs_t[2]"]
set_property PACKAGE_PIN G19			[get_ports "ddr4_pl_dqs_t[3]"]
set_property PACKAGE_PIN D23			[get_ports "ddr4_pl_dqs_t[4]"]
set_property PACKAGE_PIN J20			[get_ports "ddr4_pl_dqs_t[5]"]
set_property PACKAGE_PIN B22			[get_ports "ddr4_pl_dqs_t[6]"]
set_property PACKAGE_PIN K21			[get_ports "ddr4_pl_dqs_t[7]"]

set_property IOSTANDARD DIFF_POD12_DCI		[get_ports {ddr4_pl_dqs_t[?]}]

set_property PACKAGE_PIN A15			[get_ports "ddr4_pl_odt[0]"]
set_property IOSTANDARD SSTL12_DCI		[get_ports {ddr4_pl_odt[?]}]
set_property PACKAGE_PIN E14			[get_ports "ddr4_pl_reset_n"]
set_property IOSTANDARD LVCMOS12		[get_ports {ddr4_pl_reset_n}]

# -------------------------------------------------------------------------------------------------
# Clock domain crossings of the DDR4 player
# -------------------------------------------------------------------------------------------------
# 1. Control and status words between the AXI GPIOs (PS clock) and the player. They are levels:
#    configuration is written before `enable`, status flags are sticky or slow.
set ddr4_gpio_regs   [get_cells -quiet -hier -filter {IS_SEQUENTIAL && NAME =~ "*axi_gpio_ddr*"}]
set ddr4_player_regs [get_cells -quiet -hier -filter {IS_SEQUENTIAL && NAME =~ "*DDR4Player_0/*"}]
set_false_path -from $ddr4_gpio_regs   -to $ddr4_player_regs
set_false_path -from $ddr4_player_regs -to $ddr4_gpio_regs

# 2. Short FIFO between the DDR4 controller clock (300 MHz) and the stream clock (307.2 MHz):
#    gray-coded pointers and three level flags, each into a two-flop synchroniser.
set_max_delay -datapath_only 3.0 \
    -from [get_cells -hier -filter {NAME =~ "*DDR4Player_0/*u_fifo/wgray_reg[*]"}] \
    -to   [get_cells -hier -filter {NAME =~ "*DDR4Player_0/*u_fifo/wgray_r1_reg[*]"}]
set_bus_skew 1.5 \
    -from [get_cells -hier -filter {NAME =~ "*DDR4Player_0/*u_fifo/wgray_reg[*]"}] \
    -to   [get_cells -hier -filter {NAME =~ "*DDR4Player_0/*u_fifo/wgray_r1_reg[*]"}]
set_max_delay -datapath_only 3.0 \
    -from [get_cells -hier -filter {NAME =~ "*DDR4Player_0/*u_fifo/rgray_reg[*]"}] \
    -to   [get_cells -hier -filter {NAME =~ "*DDR4Player_0/*u_fifo/rgray_w1_reg[*]"}]
set_bus_skew 1.5 \
    -from [get_cells -hier -filter {NAME =~ "*DDR4Player_0/*u_fifo/rgray_reg[*]"}] \
    -to   [get_cells -hier -filter {NAME =~ "*DDR4Player_0/*u_fifo/rgray_w1_reg[*]"}]
set_max_delay -datapath_only 3.0 \
    -from [get_cells -hier -filter {NAME =~ "*DDR4Player_0/*u_streamer/wrun_q_reg"}] \
    -to   [get_cells -hier -filter {NAME =~ "*DDR4Player_0/*u_streamer/wrun_r1_reg"}]
set_max_delay -datapath_only 3.0 \
    -from [get_cells -hier -filter {NAME =~ "*DDR4Player_0/*u_streamer/ready_q_reg"}] \
    -to   [get_cells -hier -filter {NAME =~ "*DDR4Player_0/*u_streamer/ready_r1_reg"}]
