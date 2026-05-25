# -------------------------------------------------------------------------------------------------
# Copyright (C) 2023 Advanced Micro Devices, Inc
# SPDX-License-Identifier: MIT
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- --
# -------------- -------------- -------------- -------------- -------------- -------------- -------
# Pin Assignments -- all other pins handled by board BSP
# -------------- -------------- -------------- -------------- -------------- -------------- -------
set_property PACKAGE_PIN AP18 [get_ports {PL_SYSREF[0]}]
set_property PACKAGE_PIN AN11 [get_ports {PL_CLK[0]}]

set_property IOSTANDARD LVCMOS18 [get_ports {PL_CLK[*]}]
set_property IOSTANDARD LVCMOS18 [get_ports {PL_SYSREF[*]}]

# -------------- -------------- -------------- -------------- -------------- -------------- -------
# Synthesis Guidance
# -------------- -------------- -------------- -------------- -------------- -------------- -------
set_property BLOCK_SYNTH.RETIMING 1 [get_cells {bd_i/usp_rf_data_converter_1/*}]
set_property BLOCK_SYNTH.STRATEGY {PERFORMANCE_OPTIMIZED} [get_cells {bd_i/usp_rf_data_converter_1/*}]
# -------------- -------------- -------------- -------------- -------------- -------------- -------
# Timing Constraints
# -------------- -------------- -------------- -------------- -------------- -------------- -------
# PL_CLK = 491.52 MHz
#create_clock -period 2.034505208 -name PL_CLK [get_ports {PL_CLK[0]}]
## Input Delay for PL_SYSREF to ensure MTS requirements via PG269.
#set_input_delay -clock [get_clocks PL_CLK] -min -add_delay 2.034505208 [get_ports {PL_SYSREF[0]}]
#set_input_delay -clock [get_clocks PL_CLK] -max -add_delay 2.065505208 [get_ports {PL_SYSREF[0]}]

# 122.88 MHz PL clock
# Period = 1 / 122.88e6 = 8.138020833 ns
create_clock -period 8.138020833 -name PL_CLK [get_ports {PL_CLK[0]}]

# Input Delay for PL_SYSREF to ensure MTS requirements via PG269.
set_input_delay -clock [get_clocks PL_CLK] -min -add_delay 8.138020833 [get_ports {PL_SYSREF[0]}]
set_input_delay -clock [get_clocks PL_CLK] -max -add_delay 8.262020833 [get_ports {PL_SYSREF[0]}]

set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets bd_i/clocktree/BUFG_PL_CLK/U0/BUFG_O[0]]

set_false_path -from [get_ports reset]
set_false_path -from [get_pins -quiet {bd_i/gpio_control/axi_gpio_dac/U0/gpio_core_1/Not_Dual.gpio_Data_Out_reg[*]/C}]
set_false_path -from [get_pins -quiet {bd_i/gpio_control/axi_gpio_dac/U0/gpio_core_1/Dual.gpio_Data_Out_reg[*]/C}]
set_false_path -from [get_pins -quiet {bd_i/gpio_control/axi_gpio_dac/U0/gpio_core_1/Dual.gpio2_Data_Out_reg[*]/C}]
set_false_path -from [get_pins -quiet {bd_i/gpio_control/axi_gpio_dac2/U0/gpio_core_1/Not_Dual.gpio_Data_Out_reg[*]/C}]
set_false_path -from [get_pins {bd_i/clocktree/RFegressReset/U0/ACTIVE_LOW_PR_OUT_DFF[*].*/C}]

# Constrain the user_sysref clocks
set_max_delay -from [get_pins {bd_i/clocktree/synchronizeSYSREF/inst/xsingle/syncstages_ff_reg[1]/C}] 1.017252604
set_max_delay -from [get_pins {bd_i/clocktree/synchronizeSYSREF/inst/xsingle/src_ff_reg/C}] 1.017252604

# -------------- -------------- -------------- -------------- -------------- -------------- -------
# Debug / Chipscope
# -------------- -------------- -------------- -------------- -------------- -------------- -------
set_property C_CLK_INPUT_FREQ_HZ 300000000 [get_debug_cores dbg_hub]
set_property C_ENABLE_CLK_DIVIDER false [get_debug_cores dbg_hub]
set_property C_USER_SCAN_CHAIN 1 [get_debug_cores dbg_hub]
connect_debug_port dbg_hub/clk [get_nets clk]

# -------------- -------------- -------------- -------------- -------------- -------------- -------
# Bitstream Generation
# -------------- -------------- -------------- -------------- -------------- -------------- -------
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
set_property BITSTREAM.CONFIG.UNUSEDPIN PULLNONE [current_design]
set_property BITSTREAM.CONFIG.OVERTEMPSHUTDOWN ENABLE [current_design]
set_property BITSTREAM.CONFIG.USR_ACCESS TIMESTAMP [current_design]
