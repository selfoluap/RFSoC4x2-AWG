# SPDX-License-Identifier: MIT
# Used when the sequencer (RFSOC_AWG_SEQ=1) or the DDR4 player (RFSOC_AWG_DDR4=1) is built.
# -------------------------------------------------------------------------------------------------
# External trigger input: PMOD0 pin 0 (pin location from the same base overlay constraints).
# The FPGA pin is LVCMOS18. Check the board manual for the voltage allowed at the connector
# before wiring a trigger source to it.
# -------------------------------------------------------------------------------------------------
set_property PACKAGE_PIN AF16      [get_ports ext_trigger]
set_property IOSTANDARD LVCMOS18   [get_ports ext_trigger]
set_property PULLDOWN true         [get_ports ext_trigger]
set_false_path -from [get_ports ext_trigger]
