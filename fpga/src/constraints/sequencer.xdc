# SPDX-License-Identifier: MIT
# Constraints for sequence mode (only used when RFSOC_AWG_SEQ=1).
#
# Control and status words between the AXI GPIOs (PS clock) and DACSeqStreamer (stream clock)
# are levels: the configuration is written before `arm`, status flags change slowly.
set seq_gpio_regs   [get_cells -quiet -hier -filter {IS_SEQUENTIAL && NAME =~ "*axi_gpio_seq*"}]
set seq_player_regs [get_cells -quiet -hier -filter {IS_SEQUENTIAL && NAME =~ "*DACRAMstreamer_0/*"}]
set_false_path -from $seq_gpio_regs   -to $seq_player_regs
set_false_path -from $seq_player_regs -to $seq_gpio_regs
