# SPDX-License-Identifier: MIT
# Constraints for the ADC capture path (only used when RFSOC_AWG_ADC=1).
#
# Control and status words between the AXI GPIOs (PS clock) and ADCCapture (stream clock) are
# levels: the configuration is written before `arm`, status flags change slowly.
set adc_gpio_regs    [get_cells -quiet -hier -filter {IS_SEQUENTIAL && NAME =~ "*axi_gpio_adc_*"}]
set adc_capture_regs [get_cells -quiet -hier -filter {IS_SEQUENTIAL && NAME =~ "*adc_capture_0/*"}]
set_false_path -from $adc_gpio_regs    -to $adc_capture_regs
set_false_path -from $adc_capture_regs -to $adc_gpio_regs
