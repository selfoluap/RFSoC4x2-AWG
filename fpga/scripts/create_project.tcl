proc create_project_from_sources {} {
    require_vivado_version

    foreach rtl_file $::cfg(rtl_files) {
        require_file "RTL source" $rtl_file
    }
    require_file "XDC constraints" $::cfg(xdc_file)
    require_file "block design Tcl" $::cfg(bd_tcl_file)

    file mkdir $::cfg(build_root)
    if {[file exists $::cfg(project_dir)]} {
        file delete -force $::cfg(project_dir)
    }

    create_project $::cfg(project_name) $::cfg(project_dir) -part $::cfg(part)
    set_property board_part $::cfg(board_part) [current_project]
    set_property target_language VHDL [current_project]
    set_property default_lib xil_defaultlib [current_project]

    add_files -fileset sources_1 $::cfg(rtl_files)
    add_files -fileset constrs_1 $::cfg(xdc_file)
    update_compile_order -fileset sources_1

    source $::cfg(bd_tcl_file)
    open_bd_design [get_files [format "*/%s.bd" $::cfg(bd_name)]]

    validate_current_design
    generate_target all [get_files [format "*/%s.bd" $::cfg(bd_name)]]
    export_ip_user_files -of_objects [get_files [format "*/%s.bd" $::cfg(bd_name)]] -no_script -sync -force -quiet
    make_wrapper -files [get_files [format "*/%s.bd" $::cfg(bd_name)]] -top -import -force
    update_compile_order -fileset sources_1
    set_property top $::cfg(top) [current_fileset]

    puts "PROJECT_CREATED [current_project]"
}

proc validate_current_design {} {
    validate_bd_design
    save_bd_design

    require_equal "clkwiz PRIM_IN_FREQ" \
        [get_property CONFIG.PRIM_IN_FREQ [get_bd_cells /clocktree/clkwiz]] "122.88"
    require_equal "clkwiz CLKOUT1_REQUESTED_OUT_FREQ" \
        [get_property CONFIG.CLKOUT1_REQUESTED_OUT_FREQ [get_bd_cells /clocktree/clkwiz]] "614.4"
    require_equal "clkwiz CLKOUT2_REQUESTED_OUT_FREQ" \
        [get_property CONFIG.CLKOUT2_REQUESTED_OUT_FREQ [get_bd_cells /clocktree/clkwiz]] "307.2"

    require_equal "DAC0 streamer MEM_SIZE_BYTES" \
        [get_property CONFIG.MEM_SIZE_BYTES [get_bd_cells /hier_dac_play/DACRAMstreamer_0]] "262144"
    require_equal "DAC0 streamer USE_VECTOR_COUNT" \
        [get_property CONFIG.USE_VECTOR_COUNT [get_bd_cells /hier_dac_play/DACRAMstreamer_0]] "1"
    require_equal "DAC2 streamer MEM_SIZE_BYTES" \
        [get_property CONFIG.MEM_SIZE_BYTES [get_bd_cells /hier_dac2_play/DACRAMstreamer_0]] "262144"
    require_equal "DAC2 streamer USE_VECTOR_COUNT" \
        [get_property CONFIG.USE_VECTOR_COUNT [get_bd_cells /hier_dac2_play/DACRAMstreamer_0]] "1"

    require_equal "DAC GPIO channel 1 width" \
        [get_property CONFIG.C_GPIO_WIDTH [get_bd_cells /gpio_control/axi_gpio_dac]] "2"
    require_equal "DAC GPIO channel 2 width" \
        [get_property CONFIG.C_GPIO2_WIDTH [get_bd_cells /gpio_control/axi_gpio_dac]] "18"
    require_equal "DAC2 GPIO width" \
        [get_property CONFIG.C_GPIO_WIDTH [get_bd_cells /gpio_control/axi_gpio_dac2]] "18"

    require_connected "DAC0 enable" [get_bd_pins /hier_dac_play/enable]
    require_connected "DAC2 enable" [get_bd_pins /hier_dac2_play/enable]
    require_connected "DAC0 sample_count" [get_bd_pins /hier_dac_play/sample_count]
    require_connected "DAC2 sample_count" [get_bd_pins /hier_dac2_play/sample_count]
    require_connected "DAC2 RFDC stream" [get_bd_intf_pins /usp_rf_data_converter_1/s20_axis]
}
