proc build_bitstream {} {
    reset_run synth_1
    launch_runs synth_1 -jobs $::cfg(jobs)
    wait_on_run synth_1

    set synth_status [get_property STATUS [get_runs synth_1]]
    puts "RUN_STATUS synth_1: $synth_status"
    if {$synth_status ne "synth_design Complete!"} {
        error "synth_1 did not complete successfully: $synth_status"
    }

    create_run $::cfg(run_name) \
        -parent_run synth_1 \
        -flow {Vivado Implementation 2022} \
        -strategy Performance_ExplorePostRoutePhysOpt \
        -constrset constrs_1

    launch_runs $::cfg(run_name) -to_step write_bitstream -jobs $::cfg(jobs)
    wait_on_run $::cfg(run_name)

    set impl_status [get_property STATUS [get_runs $::cfg(run_name)]]
    puts "RUN_STATUS $::cfg(run_name): $impl_status"
    if {![string match "*Complete*" $impl_status]} {
        error "$::cfg(run_name) did not complete successfully: $impl_status"
    }

    open_run $::cfg(run_name) -name ${::cfg(run_name)}_routed
    report_route_status -file [file join $::cfg(project_dir) ${::cfg(project_name)}.runs $::cfg(run_name) ${::cfg(top)}_route_status.rpt]
    report_timing_summary -max_paths 20 -report_unconstrained \
        -file [file join $::cfg(project_dir) ${::cfg(project_name)}.runs $::cfg(run_name) ${::cfg(top)}_timing_summary_final.rpt]
    report_drc -file [file join $::cfg(project_dir) ${::cfg(project_name)}.runs $::cfg(run_name) ${::cfg(top)}_drc_final.rpt]
}
