set ::cfg(repo_root) [file normalize [file join [file dirname [info script]] ..]]
set ::cfg(vivado_version) "2022.1"
set ::cfg(project_name) "thesis_9_83GHz"
set ::cfg(part) "xczu48dr-fsvg1517-2-e"
set ::cfg(board_part) "xilinx.com:zcu208:part0:2.0"
set ::cfg(top) "bd_wrapper"
set ::cfg(bd_name) "bd"
set ::cfg(jobs) 12

set ::cfg(build_root) [file join $::cfg(repo_root) build]
set ::cfg(project_dir) [file join $::cfg(build_root) vivado]
set ::cfg(export_root) [file join $::cfg(repo_root) artifacts]
set ::cfg(stamp) [clock format [clock seconds] -format "%Y%m%d_%H%M%S"]
set ::cfg(run_name) "impl_dual_dacplay_full_$::cfg(stamp)"
set ::cfg(export_name) "pl122p88-ps-s00-100m-dual-dacplay-full-$::cfg(stamp)"
set ::cfg(export_dir) [file join $::cfg(export_root) $::cfg(export_name)]
set ::cfg(pynq_name) "thesis"

set ::cfg(rtl_files) [list \
    [file join $::cfg(repo_root) src rtl DACRAMstreamer.v] \
]
set ::cfg(xdc_file) [file join $::cfg(repo_root) src constraints bd.xdc]
set ::cfg(bd_tcl_file) [file join $::cfg(repo_root) src bd bd_bd.tcl]

proc require_vivado_version {} {
    set actual [version -short]
    if {[string first $::cfg(vivado_version) $actual] == -1} {
        error "This build is tested only with Vivado $::cfg(vivado_version); running $actual"
    }
    puts "CHECK OK: Vivado version $actual"
}

proc require_file {label path} {
    if {![file exists $path]} {
        error "Missing $label: $path"
    }
    puts "CHECK OK: $label: $path"
}

proc require_equal {label actual expected} {
    if {$actual ne $expected} {
        error "$label mismatch: expected '$expected', got '$actual'"
    }
    puts "CHECK OK: $label = $actual"
}

proc require_connected {label obj} {
    set connected [find_bd_objs -relation connected_to $obj]
    if {[llength $connected] == 0} {
        error "$label is not connected"
    }
    puts "CHECK OK: $label connected"
}
