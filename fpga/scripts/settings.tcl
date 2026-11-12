set ::cfg(repo_root) [file normalize [file join [file dirname [info script]] ..]]
set ::cfg(vivado_version) "2024.1"
set ::cfg(project_name) "rfsocawg"
set ::cfg(part) "xczu48dr-fsvg1517-2-e"
set ::cfg(board_part) "xilinx.com:zcu208:part0:2.0"
set ::cfg(top) "bd_wrapper"
set ::cfg(bd_name) "bd"
set ::cfg(jobs) 3

set ::cfg(build_root) [file join $::cfg(repo_root) build]
set ::cfg(project_dir) [file join $::cfg(build_root) vivado]
set ::cfg(export_root) [file join $::cfg(repo_root) artifacts]
set ::cfg(stamp) [clock format [clock seconds] -format "%Y%m%d_%H%M%S"]
set ::cfg(run_name) "impl_dual_dacplay_full_$::cfg(stamp)"
set ::cfg(export_name) "rfsocawg-dual-dacplay-full-$::cfg(stamp)"
set ::cfg(export_dir) [file join $::cfg(export_root) $::cfg(export_name)]
set ::cfg(pynq_name) "rfsocawg"

set ::cfg(rtl_files) [list \
    [file join $::cfg(repo_root) src rtl DACRAMstreamer.v] \
]
set ::cfg(xdc_file) [file join $::cfg(repo_root) src constraints bd.xdc]
set ::cfg(bd_tcl_file) [file join $::cfg(repo_root) src bd bd_bd.tcl]

# Optional PL DDR4 long-waveform player: build with RFSOC_AWG_DDR4=1 in the environment.
# Off by default, so the reference bitstream is reproduced unchanged. See DDR4_PLAYER.md.
set ::cfg(enable_ddr4) [expr {[info exists ::env(RFSOC_AWG_DDR4)] && $::env(RFSOC_AWG_DDR4) eq "1"}]
set ::cfg(ddr4_rtl_files) [list \
    [file join $::cfg(repo_root) src rtl awg_async_fifo.v] \
    [file join $::cfg(repo_root) src rtl DDR4Streamer.v] \
    [file join $::cfg(repo_root) src rtl DacSourceMux.v] \
    [file join $::cfg(repo_root) src rtl DDR4Player.v] \
]
set ::cfg(ddr4_xdc_file) [file join $::cfg(repo_root) src constraints ddr4_pl.xdc]

# Optional sequence mode for the on-chip-memory player: build with RFSOC_AWG_SEQ=1.
# Off by default. See SEQUENCER.md. Can be combined with RFSOC_AWG_DDR4=1.
set ::cfg(enable_seq) [expr {[info exists ::env(RFSOC_AWG_SEQ)] && $::env(RFSOC_AWG_SEQ) eq "1"}]
set ::cfg(seq_rtl_files) [list [file join $::cfg(repo_root) src rtl DACSeqStreamer.v]]
set ::cfg(seq_xdc_file) [file join $::cfg(repo_root) src constraints sequencer.xdc]
set ::cfg(seq_bd_tcl_file) [file join $::cfg(repo_root) src bd add_sequencer.tcl]
set ::cfg(trigger_xdc_file) [file join $::cfg(repo_root) src constraints ext_trigger.xdc]

# Optional ADC capture path: build with RFSOC_AWG_ADC=1 (needs RFSOC_AWG_SEQ=1).
# Off by default. See ADC_CAPTURE.md.
set ::cfg(enable_adc) [expr {[info exists ::env(RFSOC_AWG_ADC)] && $::env(RFSOC_AWG_ADC) eq "1"}]
set ::cfg(adc_rtl_files) [list [file join $::cfg(repo_root) src rtl ADCCapture.v]]
set ::cfg(adc_xdc_file) [file join $::cfg(repo_root) src constraints adc_capture.xdc]
set ::cfg(adc_bd_tcl_file) [file join $::cfg(repo_root) src bd add_adc_capture.tcl]
set ::cfg(ddr4_bd_tcl_file) [file join $::cfg(repo_root) src bd add_ddr4_player.tcl]

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

proc require_absent {label obj} {
    if {[llength $obj] != 0} {
        error "$label should not exist: $obj"
    }
    puts "CHECK OK: $label absent"
}
