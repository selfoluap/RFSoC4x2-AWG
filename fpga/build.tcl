set script_dir [file dirname [file normalize [info script]]]

source [file join $script_dir scripts settings.tcl]
source [file join $script_dir scripts create_project.tcl]
source [file join $script_dir scripts build_bitstream.tcl]
source [file join $script_dir scripts export_artifacts.tcl]

create_project_from_sources
build_bitstream
export_artifacts

puts "BUILD_DONE $::cfg(export_dir)"
