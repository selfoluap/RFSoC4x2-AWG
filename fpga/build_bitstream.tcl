set script_dir [file dirname [file normalize [info script]]]

source [file join $script_dir scripts settings.tcl]
source [file join $script_dir scripts build_bitstream.tcl]
source [file join $script_dir scripts export_artifacts.tcl]

set project_file [file join $::cfg(project_dir) ${::cfg(project_name)}.xpr]
if {![file exists $project_file]} {
    error "Missing Vivado project: $project_file. Run create_project.tcl first."
}

open_project $project_file
build_bitstream
export_artifacts

puts "BUILD_BITSTREAM_DONE $::cfg(export_dir)"
