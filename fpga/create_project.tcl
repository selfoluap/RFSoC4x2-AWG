set script_dir [file dirname [file normalize [info script]]]

source [file join $script_dir scripts settings.tcl]
source [file join $script_dir scripts create_project.tcl]

create_project_from_sources

puts "CREATE_PROJECT_DONE [file join $::cfg(project_dir) ${::cfg(project_name)}.xpr]"
