set script_dir [file dirname [file normalize [info script]]]

puts "ERROR: build.tcl has been split into explicit steps."
puts "Run create_project.tcl first, then build_bitstream.tcl."
puts "For a one-command full rebuild, run build_all.tcl."
error "Use create_project.tcl, build_bitstream.tcl, or build_all.tcl"
