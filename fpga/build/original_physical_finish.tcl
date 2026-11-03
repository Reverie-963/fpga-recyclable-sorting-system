set root C:/FPGA-pl-motion/local-only/jm2
set_param general.maxThreads 2
open_checkpoint C:/FPGA-pl-motion/local-only/jm1/v/j.runs/impl_1/design_1_wrapper_routed.dcp
report_methodology -json [file join $root results methodology.json]
write_xdc -force -exclude_physical [file join $root results constraints.xdc]
report_timing -max_paths 20 -file [file join $root results timing_before.rpt]
help phys_opt_design
phys_opt_design -directive Explore
route_design -directive Explore
report_timing_summary -file [file join $root results timing_after.rpt]
report_drc -file [file join $root results drc_after.rpt]
report_methodology -json [file join $root results methodology_after_fix.json]
write_xdc -force -exclude_physical [file join $root results constraints_after_fix.xdc]
write_checkpoint -force [file join $root joint_physopt.dcp]
foreach delay {max min} {
 set paths [get_timing_paths -delay_type $delay -max_paths 1]
 if {![llength $paths] || [get_property SLACK [lindex $paths 0]]<0} {error "$delay timing failed"}
}
if {[llength [get_drc_violations -filter {SEVERITY == Error}]]} {error "Joint DRC failed"}
file mkdir [file join $root board]
report_utilization -file [file join $root results utilization.rpt]
write_bitstream -force [file join $root board vision_motion.bit]
file copy C:/FPGA-pl-motion/local-only/jm1/v/j.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh [file join $root board vision_motion.hwh]
exit
