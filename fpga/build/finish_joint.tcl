set root [file normalize $::env(KM1_FINISH_ROOT)]
file mkdir $root
file mkdir [file join $root results]
set_param general.maxThreads 2
open_checkpoint $::env(KM1_ROUTED_CHECKPOINT)
report_timing_summary -file [file join $root results timing_before.rpt]
phys_opt_design -directive Explore
route_design -directive Explore
report_timing_summary -file [file join $root results timing_after.rpt]
report_utilization -file [file join $root results utilization.rpt]
report_drc -file [file join $root results drc_after.rpt]
report_methodology -file [file join $root results methodology_after.rpt]
write_checkpoint -force [file join $root joint_physopt.dcp]
foreach delay {max min} {
 set paths [get_timing_paths -delay_type $delay -max_paths 1]
 if {![llength $paths] || [get_property SLACK [lindex $paths 0]]<0} {error "$delay timing failed"}
}
if {[llength [get_drc_violations -filter {SEVERITY == Error}]]} {error "Joint DRC failed"}
file mkdir [file join $root board]
write_bitstream -force [file join $root board vision_motion.bit]
file copy $::env(KM1_ROUTED_HWH) [file join $root board vision_motion.hwh]
exit
