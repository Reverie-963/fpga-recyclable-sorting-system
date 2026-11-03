# Historical entry point: intentionally disabled to prevent stale artifact export.
error {Use tools/build_dual_core.ps1 with a fresh output directory; see pynq_km1_vision/board/README_board_test.md}

# mask 集成收尾：report + 复制 bit/hwh + 导出 BD tcl（综合已成功，不重跑）
set root [file normalize [file join [file dirname [info script]] .. ..]]
file mkdir -force [file join $root report]

open_project [file join $root build/vivado_project_mask mask_bd.xpr]
open_run impl_1
report_timing_summary -file [file join $root report timing_mask.rpt]
report_utilization -file [file join $root report utilization_mask.rpt]
report_drc -file [file join $root report drc_mask.rpt]

set paths [get_timing_paths -delay_type max -max_paths 1]
if {[llength $paths] == 0 || [get_property SLACK [lindex $paths 0]] < 0} {
    error "Timing not met (worst slack below 0)"
}
puts "MASK_WORST_SLACK=[get_property SLACK [lindex $paths 0]]"

file mkdir -force [file join $root board]
file copy -force [file join $root build/vivado_project_mask/mask_bd.runs/impl_1/design_1_wrapper.bit] \
    [file join $root board vision.bit]
file copy -force [file join $root build/vivado_project_mask/mask_bd.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh] \
    [file join $root board vision.hwh]
puts "MASK_BIT_COPIED=[file exists [file join $root board vision.bit]]"
puts "MASK_HWH_COPIED=[file exists [file join $root board vision.hwh]]"

open_bd_design [get_files design_1.bd]
write_bd_tcl -force [file join $root build generated_mask_bd.tcl]
exit
