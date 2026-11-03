# Append the candidate pixel core to a copied, reviewed vision+motion BD.
# No board programming, UART/GPIO outputs, PS clock or existing address changes.
set root [file normalize $::env(KM1_PHASE_JOINT_BUILD)]
set_param general.maxThreads 2
set_param board.repoPaths [list $::env(PYNQ_BOARD_REPO)]
create_project j [file join $root v] -part xc7z020clg400-1
set_property ip_repo_paths [list [file join $root ip vision] [file join $root ip motion] [file join $root ip phase]] [current_project]
update_ip_catalog
import_files -norecurse [file join $root sources design_1.bd]
open_bd_design [get_files design_1.bd]
source -notrace $::env(IPCFG_LIBRARY)
set ::ipcfg::native_log_path [file join $root native-vivado.log]
set runtime [ipcfg::python_status]
puts $runtime
if {![string match {PYTHON:OK *} $runtime]} {error $runtime}
puts [ipcfg::python_exec -c {import argparse,base64,hashlib,json,pathlib,re,sys,typing;print("PYTHON_IMPORTS:OK")}]
proc phase_operation {name script} {
    global root
    set receipt [ipcfg::capture [file join $root evidence "$name.tcldict"] $script]
    puts "PHASE_CAPTURE:$name $receipt"
    if {[dict get $receipt status] ne "OK"} {error "phase_configuration_failed:$name"}
}
set ps_expected [list CONFIG.PCW_USE_M_AXI_GP0 1 CONFIG.PCW_USE_S_AXI_HP0 1 CONFIG.PCW_USE_S_AXI_HP1 1 CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ 100]
proc verify_existing {} {
    global ps_expected
    foreach {key expected} $ps_expected {
        if {[expr {double([get_property $key [get_bd_cells ps7]])}] != $expected} {error "existing_PS_configuration_changed:$key"}
    }
    foreach {name expected} {station_vision_0 0x40000000 motion_path_0 0x40010000} {
        set seg [get_bd_addr_segs ps7/Data/SEG_${name}_Reg]
        if {[llength $seg]!=1 || [expr {[get_property OFFSET $seg]}]!=$expected} {error "existing_control_address_changed:$name"}
        set raw_range [get_property RANGE $seg]
        if {[regexp {^([0-9]+)[Kk]$} $raw_range -> count]} {set range [expr {$count*1024}]} else {set range [expr {$raw_range+0}]}
        if {$range!=65536} {error "existing_control_range_changed:$name"}
    }
}
verify_existing
phase_operation create_pixel {set created [ipcfg::create_cell xilinx.com:hls:phase_pixel_accel:1.0 phase_pixel_0]; if {[string match {CONFIGURE_FAIL:*} $created]} {error $created}}
phase_operation control {
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config \
        {Master "/ps7/M_AXI_GP0" Clk_master "/ps7/FCLK_CLK0 (100 MHz)" Clk_slave "/ps7/FCLK_CLK0 (100 MHz)" Clk_xbar "/ps7/FCLK_CLK0 (100 MHz)"} \
        [get_bd_intf_pins phase_pixel_0/s_axi_control]
}
foreach {port hp intercon} {frame_a HP0 axi_mem_intercon frame_b HP1 axi_mem_intercon_1} {
    set options [list Master "/phase_pixel_0/m_axi_$port" Slave "/ps7/S_AXI_$hp" intc_ip "/$intercon" \
        Clk_master "/ps7/FCLK_CLK0 (100 MHz)" Clk_slave "/ps7/FCLK_CLK0 (100 MHz)" Clk_xbar "/ps7/FCLK_CLK0 (100 MHz)"]
    phase_operation $port [list apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config $options [get_bd_intf_pins ps7/S_AXI_$hp]]
}
phase_operation addresses {
    assign_bd_address -offset 0x40020000 -range 64K -target_address_space [get_bd_addr_spaces ps7/Data] [get_bd_addr_segs phase_pixel_0/s_axi_control/Reg]
    assign_bd_address
}
verify_existing
set pixel_seg [get_bd_addr_segs ps7/Data/SEG_phase_pixel_0_Reg]
if {[expr {[get_property OFFSET $pixel_seg]}]!=0x40020000} {error "pixel_address_not_assigned"}
foreach pin {ap_clk ap_rst_n} {
    set expected [get_bd_nets -of_objects [get_bd_pins station_vision_0/$pin]]
    set actual [get_bd_nets -of_objects [get_bd_pins phase_pixel_0/$pin]]
    if {$actual ne $expected} {error "pixel_clock_or_reset_not_shared:$pin"}
}
phase_operation validate {validate_bd_design; save_bd_design}
set proof [open [file join $root evidence interfaces.txt] w]
foreach path {phase_pixel_0/s_axi_control phase_pixel_0/m_axi_frame_a phase_pixel_0/m_axi_frame_b} {
    set pin [get_bd_intf_pins $path]
    puts $proof "$path VLNV=[get_property VLNV $pin] MODE=[get_property MODE $pin] DATA_WIDTH=[get_property CONFIG.DATA_WIDTH $pin] NET=[get_bd_intf_nets -of_objects $pin]"
}
puts $proof "old_addresses_preserved=true pixel_base=0x40020000 old_PS_GP0_HP0_HP1_100MHz=true pixel_mutations=5"
close $proof
report_ip_status -file [file join $root evidence ip-status.rpt]
generate_target all [get_files design_1.bd]
set wrappers [make_wrapper -files [get_files design_1.bd] -top]
add_files -norecurse $wrappers
set_property top design_1_wrapper [current_fileset]
launch_runs synth_1 -jobs 2
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "phase_joint_synthesis_failed"}
launch_runs impl_1 -to_step route_design -jobs 2
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "phase_joint_route_failed"}
open_run impl_1
report_timing_summary -file [file join $root reports timing.rpt]
report_utilization -file [file join $root reports utilization.rpt]
report_drc -file [file join $root reports drc.rpt]
report_methodology -file [file join $root reports methodology.rpt]
foreach delay {max min} {
    set paths [get_timing_paths -delay_type $delay -max_paths 1]
    if {![llength $paths] || [get_property SLACK [lindex $paths 0]]<0} {error "phase_joint_${delay}_timing_failed"}
}
if {[llength [get_drc_violations -filter {SEVERITY == Error}]]} {error "phase_joint_DRC_failed"}
file mkdir [file join $root board]
write_bitstream -force [file join $root board phase_vision_motion.bit]
set handoffs [glob -nocomplain -directory [file join $root v/j.gen/sources_1/bd] */hw_handoff/*.hwh]
if {[llength $handoffs]!=1} {error "unambiguous_same_build_HWH_required"}
file copy [lindex $handoffs 0] [file join $root board phase_vision_motion.hwh]
puts "PHASE_JOINT_PAIRED_CANDIDATE_COMPLETE: no board load or physical acceptance"
exit
