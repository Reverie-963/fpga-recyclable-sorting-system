# Combined vision + nonvisual height overlay. No UART/GPIO motion outputs.
set root [file normalize $::env(KM1_JOINT_BUILD)]
set_param board.repoPaths [list $::env(PYNQ_BOARD_REPO)]
set_param general.maxThreads 2
create_project j [file join $root v] -part xc7z020clg400-1
set boards [get_board_parts -quiet *:pynq-z2:part0:*]
if {![llength $boards]} {error "PYNQ-Z2 board preset required"}
set_property board_part [lindex $boards end] [current_project]
set_property ip_repo_paths [list [file join $root ip vision] [file join $root ip motion]] [current_project]
update_ip_catalog
create_bd_design design_1
set use_helper [info exists ::env(IPCFG_LIBRARY)]
if {$use_helper} {
    source -notrace $::env(IPCFG_LIBRARY)
    set ::ipcfg::native_log_path [file join $root native-vivado.log]
    set status [ipcfg::python_status]
    puts $status
    if {![string match {PYTHON:OK *} $status]} {error $status}
    puts [ipcfg::python_exec -c {import argparse,base64,hashlib,json,pathlib,re,sys,typing;print("PYTHON_IMPORTS:OK")}]
}
proc joint_op {name command} {
    global root use_helper
    if {!$use_helper} {return [uplevel #0 $command]}
    set ::joint_result __not_returned__
    set receipt [ipcfg::capture [file join $root evidence "$name.tcldict"] "set ::joint_result \[[list {*}$command]\]"]
    puts "CONFIG_EVIDENCE:$name $receipt"
    if {$::joint_result eq "__not_returned__" || [string match {CONFIGURE_FAIL:*} $::joint_result]} {
        error "Configuration failed: $name $::joint_result"
    }
    # Native preset critical warnings remain FAIL in the capture evidence.
    # Required readbacks/interfaces below are independently mandatory.
    return $::joint_result
}
if {$use_helper} {
    puts [joint_op ps_create [list ipcfg::create_cell xilinx.com:ip:processing_system7:5.5 ps7]]
    puts [joint_op ps_board [list ipcfg::run_automation ps7 xilinx.com:bd_rule:processing_system7 \
        {make_external {FIXED_IO, DDR} apply_board_preset 1 Master Disable Slave Disable}]]
} else {
    create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
    apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
        -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} [get_bd_cells ps7]
}
set desired [list CONFIG.PCW_USE_M_AXI_GP0 1 CONFIG.PCW_USE_S_AXI_HP0 1 CONFIG.PCW_USE_S_AXI_HP1 1 \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ 100]
if {$use_helper} {
    puts [joint_op ps_ports [list ipcfg::apply_dict ps7 $desired]]
    set bad [ipcfg::verify_stuck ps7 $desired]
    if {[llength $bad]} {error "PS configuration readback failed: $bad"}
} else {set_property -dict $desired [get_bd_cells ps7]}
foreach {key value} $desired {
    if {[expr {double([get_property $key [get_bd_cells ps7]])}] != [expr {double($value)}]} {error "PS configuration changed: $key"}
}
foreach interface {M_AXI_GP0 S_AXI_HP0 S_AXI_HP1} {
    if {[llength [get_bd_intf_pins ps7/$interface]]!=1} {error "Missing PS interface: $interface"}
}
foreach {name vlnv} {station_vision_0 xilinx.com:hls:station_vision_accel:1.0 motion_path_0 xilinx.com:hls:motion_path_accel:1.0} {
    if {$use_helper} {puts [joint_op $name [list ipcfg::create_cell $vlnv $name]]} \
    else {create_bd_cell -type ip -vlnv $vlnv $name}
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config \
        {Master "/ps7/M_AXI_GP0" Clk_master "/ps7/FCLK_CLK0 (100 MHz)" Clk_slave "/ps7/FCLK_CLK0 (100 MHz)" Clk_xbar "/ps7/FCLK_CLK0 (100 MHz)"} \
        [get_bd_intf_pins $name/s_axi_control]
}
foreach port {base_mem frame_mem history_mem config_mem} {
    set options [list Master "/station_vision_0/m_axi_$port" Slave "/ps7/S_AXI_HP0" \
        intc_ip [expr {$port eq "base_mem" ? "New AXI Interconnect" : "/axi_mem_intercon"}] \
        Clk_master "/ps7/FCLK_CLK0 (100 MHz)" Clk_slave "/ps7/FCLK_CLK0 (100 MHz)" Clk_xbar "/ps7/FCLK_CLK0 (100 MHz)"]
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config $options [get_bd_intf_pins ps7/S_AXI_HP0]
}
foreach port {input_mem output_mem} {
    set options [list Master "/motion_path_0/m_axi_$port" Slave "/ps7/S_AXI_HP1" \
        intc_ip [expr {$port eq "input_mem" ? "New AXI Interconnect" : "/axi_mem_intercon_1"}] \
        Clk_master "/ps7/FCLK_CLK0 (100 MHz)" Clk_slave "/ps7/FCLK_CLK0 (100 MHz)" Clk_xbar "/ps7/FCLK_CLK0 (100 MHz)"]
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config $options [get_bd_intf_pins ps7/S_AXI_HP1]
}
assign_bd_address
validate_bd_design
save_bd_design
set report [open [file join $root evidence interfaces.txt] w]
foreach {key value} $desired {puts $report "$key=[get_property $key [get_bd_cells ps7]]"}
foreach interface {ps7/M_AXI_GP0 ps7/S_AXI_HP0 ps7/S_AXI_HP1 station_vision_0/s_axi_control motion_path_0/s_axi_control} {
    set pin [get_bd_intf_pins $interface]
    puts $report "$interface VLNV=[get_property VLNV $pin] MODE=[get_property MODE $pin] NET=[get_bd_intf_nets -of_objects $pin]"
}
close $report
puts "COVERAGE:FULL GP0+HP0+HP1+100MHz configuration readbacks and interfaces; preset warning evidence retained"
generate_target all [get_files design_1.bd]
make_wrapper -files [get_files design_1.bd] -top
add_files -norecurse [file join $root v/j.gen/sources_1/bd/design_1/hdl/design_1_wrapper.v]
set_property top design_1_wrapper [current_fileset]
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "Joint synthesis failed"}
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Joint implementation failed"}
open_run impl_1
file mkdir [file join $root reports]
report_timing_summary -file [file join $root reports timing.rpt]
report_utilization -file [file join $root reports utilization.rpt]
report_drc -file [file join $root reports drc.rpt]
foreach delay {max min} {
    set paths [get_timing_paths -delay_type $delay -max_paths 1]
    if {![llength $paths] || [get_property SLACK [lindex $paths 0]]<0} {error "$delay timing failed"}
}
if {[llength [get_drc_violations -filter {SEVERITY == Error}]]} {error "Joint DRC failed"}
file mkdir [file join $root board]
file copy [file join $root v/j.runs/impl_1/design_1_wrapper.bit] [file join $root board vision_motion.bit]
file copy [file join $root v/j.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh] [file join $root board vision_motion.hwh]
exit
