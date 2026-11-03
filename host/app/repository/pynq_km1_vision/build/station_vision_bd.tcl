# Isolated board-local vision candidate. Never overwrites the deployed dual-core assets.
set root [file normalize $::env(KM1_STATION_BUILD)]
set_param board.repoPaths [list $::env(PYNQ_BOARD_REPO)]
create_project station_vision [file join $root vivado_project] -part xc7z020clg400-1
set boards [get_board_parts -quiet *:pynq-z2:part0:*]
if {![llength $boards]} {error "PYNQ-Z2 board preset not found"}
set_property board_part [lindex $boards end] [current_project]
set_property ip_repo_paths [list [file join $root station_vision_accel hls impl ip]] [current_project]
update_ip_catalog
create_bd_design design_1
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} [get_bd_cells ps7]
set_property -dict [list CONFIG.PCW_USE_M_AXI_GP0 {1} CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100}] [get_bd_cells ps7]
create_bd_cell -type ip -vlnv xilinx.com:hls:station_vision_accel:1.0 station_vision_0
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config \
    {Master "/ps7/M_AXI_GP0" Clk_master "/ps7/FCLK_CLK0 (100 MHz)" Clk_slave "/ps7/FCLK_CLK0 (100 MHz)" Clk_xbar "/ps7/FCLK_CLK0 (100 MHz)"} \
    [get_bd_intf_pins station_vision_0/s_axi_control]
foreach port {base_mem frame_mem history_mem config_mem} {
    set options [list Master "/station_vision_0/m_axi_$port" Slave "/ps7/S_AXI_HP0" \
        intc_ip [expr {$port eq "base_mem" ? "New AXI Interconnect" : "/axi_mem_intercon"}] \
        Clk_master "/ps7/FCLK_CLK0 (100 MHz)" Clk_slave "/ps7/FCLK_CLK0 (100 MHz)" Clk_xbar "/ps7/FCLK_CLK0 (100 MHz)"]
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config $options [get_bd_intf_pins ps7/S_AXI_HP0]
}
assign_bd_address
validate_bd_design
save_bd_design
generate_target all [get_files design_1.bd]
make_wrapper -files [get_files design_1.bd] -top
add_files -norecurse [file join $root vivado_project/station_vision.gen/sources_1/bd/design_1/hdl/design_1_wrapper.v]
set_property top design_1_wrapper [current_fileset]
launch_runs impl_1 -to_step write_bitstream -jobs 1
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Station vision implementation failed"}
open_run impl_1
file mkdir [file join $root reports]
report_timing_summary -file [file join $root reports timing.rpt]
report_utilization -file [file join $root reports utilization.rpt]
report_drc -file [file join $root reports drc.rpt]
foreach delay {max min} {
    set paths [get_timing_paths -delay_type $delay -max_paths 1]
    if {![llength $paths] || [get_property SLACK [lindex $paths 0]] < 0} {error "$delay timing failed"}
}
if {[llength [get_drc_violations -filter {SEVERITY == Error}]]} {error "Station vision DRC failed"}
file mkdir [file join $root board]
file copy [file join $root vivado_project/station_vision.runs/impl_1/design_1_wrapper.bit] [file join $root board station_vision.bit]
file copy [file join $root vivado_project/station_vision.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh] [file join $root board station_vision.hwh]
exit
