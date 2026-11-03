# Isolated board-local vision candidate. Never overwrites the deployed dual-core assets.
set root [file normalize $::env(KM1_FRAME_BUILD)]
set_param board.repoPaths [list $::env(PYNQ_BOARD_REPO)]
create_project frame_pipeline [file join $root vivado_project] -part xc7z020clg400-1
set boards [get_board_parts -quiet *:pynq-z2:part0:*]
if {![llength $boards]} {error "PYNQ-Z2 board preset not found"}
set_property board_part [lindex $boards end] [current_project]
set_property ip_repo_paths [list [file join $root station_frame_pipeline hls impl ip]] [current_project]
update_ip_catalog
create_bd_design design_1
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} [get_bd_cells ps7]
set_property -dict [list CONFIG.PCW_USE_M_AXI_GP0 {1} CONFIG.PCW_USE_S_AXI_HP0 {1} CONFIG.PCW_USE_S_AXI_HP1 {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100}] [get_bd_cells ps7]
create_bd_cell -type ip -vlnv xilinx.com:hls:station_frame_pipeline:1.0 frame_pipeline_0
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config \
    {Master "/ps7/M_AXI_GP0" Clk_master "/ps7/FCLK_CLK0 (100 MHz)" Clk_slave "/ps7/FCLK_CLK0 (100 MHz)" Clk_xbar "/ps7/FCLK_CLK0 (100 MHz)"} \
    [get_bd_intf_pins frame_pipeline_0/s_axi_control]
foreach {port hp interconnect first} {
    base_mem HP0 frame_hp0 1
    old_hist_mem HP0 frame_hp0 0
    config_mem HP0 frame_hp0 0
    frame_mem HP1 frame_hp1 1
    new_hist_mem HP1 frame_hp1 0
    preview_mem HP1 frame_hp1 0
} {
    set options [list Master "/frame_pipeline_0/m_axi_$port" Slave "/ps7/S_AXI_$hp" \
        intc_ip [expr {$first ? "New AXI Interconnect" : "/axi_mem_intercon"}] \
        Clk_master "/ps7/FCLK_CLK0 (100 MHz)" Clk_slave "/ps7/FCLK_CLK0 (100 MHz)" Clk_xbar "/ps7/FCLK_CLK0 (100 MHz)"]
    if {!$first} {
        set options [lreplace $options 5 5 "/$interconnect"]
    }
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config $options [get_bd_intf_pins ps7/S_AXI_$hp]
    if {$first} {
        # Name each independent PS HP crossbar explicitly for subsequent masters.
        set nets [get_bd_intf_nets -of_objects [get_bd_intf_pins ps7/S_AXI_$hp]]
        set pins [get_bd_intf_pins -of_objects $nets -filter {MODE == Master}]
        set cell [get_bd_cells -of_objects $pins]
        if {[llength $cell]!=1} {error "Could not resolve $hp AXI crossbar"}
        set_property name $interconnect $cell
    }
}
assign_bd_address
validate_bd_design
save_bd_design
generate_target all [get_files design_1.bd]
make_wrapper -files [get_files design_1.bd] -top
add_files -norecurse [file join $root vivado_project/frame_pipeline.gen/sources_1/bd/design_1/hdl/design_1_wrapper.v]
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
file copy [file join $root vivado_project/frame_pipeline.runs/impl_1/design_1_wrapper.bit] [file join $root board frame_pipeline.bit]
file copy [file join $root vivado_project/frame_pipeline.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh] [file join $root board frame_pipeline.hwh]
exit
