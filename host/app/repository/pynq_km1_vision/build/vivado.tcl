# Run from an ASCII local path, with official PYNQ-Z2 board files installed.
set root [file normalize [file join [file dirname [info script]] ..]]
set version_file [open [file join $root build toolchain_version.txt] r]
set required_version [string trim [read $version_file]]
close $version_file
if {[version -short] ne $required_version} {
 error "Vivado $required_version is required; running [version -short]"
}
cd $root
if {![info exists ::env(PYNQ_BOARD_REPO)]} {error "Set PYNQ_BOARD_REPO to directory containing PYNQ-Z2 board files"}
set_param board.repoPaths [list $::env(PYNQ_BOARD_REPO)]
create_project -force vision build/vivado_project -part xc7z020clg400-1
set boards [get_board_parts -quiet *:pynq-z2:part0:*]
if {[llength $boards] == 0} {error "PYNQ-Z2 board preset not found. Do not substitute Z1 preset."}
set_property board_part [lindex $boards end] [current_project]
set kernel segment
if {[info exists ::env(KM1_KERNEL)]} {set kernel $::env(KM1_KERNEL)}
if {$kernel ni {segment segment_open}} {error "Invalid KM1_KERNEL"}
set hls_ip_dir [file join $root build hls_${kernel} hls impl ip]
if {![file isfile [file join $hls_ip_dir component.xml]]} {error "Build and package HLS $kernel first"}
set_property ip_repo_paths [list $hls_ip_dir] [current_project]
update_ip_catalog
add_files -norecurse src/rtl/conveyor_encoder.v
create_bd_design design_1
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} [get_bd_cells ps7]
set_property -dict [list CONFIG.PCW_USE_M_AXI_GP0 {1} CONFIG.PCW_USE_S_AXI_HP0 {1} CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100}] [get_bd_cells ps7]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
set_property -dict [list CONFIG.c_include_sg {0} CONFIG.c_sg_length_width {23} CONFIG.c_m_axis_mm2s_tdata_width {32} CONFIG.c_s_axis_s2mm_tdata_width {32} CONFIG.c_include_mm2s_dre {1} CONFIG.c_include_s2mm_dre {1}] [get_bd_cells axi_dma_0]
set kernels [get_ipdefs -all -filter "NAME == $kernel"]
if {[llength $kernels]==0} {error "Build HLS segment IP first"}
create_bd_cell -type ip -vlnv [lindex $kernels end] segment_0
set axis_in [get_bd_intf_pins -of_objects [get_bd_cells segment_0] -filter {MODE == Slave && VLNV =~ "*:axis_rtl:*"}]
set axis_out [get_bd_intf_pins -of_objects [get_bd_cells segment_0] -filter {MODE == Master && VLNV =~ "*:axis_rtl:*"}]
if {[llength $axis_in]!=1 || [llength $axis_out]!=1} {error "Expected exactly one AXIS input and output"}
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] $axis_in
connect_bd_intf_net $axis_out [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
foreach slave {axi_dma_0/S_AXI_LITE segment_0/s_axi_control} {
 apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master "/ps7/M_AXI_GP0" Clk_master "/ps7/FCLK_CLK0 (100 MHz)" Clk_slave "/ps7/FCLK_CLK0 (100 MHz)" Clk_xbar "/ps7/FCLK_CLK0 (100 MHz)"} [get_bd_intf_pins $slave]
}
# Create the DDR interconnect once, then add S2MM to that same interconnect.
# Re-running automation on HP0 created an unconnected second interconnect.
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master "/axi_dma_0/M_AXI_MM2S" Slave "/ps7/S_AXI_HP0" intc_ip "New AXI Interconnect" Clk_master "/ps7/FCLK_CLK0 (100 MHz)" Clk_slave "/ps7/FCLK_CLK0 (100 MHz)" Clk_xbar "/ps7/FCLK_CLK0 (100 MHz)"} [get_bd_intf_pins ps7/S_AXI_HP0]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master "/axi_dma_0/M_AXI_S2MM" Slave "/ps7/S_AXI_HP0" intc_ip "/axi_mem_intercon" Clk_master "/ps7/FCLK_CLK0 (100 MHz)" Clk_slave "/ps7/FCLK_CLK0 (100 MHz)" Clk_xbar "/ps7/FCLK_CLK0 (100 MHz)"} [get_bd_intf_pins axi_dma_0/M_AXI_S2MM]
# Encoder input pins are installation-specific: require a reviewed XDC, never guess package pins.
if {![info exists ::env(KM1_ENCODER_XDC)] || ![file exists $::env(KM1_ENCODER_XDC)]} {
 error "Set KM1_ENCODER_XDC to reviewed encoder pin constraints"
}
add_files -fileset constrs_1 -norecurse $::env(KM1_ENCODER_XDC)
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 conveyor_gpio
set_property -dict [list CONFIG.C_IS_DUAL {1} CONFIG.C_GPIO_WIDTH {32} CONFIG.C_GPIO2_WIDTH {32} CONFIG.C_ALL_OUTPUTS {1} CONFIG.C_ALL_INPUTS_2 {1}] [get_bd_cells conveyor_gpio]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master "/ps7/M_AXI_GP0" Clk_master "/ps7/FCLK_CLK0 (100 MHz)" Clk_slave "/ps7/FCLK_CLK0 (100 MHz)" Clk_xbar "/ps7/FCLK_CLK0 (100 MHz)"} [get_bd_intf_pins conveyor_gpio/S_AXI]
create_bd_cell -type module -reference conveyor_encoder encoder_0
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins encoder_0/clk]
connect_bd_net [get_bd_pins conveyor_gpio/s_axi_aresetn] [get_bd_pins encoder_0/resetn]
connect_bd_net [get_bd_pins conveyor_gpio/gpio_io_o] [get_bd_pins encoder_0/cmd]
connect_bd_net [get_bd_pins encoder_0/data] [get_bd_pins conveyor_gpio/gpio2_io_i]
foreach pin {encoder_a encoder_b} {
 create_bd_port -dir I $pin
 connect_bd_net [get_bd_ports $pin] [get_bd_pins encoder_0/$pin]
}

# AXI automation connects associated clocks/resets. Catch missing ones at validation.
assign_bd_address
validate_bd_design
foreach space {Data_MM2S Data_S2MM} {
 set segments [get_bd_addr_segs -of_objects [get_bd_addr_spaces axi_dma_0/$space]]
 if {![string match *HP0_DDR_LOWOCM* $segments]} {error "DMA $space has no mapped DDR address segment"}
 puts "KM1_DMA_${space}_DDR=$segments"
}
save_bd_design
generate_target all [get_files design_1.bd]
make_wrapper -files [get_files design_1.bd] -top
add_files -norecurse build/vivado_project/vision.gen/sources_1/bd/design_1/hdl/design_1_wrapper.v
set_property top design_1_wrapper [current_fileset]
# Use serial child launches after observing a Tcl-app startup failure during
# a parallel Windows build; also keep memory use bounded.
launch_runs impl_1 -to_step write_bitstream -jobs 1
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Implementation failed"}
open_run impl_1
report_timing_summary -file report/timing.rpt
report_utilization -file report/utilization.rpt
report_drc -file report/drc.rpt
set paths [get_timing_paths -delay_type max -max_paths 1]
if {[llength $paths]==0 || [get_property SLACK [lindex $paths 0]]<0} {error "Timing not met"}
file copy -force build/vivado_project/vision.runs/impl_1/design_1_wrapper.bit board/vision.bit
file copy -force build/vivado_project/vision.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh board/vision.hwh
open_bd_design [get_files design_1.bd]
write_bd_tcl -force build/generated_bd.tcl
exit
